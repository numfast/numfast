# NumFast

**NumPy-shaped API, C-like cost, any hardware.**

NumFast is a columnar compute engine with a small, NumPy-and-pandas-shaped
surface. You hand it columns; it hands you columns. Integer columns stay
`int32` and are computed exactly, on the CPU or on the GPU through WebGPU, and
the plan it builds is one call through one planner.

It is **not** faster than Polars, and this README will not pretend otherwise.
See [Where it wins and where it loses](#where-it-wins-and-where-it-loses) —
there is a table of the queries where NumFast is 7× *slower* than DuckDB, and it
is not a short table.

* **[Quickstart](#quickstart)** — a program you can run in four lines.
* **[What it supports](#what-it-supports-honestly)** — 34 of 43 ClickBench queries.
* **[Known limitations](#known-limitations)** — what does not work, tagged by
  how sure we are. Read it before you rely on anything.
* **[Benchmarks](#benchmarks)** — every number, with its command and its date.
* **[Licence](#licence)** — AGPL-3.0-only. `LICENSE` and `NOTICE` are in the tree
  and reach the built artefacts.

---

## Quickstart

Requires Python 3.11+ and numpy. `pandas` and `pyarrow` are optional.

```bash
export PYTHONPATH="C:/path/to/numfast/src"      # or: pip install numfast
python examples/quickstart.py
```

The whole program:

```python
import pandas as pd
import numfast as nf

orders = pd.DataFrame({
    "region":  ["emea", "apac", "emea", "amer", "apac", "emea"],
    "revenue": [120.0, 80.0, 240.5, 60.0, 95.5, 310.0],
})

result = (nf.from_pandas(orders)              # or nf.from_numpy(...)
          .query()                            # start a lazy chain
          .group("region", {"revenue": ("sum", "count")})
          .sort("revenue.sum", desc=True)
          .compile())                         # -> a numfast Table

print(result.to_pandas())
```

Output:

```
region  revenue.sum  revenue.count
emea        670.5              3
apac        175.5              2
amer         60.0              1
```

The same program with an answer checked against pandas is
[`examples/quickstart.py`](examples/quickstart.py). It runs on a clean clone
with `PYTHONPATH=<repo>/src` and needs nothing from this repository's
development dependencies.

The vocabulary is the whole of it: `query` → `filter` / `derive` / `sort` /
`limit` / `group` → `compile`, with expressions built from `app.c("col")` and
`app.c("col") > 0`. There is no query planner to learn and no index to declare.

---

## What it supports, honestly

**34 of the 43 ClickBench queries run and return a verified answer. 9 do not.**

| | Count | Detail |
|---|---|---|
| Supported | **34** | Correctness verified per query as `EXACT` (integers bit-exact), `EXACT+TOL` (floats inside `specs/core/conformance-profile.toml`), `MULTISET`, or `TIE-MULTISET` — the weaker labels are weaker, and are not rounded up to "exact" |
| Unsupported | **9** | 8 blocked by **BIGINT narrowing**, 1 (Q40) blocked by a **missing conditional/CASE primitive** |

The BIGINT narrowing is the point of the whole design, not an oversight: logical
values are `int32`, and ClickBench's `UserID` / `WatchID` / `URLHash` values run
past 2³¹. NumFast **raises** on those keys and literals rather than narrowing
them silently. Nine queries cost you; a wrong answer on any table would cost
more.

Per-query detail, commands and dates: **[BENCHMARKS.md](BENCHMARKS.md)**.

---

## Where it wins and where it loses

Measured against DuckDB on ClickBench @ ~1M rows. `ratio` is NumFast's total
time over DuckDB's — **above 1.0 means NumFast is slower**. Source artefact,
command and date in [BENCHMARKS.md](BENCHMARKS.md) §1.

### Where it loses

| Query | Shape | NumFast | DuckDB | NumFast slower by |
|---|---|---|---|---|
| Q30 | group-by-distinct | 231.25 ms | 31.28 ms | **7.39×** |
| Q36 | 4-column composite group | 167.47 ms | 23.69 ms | **7.07×** |
| Q9 | high-cardinality group | 165.72 ms | 37.14 ms | **4.46×** |
| Q5 | distinct count | 78.75 ms | 19.21 ms | **4.10×** |
| Q18 | large aggregate | 122.01 ms | 32.25 ms | **3.78×** |
| Q10 | range scan + aggregate | 178.59 ms | 48.54 ms | **3.68×** |
| Q17 | large aggregate | 122.01 ms | 40.71 ms | **3.00×** |
| Q22 | medium aggregate | 97.06 ms | 37.46 ms | **2.59×** |

The pattern is consistent: **high-cardinality grouping and distinct-count
shapes are where NumFast is behind**, and behind by several times, not by a few
percent. If your workload is one of those, DuckDB is the better tool today and
this README will not dress that up.

### Where it wins

Q28 133.66×, Q43 86.04×, Q39 55.80×, Q8 28.51×, Q15 20.30×.

Read those honestly: they are the **small, aggregate-shaped** queries, where
DuckDB pays a fixed per-query overhead that a single planner call does not.
The honest claim is *NumFast has a very low fixed cost*, not *NumFast is 86×
faster than DuckDB*.

### The GPU

The GPU path runs through `wgpu-py` (WebGPU over Vulkan/DX12 — no CUDA). Its
recorded measurements at N = 10M were **slower than the CPU path** on the
hardware tested. What the GPU buys today is **parity and residency**: the same
permutation, the same sums, with data staying resident. **No GPU speedup is
claimed.** Only a Windows binary is committed; there is no Linux build.

---

## What NumFast is not

* **Not a Polars replacement.** Polars has a query language, a multi-threaded
  executor, streaming CSV and Parquet readers, and years of production use.
  NumFast has a `int32`-exact kernel layer, a WebGPU path, and a 30-name
  consumer vocabulary. Different tools.
* **The Rust layer is kernels, not an executor.** `numfast_native.dll` is a set
  of column-at-a-time compute kernels reached over `ctypes`. It does not
  schedule, does not parallelise across queries, does not stream, and does not
  own memory. The executor is the Python planner and the CPU driver.
* **Not a drop-in for pandas.** Semantics are 3VL and validity-aware;
  `to_numpy()` and `to_pandas()` can disagree on NULL (see limitations §6).
* **Not faster than Polars.** See the table above; it has not been measured
  against Polars in this repository at all, and an unmeasured claim is not a
  claim.

---

## Known limitations

Read **[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)** before relying on
anything. Every entry is tagged **PROVEN / MEASURED / INFERENCE / HYPOTHESIS /
UNKNOWN** and carries a reproduction. The short version:

| | Limitation |
|---|---|
| Absent | `window` (rolling) — silent exactness loss on int64 > 2⁵³, fix is FROZEN |
| Absent | `or_` — would silently drop rows; refused loudly instead |
| Absent | `join`, `replace` — no consumer-facing spelling |
| Absent | `fill_null` — not chunkable over the current layout |
| Refused | `group` on a NULL key — a NULL group is *absent*, not wrong, and cannot be told apart from an answer |
| Refused | arithmetic on a **text** column — it computed on dictionary codes (ranks, not values); all six operators now refuse, and `str_len()` is the honest route |
| Silent | `group` drops a key whose measure is entirely NULL |
| Mismatch | `to_numpy()` reads an integer NULL as `0`; use `to_pandas()` |
| Driver | GPU sort needs a power-of-two valid-row count, and refuses `int64`/`float64` keys |

One of those — the all-NULL measure — still returns a correct-shaped wrong
answer with no exception. It is documented, not fixed; fixing it means changing
the grouping path, which is outside what this repository is currently allowed to
change.

---

## Known architectural limits

Two facts about the architecture, stated as **measurements**, not as apologies.
Source: `develop/audit_foundation/ORIGIN_PIPELINE_BENCH.md` in the NumFast
dev-env repository — origin pipeline benchmark, engine `4fc9914`, RTX 2060 /
Xeon E5-2698 v4, median of 31, cycles from `QueryThreadCycleTime`.

> **Provenance correction, 2026-10-05.** That audit states it verified
> `git diff --name-only 4fc9914..1a3caa4 -- src/` returns 0 files, so that
> "every number below was measured against a byte-identical Python engine". That
> check was true when written and is **false now**: at `e415fd1`,
> `git diff --name-only 4fc9914..e415fd1 -- src/` returns **11 files**, of which
> 8 are engine sources (`Compute/Fused/_lib/fused.py`,
> `Drivers/CPU/_lib/native_cpu.py`, `Relational/Join/_lib/native{,_i64}.py`,
> `Runtime/Planner/_lib/calibrate.py`, `Semantic/TableExpr/_lib/{chain,expr,plan}.py`).
> Those changes are fork-root path resolution and facade node lowering; none of
> them touch the window-composition or GPU-execution path these two limits
> describe, so the measurements below are still the best available account of
> that path. But they were taken against `4fc9914`, not against this tree, and
> this file should not imply otherwise. The audit record itself is unchanged and
> is reported, not rewritten.

### There are no cheap strided or windowed views

A window of width *W* is **not a view**. The only window composition the
expression language offers is *W* × `shift`, and each `shift` writes a
**full-length buffer**: at *n* = 1 048 576, *W* = 64 that is **64 buffers,
512 MiB**, followed by a second full materialisation of the (M, W) matrix
(another 512 MiB) — **2.00× the memory of the answer numpy gives for one
512 MiB copy.** numpy's `sliding_window_view` is a **view**: 64 680 cycles at
*n* = 1 048 576, *W* = 64, flat in both *n* and *W* — ~0.06 cycles/row, **zero
bytes written**. The gap is not the kernels; the kernels are within **1.76×** of
numpy on the same pipeline. **MATERIALISATION is 69.6 %–89 % of what the user
pays.**

### There is no resident GPU execution

`gpu_execute` is a **batch execute with per-node readback**: an 8-node graph
returned **9 host buffers** for a 0.2 MiB input — **11× byte amplification,
zero residency**. The resident primitives (`r_upload`, `r_alloc`, `r_gather`,
`r_download`) exist and are not what `gpu_execute` uses. Through the consumer
surface the GPU is unreachable at all: `Chain.compile()` is hardwired to
`cpu_execute` and `Chain.explain()` prints `backend=n/a`.

The one regime where the GPU reaches parity is **regime B**: the index built
**once**, kept resident, and **never shuffled** — 0.915× prep/consume at
*n* = 1 048 576, and **1.0×–1.5×** elsewhere. Add the per-epoch shuffle every
training loop actually does and the same index costs **13.2×** prep/consume;
build it per batch (regime A) and it is **158–168×**. The index buffer is also
**64× the data it indexes** — at *W* = 64, 4 B per window element, exactly as
large as the f32 window it produces — and the first epoch pays **319 ms** before
a single row is consumed.

So: **the GPU is at parity in a resident-index regime and loses once a shuffle
is required.** Both limits are properties of the buffer layout and the executor
boundary, not of the compute kernels.

---

## Benchmarks

Every performance number in this repository, with the command that produces it,
the environment, the date, and whether it can still be re-run:
**[BENCHMARKS.md](BENCHMARKS.md)**.

Summary of what is and is not reproducible:

| Artefact | Verdict |
|---|---|
| `tests/heavy/bench_clickbench_43.json` | **Reproducible in principle.** Script committed; needs `hits_1m.parquet`, which is not on this machine |
| `tests/heavy/bench_h2o_100M_cpu_gpu.json` | **Historical.** No producing script exists |
| `tests/heavy/bench_h2o_gpu_10M.json`, `..._scaled_gpu_10M.json` | **Historical.** Scripts committed; H2O CSV not on this machine |
| `develop/benchmarks_public/` (5 suites) | **Historical and unreproducible.** All nine source artefacts were deleted; each `summary.json` now says so in a machine-readable `provenance` field |

Two claims were **removed** during the 2026-10-04 audit because the artefacts
contradict them: "ClickBench 8/43 supported" (the tree measures 34/43) and
"Q30 affine 43.02× faster" (an unconfirmed prototype, not reproduced by current
code).

No wall-clock number is quoted anywhere in this repository that was not produced
by a command named in `BENCHMARKS.md`.

---

## Tests and CI

```bash
export PYTHONPATH="C:/path/to/numfast/src;C:/path/to/app-builder"
python -m pytest tests/ -q              # fast suite only; heavy needs -m heavy
```

Last clean-clone run — `git clone` of this repository into an empty directory,
2026-10-05, commit `e415fd1` (Windows, Python 3.14.6):

```
740 passed, 5 skipped, 0 failed in 171.49s (0:02:51)
```

The 5 skips, and why each one skips on a fresh clone:

| Skipped | Reason |
|---|---|
| `test_ops_null_pattern.py` (the wall-clock test) | `heavy` — run with `pytest -m heavy` |
| `test_ops_text.py`, `test_ops_text_affix.py` | WASM artefact absent (the `wasm32` build is not committed) |
| `test_packaging_adapters.py` | "wheel venv not built here" — needs `python -m build` to have run first |
| `test_rng_gate.py` | reference file lives in `scratch/`, which is gitignored |

So on a clean clone **two of these are not optional**: the WASM tests and the
RNG-gate test never run without artefacts that are not in the tree. They run in
a developer's checkout and they skip in CI. That is a real coverage hole and it
is named here rather than hidden behind a green line.

**[.github/workflows/ci.yml](.github/workflows/ci.yml)** has three Windows jobs:

1. **build the wheel, install it into a clean venv, import it** — gating.
2. **223 tests that need nothing but this repo** — gating. The other test files
   build their kernel through `from builder import MAIN`, and `builder` lives in
   a separate repository whose published `main` is 11 commits behind what this
   tree needs, so a job that fetched it would fail. That job is present,
   non-gating, and says so in its own step name. 223 of the 745 collected tests
   is the gating coverage — both numbers measured on a clean clone at `e415fd1`
   on 2026-10-05; the rest is exercised by developers with a Builder checkout.
3. **the whole suite** — non-gating for the reason above.

There is **no Linux CI job**: only a Windows `.dll` is committed, and with the
native library disabled the suite is 10 failed / 627 passed, so the native
lanes are load-bearing. A Linux job would be red on its first run, and a CI file
that fails is worse than no CI.

---

## Licence

**AGPL-3.0-only.** The verbatim licence text is `LICENSE` (35 184 bytes) at the
repository root; `NOTICE` records the third-party attributions; and
`COMMERCIAL-LICENCE.md` describes the commercial alternative. Every source file
carries an `AGPL-3.0-only` SPDX header, and `pyproject.toml` declares
`license = "AGPL-3.0-only"` with `license-files = ["LICENSE", "NOTICE"]`.

Both files reach the built artefacts, verified on the 0.2.1 release candidate:
`LICENSE` and `NOTICE` land in `numfast-0.2.1.dist-info/licenses/` in **both**
wheel flavours, byte-identical to the tree; `METADATA` carries
`License-Expression: AGPL-3.0-only` and a `License-File:` line for each; and both
are present in the sdist, byte-identical.

Because the wheel conveys `numfast_native.dll` — object code — AGPL-3.0 §6 is
satisfied by shipping the Corresponding Source with it: the wheel carries all
**47** `.rs` files plus `Cargo.toml`, `Cargo.lock`, `.cargo/config.toml` and the
two link shims under `numfast/_corresp_src/numfast-native/`, every one
byte-identical to the tree, with a `SHA256SUMS` over exactly those bytes. See
[CORRESPONDING-SOURCE.md](CORRESPONDING-SOURCE.md).

---

## Layout

```
src/Core/            canonical dtypes, the public Series/Table types
src/Semantic/IR/     the semantic IR: one node per column
src/Runtime/         planner + runtime  (FROZEN)
src/Drivers/CPU/     the NumPy-backed reference driver (FROZEN)
src/Drivers/GPU/     the wgpu-py / WGSL driver
src/Storage/         schema, dictionary, npz-staging persistence
src/Semantic/TableExpr/  the consumer facade: query/filter/derive/group/sort
specs/               the frozen specs: IR, planner cost, drivers, storage, method
tests/fast/          the default suite;  tests/heavy/  needs -m heavy
examples/quickstart.py
```

`specs/core/10-benchmark-methodology.md` is the rulebook for how a number gets
into this repository.