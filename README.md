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
* **[Licence](#licence)** — not yet settled; see the pointer.

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
| Silent | arithmetic on a **text** column computes on dictionary codes, with no guard (5 of 6 operators) |
| Silent | `group` drops a key whose measure is entirely NULL |
| Mismatch | `to_numpy()` reads an integer NULL as `0`; use `to_pandas()` |
| Driver | GPU sort needs a power-of-two valid-row count, and refuses `int64`/`float64` keys |

Two of those — text arithmetic and the all-NULL measure — return
correct-shaped wrong numbers with no exception. They are documented, not fixed;
fixing them means changing the arithmetic and grouping paths, which is outside
what this repository is currently allowed to change.

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

Last clean-clone run, 2026-10-04, commit `f2af608`: **653 passed, 4 skipped, 0
failed**. See the audit note in the repository history for the two failures that
run originally reported and what was done about each.

**[.github/workflows/ci.yml](.github/workflows/ci.yml)** has three Windows jobs:

1. **build the wheel, install it into a clean venv, import it** — gating.
2. **223 tests that need nothing but this repo** — gating. The other test files
   build their kernel through `from builder import MAIN`, and `builder` lives in
   a separate repository whose published `main` is 11 commits behind what this
   tree needs, so a job that fetched it would fail. That job is present,
   non-gating, and says so in its own step name.
3. **the whole suite** — non-gating for the reason above.

There is **no Linux CI job**: only a Windows `.dll` is committed, and with the
native library disabled the suite is 10 failed / 627 passed, so the native
lanes are load-bearing. A Linux job would be red on its first run, and a CI file
that fails is worse than no CI.

---

## Licence

**Not yet decided.** A licence proposal is being produced separately; this
repository currently carries `AGPL-3.0-only` SPDX headers and
`license = { text = "AGPL-3.0-only" }` in `pyproject.toml`, and **no licence
file is present in the tree**. Treat the licensing as unsettled until a
`LICENSE` file appears at the repository root.

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