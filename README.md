# NumFast

NumFast is a columnar compute engine for Python with a small, NumPy- and
pandas-shaped surface. You hand it columns; it hands you columns back. Integer
columns are computed exactly in `int32`, NULLs are carried explicitly rather
than as `NaN`, and a query is one lazy chain that a single planner call turns
into one execution graph.

**Version 0.2.1.** AGPL-3.0-only. Python 3.11+. The native compute kernels are a
Rust library that currently ships for Windows x86-64 only.

Read this page to decide whether it is useful to you in thirty seconds:

* **What it does well, and where it is slower than DuckDB** — measured, per
  query, below. It loses on high-cardinality grouping by several times. That is
  not a footnote, it is the main performance fact.
* **What works today** — 34 of 43 ClickBench queries run and return a verified
  answer. 9 do not, and the reasons are named.
* **What is absent** — `window`, `or_`, `join` and `replace` are not in the API.
  The reasons are named too.
* **What the GPU does** — 15 of 33 operations, measured on one RTX 2060 over
  Vulkan. The other 18 run on the CPU. No GPU speedup is claimed anywhere.
* **[Known limitations](KNOWN_LIMITATIONS.md)** — every entry tagged by how sure
  we are, with a reproduction. Read it before you rely on anything.
* **[Licence](#licence)** — AGPL-3.0-only, with the Corresponding Source inside
  the wheel. There is no paid tier of the AGPL.

---

## Install

```bash
pip install numfast            # the engine
pip install "numfast[pandas]"  # + pandas adapters
pip install "numfast[arrow]"   # + pyarrow adapters
pip install "numfast[gpu]"     # + wgpu, required to execute on the GPU path
```

`numpy>=1.24` is the only hard dependency.

Two wheels are published at the same version:

| Wheel | Tag | Carries the native library | Who gets it |
|---|---|---|---|
| `numfast-0.2.1-py3-none-win_amd64.whl` | `py3-none-win_amd64` | yes | Windows x86-64: full engine, native path live |
| `numfast-0.2.1-py3-none-any.whl` | `py3-none-any` | no | Linux, macOS: full engine, native path absent |

**Linux and macOS get the complete engine and no native library.** That is
degraded, not broken, and it is visible rather than latent:

```python
>>> import numfast as nf
>>> nf.native_info()
{'disabled': False, 'dll': None, 'dll_exists': False}
```

`numfast/_native/` is simply not there, and every Rust-backed call falls back to
the NumPy CPU path. It is not a binary that fails to load at run time. Set
`NUMFAST_NATIVE_DISABLE=1` to force the same path off on Windows, and
`NUMFAST_NATIVE_DLL` to point at a locally built binary.

Nothing in the distribution is a CPython extension module: the `.dll` is loaded
through `ctypes`, so the implementation tag stays `py3` and neither wheel is
pinned to one interpreter version.

To run from a clone instead of installing:

```bash
export PYTHONPATH="<repo>/src"
python examples/quickstart.py
```

---

## Quick start

The smallest program that produces a real result — group, sort, and check the
answer against pandas:

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

```
region  revenue.sum  revenue.count
  emea        670.5              3
  apac        175.5              2
  amer         60.0              1
```

[`examples/quickstart.py`](examples/quickstart.py) is the same program with the
pandas comparison and the assertion inside it. It runs on a clean clone with only
`<repo>/src` on `PYTHONPATH`, needs nothing else from this repository, and its
output is the block above.

The vocabulary is the whole of it: `query` → `filter` / `derive` / `group` /
`sort` / `limit` / `reduce` → `compile`, with expressions built from
`app.c("col")`, `app.c("col") > 0`, `.and_()`, `.isin([...])`. There is no query
language to learn and no index to declare.

More examples, each one runnable and each one with its real output:
**[docs/EXAMPLES.md](docs/EXAMPLES.md)**.

---

## What works today

**34 of the 43 ClickBench queries run and return a verified answer. 9 do not.**

| | Count | Detail |
|---|---|---|
| Supported | **34** | Verified per query as `EXACT` (26), `TIE-MULTISET` (4), `EXACT+TOL` (2) or `MULTISET` (2). The weaker labels are weaker and are not rounded up to "exact". |
| Unsupported | **9** | 8 blocked by **BIGINT narrowing**, 1 (Q40) by a **missing conditional/CASE primitive** |

The BIGINT narrowing is the design, not an oversight: logical values are `int32`,
and ClickBench's `UserID` / `WatchID` / `URLHash` values run past 2³¹. NumFast
**raises** on those keys and literals rather than narrowing them silently. Nine
queries cost you; a wrong answer on any table would cost more.

Per-query detail, commands, environment and dates:
**[BENCHMARKS.md](BENCHMARKS.md)**.

---

## Capabilities

**The consumer surface is 44 names, and that is all of it.** It is a new layer —
first released in 0.2.1 — and it is small on purpose.

| Group | Count | Names |
|---|---|---|
| Boundary adapters | 6 | `app`, `from_numpy`, `from_arrow`, `to_numpy`, `to_pandas`, `to_arrow` |
| App and column reference | 5 | `c`, `capabilities`, `open_stream`, `query`, `schema` |
| Chain | 10 | `compile`, `derive`, `explain`, `filter`, `group`, `jobs`, `limit`, `nrows`, `reduce`, `sort` |
| Expression | 23 | `add` `and_` `cumsum` `eq` `ge` `gt` `is_null` `isin` `le` `lt` `mod` `mul` `ne` `not_` `pow` `shift` `str_contains` `str_endswith` `str_eq` `str_len` `str_startswith` `sub` `truediv` |

The count is asserted in the test suite, not maintained by hand.

`nf.from_pandas`, `nf.Series`, `nf.Table`, `nf.native_info()`,
`nf.gpu_capabilities()`, `nf.get_kernel()` and `nf.__version__` are also part of
`import numfast`, and every example in this repository uses them. They are simply
not entries in the 44-name v0 registry, which governs the *consumer* surface: the
pandas entry point, the two public types, and the three disclosure functions.

**Absent from the surface, deliberately:**

| Absent | Why |
|---|---|
| `window` (rolling) | the lowering loses exactness silently on int64 above 2⁵³. An operation that can only be wrong quietly is not shipped as if it were right. The fix is in frozen code. |
| `or_` | `ir_mask(..., 'or')` AND-s the two operands' validity, so the filter drops rows Kleene logic keeps. `Expr.or_` exists and **raises**. Use `and_`, or run the two filters. |
| `join` | the join primitives exist in the engine and are reachable through the kernel-level API. There is no consumer-facing spelling. |
| `replace` | no method on `app` or a chain. Use `derive` with `isin` / `str_eq`. |
| `fill_null` | filling is not chunkable over the current layout; it would be a row-count trap. |

**"This refuses rather than lies" is a real feature, and here is the list of
refusals you can rely on:**

| Situation | What happens |
|---|---|
| `group` on a column with NULL rows | raises, names the column and the count, and tells you to filter first. A NULL group would be *absent*, not wrong, and absence cannot be told apart from an answer. |
| `group` with more than one measure column | raises and names the offending keys. Aggregate one measure column per call. |
| arithmetic on a TEXT column (`+ - * / % **`) | raises before any node is emitted. The DAG holds a dictionary *code* vector, and codes are ranks, not values. |
| ordering a TEXT column (`< <= > >=`) | `filter` raises: ordering on text needs a collation the engine does not define. Use `str_eq` / `isin` / `str_len`. |
| an `int64` value outside `int32` in an `int32` column | raises rather than wrapping. |
| a BIGINT key or literal (ClickBench's `UserID`, `URLHash`, …) | raises rather than narrowing. |
| `backend='gpu'` on a graph using one of the 18 CPU-only ops | `RuntimeError` naming `op:<name>`. There is no silent CPU fallback. |
| a fused GPU indicator plan with a parameter key no operation consumes | raises, naming the output, the key, the op and that op's accepted keys. Dropping the key silently used to re-root the body to the wrong input. |

Full surface, with signatures and worked output:
**[docs/API.md](docs/API.md)**.

---

## Architecture in one page

Orientation, not a specification. **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**
is the longer version.

```
  you
   │  query() → filter/derive/group/sort/limit → compile()
   ▼
┌─────────────────────────────────────────────────────────────┐
│ Semantic/IR        one node per column, one DAG per chain    │
│ Semantic/TableExpr the consumer facade; 44 public names      │
├─────────────────────────────────────────────────────────────┤
│ Runtime/Planner    capability query + measured cost model    │
│                    → ONE execution graph per chain           │
│ Runtime/Runtime    job → IR → optimise → plan → execute     │
├─────────────────────────────────────────────────────────────┤
│ Drivers/CPU        NumPy-backed reference executor (FROZEN)  │
│ Drivers/GPU        wgpu-py / WGSL, 15 of 33 ops             │
│ Drivers/GroupedHash, GroupedHashMT                           │
├─────────────────────────────────────────────────────────────┤
│ Storage            schema, dictionary coding, NFS            │
│ Core               canonical dtypes, Series / Table          │
└─────────────────────────────────────────────────────────────┘
        │ ctypes
        ▼
  numfast-native/     Rust kernels: sort, join, group-by, map,
                      carry, segmented reduce, SSSP, cost travel
```

* **The IR** is a list of jobs, `{op, inputs, params, out}`, one node per
  column. Expressions lower to nodes; a chain holds jobs, never a second IR.
* **Extensions.** The engine is 29 Builder Extensions, each a folder with a
  `.toml` manifest, a `.py` entry point and a `_lib/` implementation. The
  manifest declares its aliases and dependencies; `setup.py` vendors all 29 into
  the wheel as `numfast/_ext/` — **128 files, byte-identical to the tree**. A
  manifest that names a directory which does not exist is a hard build error.
* **The CPU driver** is the reference executor, NumPy-backed, and is FROZEN.
  Several limitations below are consequences of that, not of missing effort.
* **The native Rust layer is kernels, not an executor.** `numfast_native.dll` is
  a set of column-at-a-time compute kernels reached over `ctypes`. It does not
  schedule, does not stream, does not own memory. The executor is the Python
  planner and the CPU driver.
* **The GPU split** is by operation, declared at run time: `gpu_capabilities()`
  reports **15 of 33** operations, with a `chunkable` hint and dispatch limits
  per operation. The other **18** are CPU-only and say so.
* **The WASM path** compiles the same Rust kernels to
  `@numfast/kernels` (npm): a portable kernel library, **86 function exports, one
  memory, zero imports, 17 typed wrappers**. It is **not a compute core** — there
  is no executor, no graph runtime and no allocator inside the `.wasm`.
* **Where the semantic contract lives.** Three-valued logic and NULL handling are
  specified in `specs/core/`, pinned by the test suite, and — where the two
  disagree — the test wins and the spec is corrected.

---

## Performance

Only measured claims. Every number below was recomputed from the artefact named
beside it; the command, environment and date are in
**[BENCHMARKS.md](BENCHMARKS.md)**, which is the only place a performance claim
is quotable from.

### Against DuckDB, ClickBench at ~1M rows

`ratio = (compile + execute) / duckdb` from `tests/heavy/bench_clickbench_43.json`.
**Above 1.0 means NumFast is slower.**

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

That is the complete list of losses above 2.5×, and it is not short.
High-cardinality grouping and distinct-count shapes are where NumFast is behind,
by several times rather than a few percent. If your workload is one of those,
DuckDB is the better tool today.

Where NumFast is faster: Q28 133.66×, Q43 86.04×, Q39 55.80×, Q8 28.51×,
Q15 20.30×. Read those honestly — they are the small, aggregate-shaped queries
where DuckDB pays a fixed per-query overhead that one planner call does not. The
claim is *NumFast has a very low fixed cost*, not *NumFast is 86× faster than
DuckDB*.

### The GPU

The GPU path runs through `wgpu-py` 0.31.1 (WebGPU over Vulkan/DX12; no CUDA).
All GPU cost measurements in this repository were taken on **one** RTX 2060
driving Vulkan.

* **15 of 33** operations have a GPU implementation; **18 are CPU-only** and are
  listed by `nf.app().gpu_capabilities()`.
* The recorded GPU measurements at the sizes measured were **slower than the CPU
  path**. What the GPU buys today is **parity and residency**: the same
  permutation, the same sums. **No GPU speedup is claimed.**
* Asking for `backend='gpu'` on a graph that uses a CPU-only operation raises
  rather than falling back.

### Reproducibility of every artefact

| Artefact | Verdict |
|---|---|
| `tests/heavy/bench_clickbench_43.json` | **Reproducible in principle.** Script committed; needs `hits_1m.parquet`, which is not on this machine, so the numbers are as recorded, not re-measured here. |
| `tests/heavy/bench_h2o_100M_cpu_gpu.json` | **Historical, unreproducible.** No producing script exists anywhere in the tree. |
| `tests/heavy/bench_h2o_gpu_10M.json`, `..._scaled_gpu_10M.json` | **Historical.** Scripts committed; the H2O CSV is not on this machine. |
| `develop/benchmarks_public/` (5 suites) | **Historical and unreproducible.** All nine source artefacts were deleted; each `summary.json` says so in a machine-readable `provenance` field. |

The H2O files still record one thing that remains meaningful: CPU/GPU agreement
(`cpu_gpu_match: true`, `cpu_gpu_maxdiff: 0`). Treat every millisecond in them as
unattributable — no date, no Python version, no CPU model is recorded in them.

Two claims were **removed** during the 2026-10-04 audit because the artefacts
contradict them: "ClickBench 8/43 supported" (this tree measures 34/43) and
"Q30 affine 43.02× faster" (an unconfirmed prototype the current code does not
reproduce). No wall-clock number appears in this repository that was not produced
by a command named in `BENCHMARKS.md`.

---

## Known limitations

**[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)** is the complete list. Every entry
is tagged **PROVEN / MEASURED / INFERENCE / HYPOTHESIS / UNKNOWN** and carries a
reproduction; a tag without a reproduction is not a claim. The short version:

| | Limitation |
|---|---|
| Absent | `window` (rolling), `or_`, `join`, `replace`, `fill_null` — reasons above |
| Refused | `group` on a NULL key — a NULL group is *absent*, not wrong, and cannot be told apart from an answer |
| Refused | arithmetic on a TEXT column — it computed on dictionary codes (ranks, not values); all six operators now refuse, and `str_len()` is the honest route |
| Silent | `group` drops a key whose measure is entirely NULL. Documented, not fixed: fixing it means changing the grouping path. |
| Mismatch | `to_numpy()` reads an integer NULL as `0`; use `to_pandas()` |
| Refused | `group` with more than one measure column raises |
| Driver | GPU sort needs a power-of-two valid-row count, and refuses `int64`/`float64` keys |

Two architectural limits are stated as measurements rather than apologies. They
come from an origin-pipeline audit taken against engine commit `4fc9914`, and
they are reproduced with their provenance correction in
[BENCHMARKS.md §5](BENCHMARKS.md):

* **There are no cheap strided or windowed views.** A window of width *W* is
  composed as *W* × `shift`, and every `shift` writes a full-length buffer. At
  *n* = 1 048 576, *W* = 64 that is 64 buffers of 512 MiB plus a second full
  materialisation of the (M, W) matrix — **2.00× the memory of the answer numpy
  gives for one 512 MiB copy**, where `numpy.sliding_window_view` is a view that
  writes zero bytes. Materialisation is 69.6 %–89 % of what the caller pays.
* **There is no resident GPU execution.** `gpu_execute` is a batch execute with
  per-node read-back: an 8-node graph on a 0.2 MiB input returned 9 host
  buffers — **11× byte amplification, zero residency**. Through the consumer
  surface the GPU is unreachable at all: `Chain.compile()` runs the CPU path and
  `Chain.explain()` prints `backend=n/a`.

**Provenance correction, stated because it matters.** That audit recorded that it
had verified `git diff --name-only 4fc9914..1a3caa4 -- src/` returns zero files,
so that every number was measured against a byte-identical Python engine. That
check was true when written and is **false now**: at `597d2ec`,
`git diff --name-only 4fc9914..597d2ec -- src/` returns **13 files** — eight
engine sources (`Compute/Fused/_lib/fused.py`, `Drivers/CPU/_lib/native_cpu.py`,
`Relational/Join/_lib/native{,_i64}.py`, `Runtime/Planner/_lib/calibrate.py`,
`Semantic/TableExpr/_lib/{chain,expr,plan}.py`) and five generated packaging
metadata files under `src/numfast.egg-info/`. None of them touch the
window-composition or GPU-execution path these two limits describe, so the
measurements remain the best available account of that path — but they were taken
against `4fc9914`, not against this tree. The audit record itself is unchanged and
is reported, not rewritten.

---

## Licence

**AGPL-3.0-only.** The verbatim text is [`LICENSE`](LICENSE); third-party
attributions are in [`NOTICE`](NOTICE). Every source file carries an
`AGPL-3.0-only` SPDX header, and `pyproject.toml` declares
`license = "AGPL-3.0-only"` with `license-files = ["LICENSE", "NOTICE"]`.

**Corresponding Source ships in the wheel.** Because the wheel conveys
`numfast_native.dll` — object code — AGPL-3.0 §6 is discharged by construction,
not by a written offer: the wheel carries all **47** `.rs` files plus `Cargo.toml`,
`Cargo.lock`, `.cargo/config.toml` and the two link shims under
`numfast/_corresp_src/numfast-native/`, with a `SHA256SUMS` (52 entries) over
exactly those bytes. Verified on the 0.2.1 release candidate: 52 of 52 checksums
match in the wheel and again in the installed package, and `LICENSE` + `NOTICE`
land in `numfast-0.2.1.dist-info/licenses/` in both wheel flavours with
`License-Expression: AGPL-3.0-only` in `METADATA`. Details:
[`CORRESPONDING-SOURCE.md`](CORRESPONDING-SOURCE.md).

**There is no "paid AGPL".** [`COMMERCIAL-LICENCE.md`](COMMERCIAL-LICENCE.md)
announces that a separate, proprietary commercial licence is planned. **Its terms
do not exist yet** — they are not written, not priced and not an offer. The AGPL
grant is free, unconditional and complete, and nothing in this repository adds a
fee, a field-of-use restriction or a non-commercial condition to it. A closed-source
deployment is only possible under a separate licence from the copyright holder,
and that document is not here yet.

**Support and consulting are available now** and need no further paperwork
(AGPL-3.0 §4 permits charging for warranty, support or indemnity).

Security policy: [`SECURITY.md`](SECURITY.md).

---

## Repository layout

```
src/Core/            canonical dtypes, the public Series / Table types
src/Semantic/IR/     the semantic IR: one node per column
src/Semantic/TableExpr/  the consumer facade (44 public names)
src/Runtime/         planner + runtime  (FROZEN)
src/Drivers/CPU/     the NumPy-backed reference driver (FROZEN)
src/Drivers/GPU/     the wgpu-py / WGSL driver
src/Relational/      join, group-by, sort, dictionary, lookup, segmented
src/Storage/         schema, dictionary coding, NFS
src/numfast/         the importable package and its vendored builder
numfast-native/      the Rust kernel crate + the @numfast/kernels TS package
docs/                user-facing documentation, and the internal design record
specs/               the specs. Historical: see docs/README.md before reading
tests/fast/          the default suite;  tests/heavy/  needs -m heavy
examples/quickstart.py
```

`docs/README.md` says which documents describe the current tree and which are
the internal design record.

---

## Tests and CI

```bash
export PYTHONPATH="<repo>/src;<path-to-app-builder>"
python -m pytest tests/ -q          # the fast suite; heavy needs -m heavy
```

Last clean-clone run at the commit this README describes (`597d2ec`, Windows,
Python 3.14.6): **740 passed, 5 skipped, 0 failed in 163.48 s**. The five skips,
and why each one skips on a fresh clone:

| Skipped | Reason |
|---|---|
| `test_ops_null_pattern.py` | marked `heavy`; run it with `pytest -m heavy` |
| `test_ops_text.py`, `test_ops_text_affix.py` | the `wasm32` build artefact is not committed, so the WASM text lanes do not collect |
| `test_packaging_adapters.py` | needs `python -m build` to have run first |
| `test_rng_gate.py` | its reference file lives in `scratch/`, which is gitignored |

So on a clean clone two of these are not optional: the WASM tests and the RNG-gate
test never run without artefacts that are not in the tree. They run in a
developer's checkout and they skip in CI. That is a real coverage hole and it is
named here rather than hidden behind a green line.

Against an **installed** wheel in a fresh venv, the same suite gives **473 passed,
4 skipped, 0 failed**, with 20 test modules left uncollected because they read
engine source text that an installed package does not lay out as a source tree.
Each one is named, with its reason, at the end of the run.

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) has three Windows jobs:

1. **build the wheel, install it into a clean venv, import it** — gating.
2. **the tests that need nothing but this repository** — gating. The rest build
   their kernel through `from builder import MAIN`, and `builder` lives in a
   separate repository whose published `main` is behind what this tree needs, so
   that job is present, non-gating, and says so in its own step name.
3. **the whole suite** — non-gating, for the reason above.

There is **no publish workflow in this repository and no `v*` tag trigger**.
There is no Linux CI job either: only a Windows binary is committed, and with the
native library disabled the suite does not pass, so a Linux job would be red on
its first run.

---

## More reading

| Document | What it is |
|---|---|
| [docs/API.md](docs/API.md) | the 44-name surface, the kernel-level names, and every guard |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how the pieces fit: IR, Builder, CPU, native, GPU, WASM |
| [docs/EXAMPLES.md](docs/EXAMPLES.md) | runnable examples with their real output |
| [docs/README.md](docs/README.md) | which document describes what, and which is internal record |
| [BENCHMARKS.md](BENCHMARKS.md) | every performance number, with command, environment, date, verdict |
| [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) | what does not work, tagged by certainty |
| [README.packaging.md](README.packaging.md) | what ships in the wheel, and what a user gets |
| [CHANGELOG.md](CHANGELOG.md) | what 0.2.1 is |
| [SECURITY.md](SECURITY.md) | how to report a vulnerability |
| [CORRESPONDING-SOURCE.md](CORRESPONDING-SOURCE.md) | AGPL §6: where the source is in each artefact |
| [COMMERCIAL-LICENCE.md](COMMERCIAL-LICENCE.md) | the announced commercial licence, and what it is not |