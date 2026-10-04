# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# BENCHMARKS — what was measured, on what, and whether you can re-run it

Every performance number in this repository appears here with its command, its
environment, its date, and a verdict on whether it can be reproduced. If a
number is not in this file, this repository does not claim it.

**Rule applied:** a benchmark claim is only quotable if its artefact is in the
tree AND the command that produces it is in the tree AND the input it needs is
nameable. Artefacts whose producing script or dataset is missing are marked
HISTORICAL and must not be quoted as current results.

Last audited: **2026-10-04**, on commit `f2af608`, Windows, Python 3.14.6,
numpy 2.5.1, pandas 3.0.3, CPU backend. The audit did **not** re-run any
benchmark — the input datasets are not on this machine (see below). What it
did do is check, artefact by artefact, whether the command and the data still
exist. That check is what produced the verdicts below.

---

## 1. ClickBench Q1–Q43 @ ~1M rows — **REPRODUCIBLE** (given the dataset)

| | |
|---|---|
| Artefact | `tests/heavy/bench_clickbench_43.json` |
| Command | `python tests/heavy/bench_clickbench_43.py` — see the caveat below; the script resolves its own imports |
| Script | `tests/heavy/bench_clickbench_43.py` — **committed** |
| Input | `C:/App/competitions/ClickBench/data/hits_1m.parquet`, rows = 999978 |
| Input present on the audit machine | **NO** — `C:/App/competitions/` does not exist here |
| Method | one timed run per query; `stages_ms.compile` + `stages_ms.execute` vs a DuckDB reference; seed 42; no RNG (fixed data) |
| Verdict | **REPRODUCIBLE IN PRINCIPLE, NOT RE-RUN HERE.** The command and the input path are both named; the input was absent, so these are the numbers as recorded, not as re-measured. |

**The command does not honour `PYTHONPATH`.** `bench_clickbench_43.py:22-25`
hardcodes `APP_DIR = "C:/App/numfast/numfast"` and then does
`sys.path.insert(0, "C:/App/numfast/numfast/src")` and
`sys.path.insert(0, "C:/App/numfast/app-builder")`. Those inserts land ahead of
anything the caller exported, so from a clean clone the script imports the
developer's own `C:/App/numfast/numfast/src` rather than the tree it was invoked
from — it will run, and it will measure a different engine than the one you are
reading this file in. An earlier revision of this file documented the command as
`export PYTHONPATH=... && python tests/heavy/bench_clickbench_43.py`, which
suggests the environment variable is what selects the engine. It is not.
Recorded, not fixed: correcting the script is outside this file's write scope.

### Support table — read this, it is the honest one

**34 of 43 queries supported. 9 not.** Of the 9:

| Query | Why not |
|---|---|
| Q4, Q16, Q19, Q20, Q32, Q33, Q41, Q42 | **BIGINT narrowing.** ClickBench's `UserID` / `WatchID` / `URLHash` exceed int32; grouping or comparing on them raises rather than wrapping. |
| Q40 | **No conditional primitive.** `CASE WHEN TraficSourceID ...` needs a conditional the engine does not have. |

Note the split: **8 of the 9 are BIGINT narrowing; 1 (Q40) is a missing
capability, not a narrowing limit.** Any claim of the form "the 9 unsupported
queries are unsupported because of BIGINT narrowing" is wrong by one query.

Correctness labels are not uniform and are not smoothed over: `EXACT` (integer
exact), `EXACT+TOL` (float within `specs/core/conformance-profile.toml`),
`MULTISET`, `TIE-MULTISET`. A multiset match is a weaker guarantee than a row
match and is labelled as such.

### Where NumFast is slower than DuckDB

`ratio = (compile + execute) / duckdb`, from the same artefact. **> 1 means
NumFast is slower.** Every query at or above 0.5×:

| Query | NumFast total | DuckDB | NumFast slower by | Match |
|---|---|---|---|---|
| Q30 | 231.25 ms | 31.28 ms | **7.39×** | EXACT |
| Q36 | 167.47 ms | 23.69 ms | **7.07×** | EXACT |
| Q9 | 165.72 ms | 37.14 ms | **4.46×** | EXACT |
| Q5 | 78.75 ms | 19.21 ms | **4.10×** | EXACT |
| Q18 | 122.01 ms | 32.25 ms | **3.78×** | MULTISET |
| Q10 | 178.59 ms | 48.54 ms | **3.68×** | EXACT |
| Q17 | 122.01 ms | 40.71 ms | **3.00×** | EXACT |
| Q22 | 97.06 ms | 37.46 ms | **2.59×** | EXACT |

Q30, Q36, Q9, Q5, Q10, Q18, Q22 are the group-by-distinct / high-cardinality
shapes. This is the honest list of losses and it is not short.

### Where NumFast is faster

The same artefact records Q28 at 133.66×, Q43 at 86.04×, Q39 at 55.80×,
Q8 at 28.51×, Q15 at 20.30×. These are the small, aggregate-shaped queries
where DuckDB pays fixed per-query overhead that NumFast's single planner call
does not. Read them as "NumFast has a very low fixed cost", not as "NumFast is
86× faster than DuckDB".

---

## 2. H2O GROUPBY — **HISTORICAL**

| Artefact | Verdict |
|---|---|
| `tests/heavy/bench_h2o_100M_cpu_gpu.json` (N = 100M, dataset `G1_1e8_1e2_0_0`, GPU recorded as NVIDIA GeForce RTX 2060) | **HISTORICAL.** No producing script exists anywhere in the tree, and no date or Python version is recorded in the file. Not re-runnable. |
| `tests/heavy/bench_h2o_gpu_10M.json` (N = 10M) | **HISTORICAL.** Script `tests/heavy/bench_h2o_gpu_10M.py` IS committed, but its input `C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv` is absent, so the command cannot run here. |
| `tests/heavy/bench_h2o_scaled_gpu_10M.json` (N = 10M, carries `v3_schema` / `v3_range`) | **HISTORICAL.** Script `tests/heavy/bench_h2o_scaled_gpu_10M.py` IS committed; input absent, same as above. |

Commands, for the two that have a script:

```bash
export PYTHONPATH="C:/App/numfast/numfast/src;C:/App/numfast/app-builder"
python tests/heavy/bench_h2o_gpu_10M.py
python tests/heavy/bench_h2o_scaled_gpu_10M.py
```

Two things about these files that must not be lost: they record **CPU/GPU
agreement** (`cpu_gpu_match: true`, `cpu_gpu_maxdiff: 0`) which is a correctness
result and remains meaningful, and they record that at N = 10M the **GPU path
was slower than the CPU path** on this hardware. Neither file records a date,
a Python version, or a CPU model. Treat every millisecond in them as
unattributable.

---

## 3. `develop/benchmarks_public/` — **HISTORICAL, ALL FIVE SUITES**

This directory holds five `summary.json` snapshots. As of 2026-10-04 **all nine
source artefacts they were extracted from have been deleted**; the audit
checked each one by name. Each summary now carries a `provenance` key saying so,
and `run_all.py` requires that key and prints it above every number it shows.

| Suite | Its own verdict | Superseded? |
|---|---|---|
| `h2o` | Q1 exact parity; Q2 exact values but slower; Q3–Q5/Join not claimed | — |
| `clickbench` | 8/43 supported, Python 3.12 | **YES** — the current tree measures 34/43 (`tests/heavy/bench_clickbench_43.json`) |
| `hll_stage5` | 7 waves EXACT/DONE, small-N (n=20) proof | — |
| `gpu_proof` | parity EXACT, **CPU wins**; no speedup claimed | — |
| `q30_affine` | 43.02× on 181→2 nodes | **UNCONFIRMED** — measured in `develop/OptimizerAffine`, **not reproduced by current code** (181→181 via CSE+DCE) |

Two claims that were **removed** from this directory's `README.md` on
2026-10-04 because the artefacts contradict them:

1. "clickbench — 8/43 supported", presented as current. The tree measures 34/43.
2. "q30_affine — **43.02x faster EXACT** (215.89ms → 5.02ms)", presented as a
   verdict. Its own `summary.json` and its own `RESULTS.md` both say the
   181→2 transformation is an unconfirmed prototype and that the speedup is
   NOT claimed.

Also removed: the word "reproducible" from that README's opening line.

---

## 4. Origin-pipeline architectural limits — **MEASURED, against engine `4fc9914`**

Two facts about the architecture that are measurements, not opinions. Source:
`develop/audit_foundation/ORIGIN_PIPELINE_BENCH.md` in the NumFast dev-env
repository — an origin-pipeline benchmark, engine `4fc9914`, RTX 2060 over Vulkan
/ Xeon E5-2698 v4, median of 31 runs, cycles read from `QueryThreadCycleTime`.
**That document is not in this repository and is not published here.** The
numbers are reproduced below because they describe a path that still exists.

> **Provenance correction, verified 2026-10-05 at `879fb0b`.** The audit states it
> verified `git diff --name-only 4fc9914..1a3caa4 -- src/` returns 0 files, so
> that "every number below was measured against a byte-identical Python engine".
> That check was true when written and is **false now**:
> `git diff --name-only 4fc9914..879fb0b -- src/` returns **17 files**, of which
> twelve exist in both commits:
>
> * eight engine sources — `Compute/Fused/_lib/fused.py`,
>   `Drivers/CPU/_lib/native_cpu.py`, `Relational/Join/_lib/native{,_i64}.py`,
>   `Runtime/Planner/_lib/calibrate.py`,
>   `Semantic/TableExpr/_lib/{chain,expr,plan}.py`. Those changes are fork-root
>   path resolution and facade node lowering.
> * four changed in documentation only by this repository's publication pass —
>   `Relational/README.md` and
>   `Relational/Segmented/_lib/{adjacency,cost,segmented}.py`. Their executable
>   ASTs are identical before and after, verified by parsing both revisions.
>
> The other five are `src/numfast.egg-info/*`: generated packaging metadata that
> existed at `4fc9914` and is no longer tracked.
>
> None of the eight executable changes touch the window-composition or
> GPU-execution path these two limits describe, so the measurements below remain
> the best available account of that path. But they were taken against
> `4fc9914`, not against this tree, and this file should not imply otherwise. The
> audit record itself is unchanged and is reported, not rewritten.

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
surface the GPU is unreachable at all: `Chain.compile()` runs the CPU path and
`Chain.explain()` prints `backend=n/a` — both re-measured at `597d2ec`.

The one regime where the GPU reaches parity is **regime B**: the index built
**once**, kept resident, and **never shuffled** — 0.915× prep/consume at
*n* = 1 048 576, and **1.0×–1.5×** elsewhere. Add the per-epoch shuffle every
training loop actually does and the same index costs **13.2×** prep/consume;
build it per batch (regime A) and it is **158–168×**. The index buffer is also
**64× the data it indexes** — at *W* = 64, 4 B per window element, exactly as
large as the f32 window it produces — and the first epoch pays **319 ms** before
a single row is consumed.

So: **the GPU is at parity in a resident-index regime and loses once a shuffle is
required.** Both limits are properties of the buffer layout and the executor
boundary, not of the compute kernels.

---

## 5. Claims that are NOT made anywhere in this repository

* Full 43/43 ClickBench support.
* Any GPU speedup. The recorded GPU measurements are slower than CPU at the
  sizes measured; the GPU claim is parity and residency, not throughput.
* The Q30 181→2 / 43.02× affine collapse as a NumFast result.
* Any throughput claim from the `hll_stage5` proofs (n = 20 semantics runs).
* Anything about 100M-row ClickBench (only the 1M parquet exists).
* WASM performance of any kind.
* A wall-clock number that was not produced by a command named in this file.