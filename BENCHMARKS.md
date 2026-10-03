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
| Command | `export PYTHONPATH="C:/App/numfast/numfast/src;C:/App/numfast/app-builder" && python tests/heavy/bench_clickbench_43.py` |
| Script | `tests/heavy/bench_clickbench_43.py` — **committed** |
| Input | `C:/App/competitions/ClickBench/data/hits_1m.parquet`, rows = 999978 |
| Input present on the audit machine | **NO** — `C:/App/competitions/` does not exist here |
| Method | one timed run per query; `stages_ms.compile` + `stages_ms.execute` vs a DuckDB reference; seed 42; no RNG (fixed data) |
| Verdict | **REPRODUCIBLE IN PRINCIPLE, NOT RE-RUN HERE.** The command and the input path are both named; the input was absent, so these are the numbers as recorded, not as re-measured. |

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

## 4. Claims that are NOT made anywhere in this repository

* Full 43/43 ClickBench support.
* Any GPU speedup. The recorded GPU measurements are slower than CPU at the
  sizes measured; the GPU claim is parity and residency, not throughput.
* The Q30 181→2 / 43.02× affine collapse as a NumFast result.
* Any throughput claim from the `hll_stage5` proofs (n = 20 semantics runs).
* Anything about 100M-row ClickBench (only the 1M parquet exists).
* WASM performance of any kind.
* A wall-clock number that was not produced by a command named in this file.