# NumFast — Public Benchmarks (packaging only)

Honest, and **NOT reproducible** — that word was here before and it was false.

**Status as of 2026-10-04: every source artefact these five summaries were
extracted from has been deleted.** All nine named JSONs (`bench_h2o_hll.json`,
`bench_join_distinct_hll.json`, `bench_stage6_gpu_proof.json`, the four under
`develop/HLL/`, `bench_clickbench_hll.json` + its three siblings,
`bench_q30_auto_affine.json` + two siblings) were checked and are absent. The
benchmarks' input datasets are absent too: `C:/App/competitions/` does not
exist on this machine, so neither `hits_1m.parquet` nor the H2O CSV is
available.

Consequences, stated plainly:

* Each `summary.json` now carries a machine-readable `provenance` key saying
  HISTORICAL / UNREPRODUCIBLE, and `run_all.py` REQUIRES that key and prints
  it for all five suites before printing any number.
* **Nothing in this directory is a current measurement.** Every figure is a
  snapshot of one run on one box on one day, kept for the record.
* For a figure that is current, sourced and inspectable, use
  `../../BENCHMARKS.md` and `../../tests/heavy/bench_clickbench_43.json`.

`run_all` only verifies the packaging (schema + provenance marker + pretty
print), never mass compute.

## One command (Git Bash, from `C:/App/numfast/numfast`)

```bash
export PYTHONPATH="/c/App/numfast/numfast/src" && python develop/benchmarks_public/run_all.py
```

Per-suite pretty output (same `PYTHONPATH`):

```bash
python develop/benchmarks_public/h2o/present_h2o.py
python develop/benchmarks_public/clickbench/present_clickbench.py
python develop/benchmarks_public/hll_stage5/present_hll_stage5.py
python develop/benchmarks_public/gpu_proof/present_gpu_proof.py
python develop/benchmarks_public/q30_affine/present_q30_affine.py
```

Original heavy measurements (reference only, NOT run by `run_all` — see `suite.toml` per suite).

## Suites

All five rows below are HISTORICAL and UNREPRODUCIBLE (see above).

| Suite | Workload | Verdict as recorded then — superseded where noted |
|---|---|---|
| `h2o` | H2O GROUPBY Q1-Q5 + Join @ 10M (GOLD, CPU-only) | Q1 exact parity; Q2 exact values but HLL slower; Q3-Q5/Join not claimed |
| `clickbench` | ClickBench Q1-Q43 @ ~1M | **SUPERSEDED.** 8/43 supported here; the current tree measures **34/43** in `tests/heavy/bench_clickbench_43.json`. Use that file. |
| `hll_stage5` | 7 waves packet/float/join/window/text/multi/unified | all EXACT/DONE, small-N proof, zero-copy |
| `gpu_proof` | same packet CPU vs GPU, N=1M | parity EXACT, **CPU wins** — no speedup was ever claimed |
| `q30_affine` | Q30 181→2 nodes | **UNCONFIRMED PROTOTYPE.** 43.02x (215.89ms → 5.02ms) was measured in `develop/OptimizerAffine` and is **NOT reproduced by current code** (181→181 via CSE+DCE). Do not quote it as a NumFast result. |

Machine-readable: `results/unified.jsonl` (written by `run_all.py`).
Details + GitHub tables: `RESULTS.md`.

## What is NOT claimed

H2O 100M · H2O Q3-Q5 numbers · HLL Join numbers · full 43/43 ClickBench ·
Q36/DISTINCT as HLL-supported (they are measured AB-probes, HLL side is stub) ·
any GPU speedup (CPU wins measured) · production/GPU path · WASM.

No core / HLL semantics / RoadGraph / ABI / methodology changes in this package.
No Extension was needed (presentation scripts only, stdlib) — hence no alias/mods.
