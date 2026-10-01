# NumFast — Public Benchmarks (packaging only)

Beautiful, reproducible, honest. **No numbers were remeasured for this package.**
Every figure below is a verbatim extract of an already-measured artifact
(full JSONs live in dev-env `C:/App/numfast/develop/`, see each `suite.toml`).
`run_all` only verifies the packaging (schema + pretty print), never mass compute.

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

| Suite | Workload | Verdict |
|---|---|---|
| `h2o` | H2O GROUPBY Q1-Q5 + Join @ 10M (GOLD, CPU-only) | Q1 exact parity; Q2 exact values but HLL slower; Q3-Q5/Join not claimed |
| `clickbench` | ClickBench Q1-Q43 @ ~1M | 8/43 supported EXACT/EXACT+TOL; rest honest stubs |
| `hll_stage5` | 7 waves packet/float/join/window/text/multi/unified | all EXACT/DONE, small-N proof, zero-copy |
| `gpu_proof` | same packet CPU vs GPU, N=1M | parity EXACT, **CPU wins** 2.84ms vs 5.76ms |
| `q30_affine` | Q30 181→2 nodes | **43.02x faster EXACT** (215.89ms → 5.02ms) |

Machine-readable: `results/unified.jsonl` (written by `run_all.py`).
Details + GitHub tables: `RESULTS.md`.

## What is NOT claimed

H2O 100M · H2O Q3-Q5 numbers · HLL Join numbers · full 43/43 ClickBench ·
Q36/DISTINCT as HLL-supported (they are measured AB-probes, HLL side is stub) ·
any GPU speedup (CPU wins measured) · production/GPU path · WASM.

No core / HLL semantics / RoadGraph / ABI / methodology changes in this package.
No Extension was needed (presentation scripts only, stdlib) — hence no alias/mods.
