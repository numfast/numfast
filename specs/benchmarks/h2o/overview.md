# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# benchmarks/h2o — H2O GroupBy findings (MEASURED, non-normative)

Seed 42, RTX 2060 Vulkan, Windows. Method: `10-benchmark-methodology.md`.
Normative rules live in core specs; NOTHING here is a rule.

## Setup

- 10M rows (`G1_1e7_1e2_0_0`). Q1/Q2 = int32 path; Q3–Q5 contain float64
  column v3 (`[2.3e-05, 99.999987]`) → GPU only via scaled-int contract
  (`delta-7`: `physical = rint((v - offset) / scale)`, scale from Schema).
- Unscaled float64 on GPU = explicit `MapF64 guard` error (never silent).

## Results (total, warm)

| query | CPU | GPU | note |
|---|---|---|---|
| Q1 groupby (M=100) | ~58ms | ~249ms | CPU wins total |
| Q2 groupby (M=10K) | ~443ms | ~355ms | only warm GPU-win; planner predicts CPU (calibration debt, see below) |
| Q1 resident chain | — | ~70ms vs hybrid ~143ms | resident removes 40MB pack D2H + 40MB downstream (D2H 1212B) |
| Q2 resident chain | usual ~168ms | ~86–90ms vs hybrid ~304ms | D2H 122KB instead of 80MB; −1.87× vs usual |
| Q3/Q5 dict-materialization cost | — | — | ESTIMATE (not a claim): Q3 ~45ms, Q5 ~75ms saved by columnar result; close by running `bench_matrix_once_10M.py` + `bench_endgame_breakdown.py` before/after |

## Scaled-f64 coverage (v3)

- `*1e6` physical range `[23, 99999987]` ⊂ int32 (21× headroom).
- `scaled_sum = 499976651408061` (== GOLD Q5 s3_scaled), `f64_sum` diff 0.0.
- Total guard: `N * max|phys| < 2**63`; non-finite / out-of-int32 / overflow → explicit CPU error.

## Known debts (not architecture)

- Q2-warm mispredict (pred 131/352 vs meas 443/355, regret ≈ 88ms): groupby inter-M interpolation untrusted; M trusted only in {100, 10K, 100K}.
- Filter→GroupBy 10M total: CPU wins at all selectivities (0.15–0.55×); resident 256ms still worse than CPU 99ms.
- 1GB H2D superlinear (pred 845 vs meas 2132ms): fit trusted ≤100MB only.
