# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# benchmarks/execution-economics — planner/routing measurements (MEASURED, non-normative)

Architectural notes (normative rules live in core; numbers stay here):

1. No `N → GPU` threshold exists. Routing = measured cost per (op-multiset, N, M, selectivity, cold/warm, residence) + hard gates + cold-safety.
2. Transfer dominates total: 10M H2D ≈ 45–135ms; kernel wins (rng 5–6×) survive only where transfer is small (on-device rng, sort-tract R3) or eliminated (resident).
3. Resident removes inter-stage round-trips (bytes × crossings), it does not speed up kernels. Counter-example on file: Pack→GroupBy resident 44.57 vs round-trip 39.44 (0.885×).
4. Cold auto = CPU (device-init ~2.4–2.7s unmeasured); warm regret 0 wherever prediction is in coverage.

## Per-op fits (`cost(n) = a·n + b`, warm; cold = warm + Σ compile_miss)

| op | CPU a / b | GPU a / b | compile_miss |
|---|---|---|---|
| rng_fill_i32 | 6.06e-05 / 0.075 | 1.13e-05 / 2.21 | 0.46 |
| compare | 3.03e-07 / 0.047 | 1.53e-05 / 2.18 | 0.65 |
| mask | 3.14e-07 / 0.056 | 2.76e-05 / 2.16 | 1.50 |
| filter | 1.61e-05 / 0.065 | 3.97e-05 / 6.25 | 0.19 |
| gather | 5.75e-06 / 0.005 | 1.91e-05 / 2.44 | 0.33 |
| reduce (sum) | 9.06e-07 / 0.063 | 2.83e-06 / 2.55 | 0.57 |
| slice | −5.6e-08 / 0.029 | 4.17e-06 / 5.48 | 0.28 |
| pack_keys | 1.02e-05 / −0.064 | 1.97e-05 / 5.18 | 0.23 |
| sort | 7.75e-05 / −1.24 | 3.89e-05 / 43.58 | 0.0 |

GroupBy per-cardinality only (inter-M interpolation FORBIDDEN):

| M | CPU a / b | GPU a / b |
|---|---|---|
| 100 | 1.94e-06 / 0.235 | 2.61e-05 / 4.63 |
| 10 000 | 4.04e-06 / 1.28 | 2.56e-05 / 6.16 |
| 100 000 | 2.14e-05 / 6.58 | 3.93e-05 / 13.93 |

Fit quality (honest): `filter_cpu` MAPE 16%, groupby MAPE 73% CPU / 34% GPU, `slice_gpu` R² 0.915. Held-out N=500K rel-err ≤ ~0.16 except groupby and transfer-holdout.

## Transfer

| bytes | H2D | D2H |
|---|---|---|
| 1MB | 0.90 | 3.45 |
| 10MB | 7.51 | 18.67 |
| 100MB | 82.25 | 147.99 |
| 1GB | 2132.33 | 1438.62 |

Fits: H2D `a=7.87e-07 ms/B, b=−0.32` (trusted ≤100MB); D2H `a=1.34e-06, b=4.78` (trusted ≤1GB). 1GB H2D holdout pred 845 vs meas 2132 → extrapolation beyond coverage is UNMEASURED.

## Routing audit (resident 1M, excerpt)

R1 Filter→GroupBy → cpu ✓; R3 Sort→Slice→Gather (pow2) → gpu ✓ (only routing GPU-win); R4 Pack→GroupBy → cpu ✓; R5 CMF→GroupBy → cpu ✓; R2 Lookup→Filter → N/A (no IR op — honest gap).

## Portability pipeline (excerpt)

RNG→ARITH→FILTER→SORT→UNIQUE→GROUPBY→AGGREGATE, seed 42, N=200K/1M. Integer triangle CPU==GPU==WASM(RNG stages) bit-exact (SHA256); float under tolerance contract `08`. First GPU dispatch cold ~2400ms; compile/optimize <0.2ms. JOIN-tract out of scope.
