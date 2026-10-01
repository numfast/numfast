# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# Scaled-integer native path + multi-variant DLL (research, branch `rebuilt`)

Production ABI frozen: 8 symbols (`nf_group_sum_count`,
`nf_group_multi_sum_count`, `nf_pack_i32_direct`, `nf_pattern_encode`,
`nf_sorted_run_i64/f64`, `nf_carry_build_i64/f64`) byte-identical.
SPEC v0.2 untouched. All below is research-only additive surface.
First native ABI (`keys=i32/values=f64`) was a historic research call;
correct architecture: Schema owns the physical encoding, native sums
prepared typed contiguous buffers, user-view conversion at the API edge.

## 1. Scaled-integer semantics (Task 1, v3 scale=1e6)

Convention bridge (no SPEC change): SPEC `05` writes
`physical = (logical - offset) / scale` with `scale` as quantum
(`q = 1e-6`); this doc writes multiplier `M = 1_000_000` ticks/unit.
Same mapping: `tick = round((logical - offset) * M)`,
`logical = tick / M + offset`. Rounding: round-half-even
(`np.rint`, per `08-js-boundary`).

| Item | Rule |
|---|---|
| scale | `M = 1_000_000` (6 digits, v3). Quantum `q = 1/M`. |
| offset | general `offset` supported by formula; research runs `offset = 0` (v3 unsigned-ish; negatives allowed, see below) |
| physical dtype | int64 ticks (`physical_dtype=int64`). int32 ticks REJECTED for v3: `INT32_MAX = 2147483647` ticks = ~2147 units < v3 range (~0..10000) — per-value overflow before any sum |
| int64 range | `±9.22e18` ticks = `±9.22e12` units. Q3-10M worst case: `1e7` rows x `1e10` ticks = `1e17` total < `9.22e18` OK (proof per dataset: `n * max|tick|` vs `INT64_MAX`, asserted in bench) |
| per-group range | same bound per group (`count_g * max|tick|`); 10M/G=100k: `~200 x 1e10 = 1e12` « limit |
| overflow | explicit error `OVERFLOW = -4` via `checked_add`, never wraps, never saturates. First overflowing row aborts; partial outputs caller-owned garbage (same rule as sorted abort) |
| negatives | two's complement int64, exact (bench data `N(5000,1500)` keeps natural negatives) |
| null/validity | fast path assumes all-valid (same as frozen `dense_scatter`); nulls filtered caller-side before native call. Validity-bitmap lane reserved, not in this ABI |
| accuracy | int sum bit-exact vs big-int reference (0 ULP). `mean = int_sum / count` in f64 then `/M` at API edge: exactly ONE rounding at the boundary (error <= 0.5 ULP of the result). f64 path rounds per value AND per add (error grows with n) |
| conversion point | ticks prepared caller-side (`np.rint(logical * M)`); native never sees floats on the int path; `/M` only when deriving user-visible means |

## 2. Multi-variant DLL (Task 2)

ONE file (`target/.../release/numfast_native.dll`, 8 frozen + 3 new
symbols). Variant = `u32` runtime param, no rebuild, no per-variant DLL.

| symbol | variant | algorithm | accumulator | error codes |
|---|---|---|---|---|
| `nf_group_variant_f64` | 0 `V_DENSE` | unchecked dense scatter (existing kernel) | f64 | 0 / -1 null / -3 bad variant |
| `nf_group_variant_f64` | 1 `V_CHECKED` | range-checked scatter (existing fused, ncols=1) | f64 | + -2 out-of-range |
| `nf_group_variant_f64` | 2 `V_SORTED` | sorted-dense runs (new) | f64 | + -5 unsorted |
| `nf_group_variant_i64` | 0 `V_DENSE` | unchecked dense scatter (new) | int64 ticks | 0 / -1 / -3 / -4 overflow |
| `nf_group_variant_i64` | 1 `V_CHECKED` | range-checked scatter (new) | int64 ticks | + -2 |
| `nf_group_variant_i64` | 2 `V_SORTED` | sorted-dense runs (new) | int64 ticks | + -5 |
| `nf_group_variant_i32` | 0 `V_DENSE` | range-checked scatter (new) | int32 (overflow demo) | 0 / -1 / -2 / -3 / -4 |
| `nf_group_variant_i32` | 2 `V_SORTED` | sorted-dense runs (new) | int32 (overflow demo) | + -5 |

Research return-code table (new symbols only, no overloads):
`0` ok · `-1` null pointer · `-2` out-of-range key · `-3` unknown
variant (outputs untouched) · `-4` integer overflow · `-5` unsorted
input on `V_SORTED` (speculative abort).

Parity rule: int families bit-exact across variants (integer addition
commutes — traversal order irrelevant). f64 `V_SORTED` vs row-order
variants: tolerance (few ULP, other accumulation order), counts exact
everywhere. Bench asserts this on every run before timing.

## 3. Evidence

`bench_scaled_mv.py` (seed 42, synthetic Q3-10M, G=100k) → stage
breakdown ms + throughput + bytes/RSS. Results: `results_scaled_mv.json`.
STOP: no integration into production paths.
