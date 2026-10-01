# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# REUSE matrix — numfast-native core (branch `rebuilt`)

Goal: 12 independent implementations → one small native core: shared
data-level internals + tiny `#[inline(always)]` atoms; every hot loop
stays flat and specialized. No universal super-helper with N generic
params (kills optimization, complicates code).

Surface note: the crate exposes **8** `extern "C"` symbols
(`nf_group_sum_count`, `nf_group_multi_sum_count`, `nf_pack_i32_direct`,
`nf_pattern_encode`, `nf_sorted_run_i64`, `nf_sorted_run_f64`,
`nf_carry_build_i64`, `nf_carry_build_f64`) over 8 internal kernels.
C ABI and observable behaviour are frozen; this file only maps how the
kernels share one physical core underneath.

Legend: HOT = inside per-row/per-lane loop · WARM = per-call prologue
(zeroing, once per call) · COLD = setup / error path (runs ~never).

## 1. Helper → consumers → hot/cold → inline

| Helper (`src/…`) | Consumers (kernels / FFI) | Hot/Cold | Inline | Notes |
|---|---|---|---|---|
| `core::buffers::borrow`, `borrow_mut` (ptr+len → slice) | all 8 FFI wrappers | COLD | `#[inline]` | Single unsafe wrap point; contract comments live here |
| `core::checks::non_null` | all 8 FFI wrappers | COLD | `#[inline(always)]` | 1 comparison; error path returns `errors::NULL_*` |
| `core::checks::key_in_range` (dense bounds) | `fused_scatter_soa` only | HOT | `#[inline(always)]` | `dense_scatter` is *intentionally* unchecked (contract: safe-Rust trap on OOR, same as before) |
| `core::checks::offset_in_bounds` | `pattern_encode` | WARM/HOT | `#[inline(always)]` | Row-boundary validation inside the offset walk |
| `core::errors::*` consts (`OK`, `NULL_PTR`, `NULL_STATE`, `BAD_RANGE`, `MALFORMED`, `UNSORTED_ABORT`) | all 8 FFI wrappers + kernels returning codes | COLD | consts (no code) | `-2` means OOR / overflow / null depending on fn — table in `errors.rs`, semantics unchanged |
| `core::validity::fill_zero<T>` (1 generic param) | `dense_scatter`, `fused_scatter_soa` | WARM | `#[inline]` | `0.0`/`0` identical to old explicit loops; `Default` = zero for all lane types |
| `core::validity::clear_slot` (codes=0, valid=0) | `pattern_encode` | HOT | `#[inline(always)]` | Per-row reset, was 2 inline stores — still 2 stores |
| `core::validity::is_present` (counts != 0) | `buffer_compact` (→ both carry kernels) | HOT | `#[inline(always)]` | Absent-lane predicate, single source |
| `core::numeric::pack_elem` (radix + overflow, i64 math) | `pack_codes` | HOT | `#[inline(always)]` | Was the loop body; flat store stays in kernel |
| `core::numeric::is_digit_byte`, `int_limit` | `pattern_encode` | HOT | `#[inline(always)]` | Digit test + neg/pos int32 limit; fused digit loop stays in kernel (width/err_row intertwined, not worth splitting) |
| `core::numeric::widen_usize`, `widen_i64` | dense, fused, sorted, carry | HOT | `#[inline(always)]` | `as` casts, zero-cost, one spelling |
| `series::reduce::{add_f64, bump}` | dense, fused, sorted counts (i64 acc in sorted stays `+=` on the generic lane) | HOT | `#[inline(always)]` | Lane reduce atoms; accumulation **order unchanged** (sequential `i=0..n`, bit-exact floats) |
| `series::scan::{key_changed, key_regressed, flush_run<S>}` | `sorted_runs` (i64+f64) | HOT | `#[inline(always)]` / 1-param generic | Run-scan atoms; abort-on-regress preserved (`-1`, speculative) |
| `core::compact::buffer_compact<S: Copy>` (1 generic param) | `carry_build_i64`, `carry_build_f64` | HOT | normal fn (is the loop; monomorphizes flat) | Forward compaction, ascending codes; write frontier never overtakes input |
| `core::hash::probe_step`, `next_probe` | **none yet (reserved)** | — | `#[allow(dead_code)]` | Linear-probe atoms for the future hash-groupby surface (STOP: surface not started, nothing exported) |
| `groupby::hash` facade | **none yet (reserved)** | — | re-export only | Wiring point for hash surface; no C ABI change |

## 2. Base mechanics (7) → operations

| Mechanic | Lives in | Assembled ops (C ABI, unchanged) |
|---|---|---|
| `dense_scatter` | `groupby/dense.rs` | `nf_group_sum_count` |
| `fused_scatter` (SoA, shared counts) | `groupby/fused.rs` | `nf_group_multi_sum_count` |
| `sorted_runs` (1-param generic over lane) | `groupby/sorted.rs` | `nf_sorted_run_i64`, `nf_sorted_run_f64` |
| `hash_insert` (reserved atoms) | `core/hash.rs` → `groupby/hash.rs` | — (future surface) |
| `buffer_compact` (1-param generic over lane) | `core/compact.rs` → `result/carry.rs` | `nf_carry_build_i64`, `nf_carry_build_f64` |
| `pack_codes` | `transform/pack.rs` | `nf_pack_i32_direct` |
| `pattern scan+parse` (+ `scan_kernel`/`reduce_kernel` atoms from `series/`) | `transform/pattern.rs` | `nf_pattern_encode` |

`scan_kernel` / `reduce_kernel` are deliberately decomposed into the
`series::{scan,reduce}` inline atoms above instead of one parameterized
mega-kernel: the accumulation order (float bit-exactness) and the
abort/overflow contracts must stay visible in each flat loop.

## 3. What is shared vs deliberately NOT shared

- Shared: null checks, slice wraps, error codes, output zeroing,
  validity marking, i32 widening, digit/radix arithmetic, bounds checks,
  forward-compaction traversal, run-scan atoms, lane reduce atoms.
- NOT shared (kept specialized per kernel): the hot `for i in 0..n`
  bodies themselves (different lane types / contracts: unchecked dense
  vs `-2`-checked fused; int vs float acc; per-row variable-width
  pattern parsing), and the pattern digit loop (width + err_row state
  fused in).
- Generics budget: exactly two 1-param generics (`fill_zero<T>`,
  `buffer_compact<S>`, plus `sorted_runs<S>` / `flush_run<S>` — each a
  single lane-type param, monomorphized to the same flat loops as
  before). No trait objects, no multi-param helpers.
