// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Multi-variant dispatch (research-only, Task 2).
//!
//! One DLL, several algorithm variants per primitive, runtime selection
//! via a `variant` enum parameter — no rebuild, no per-variant DLLs.
//! Each variant keeps the identical caller-owned dense contract:
//! `keys[0..n]` int32 dense codes, value buffer, `sums[0..g]` +
//! `counts[0..g]` outputs overwritten, `g` groups.
//!
//! | symbol | variant | algorithm | accumulator |
//! |---|---|---|
//! | `nf_group_variant_f64` | 0 `V_DENSE` | unchecked dense scatter (existing `dense_scatter`) | f64 |
//! | `nf_group_variant_f64` | 1 `V_CHECKED` | range-checked scatter (existing `fused_scatter_soa`, ncols=1) | f64 |
//! | `nf_group_variant_f64` | 2 `V_SORTED` | sorted-dense runs (new `sorted_dense_f64`, needs nondecreasing keys) | f64 |
//! | `nf_group_variant_i64` | 0 `V_DENSE` | unchecked dense scatter (new `dense_scatter_i64`) | int64 ticks |
//! | `nf_group_variant_i64` | 1 `V_CHECKED` | range-checked scatter (new `checked_scatter_i64`) | int64 ticks |
//! | `nf_group_variant_i64` | 2 `V_SORTED` | sorted-dense runs (new `sorted_dense_i64`) | int64 ticks |
//! | `nf_group_variant_i32` | 0 `V_DENSE` | range-checked scatter (new `checked_scatter_i32`) | int32 (overflow demo) |
//! | `nf_group_variant_i32` | 2 `V_SORTED` | sorted-dense runs (new `sorted_dense_i32`) | int32 (overflow demo) |
//!
//! Parity rule: int families are bit-exact across variants (integer
//! addition commutes); f64 `V_SORTED` may differ from row-order
//! variants by a few ULP (different accumulation order) — tolerance,
//! not bits. Counts are exact everywhere.

use crate::core::errors::{BAD_VARIANT, OK};
use crate::groupby::dense::dense_scatter;
use crate::groupby::fused::fused_scatter_soa;
use crate::groupby::scaled::{
    checked_scatter_i32, checked_scatter_i64, dense_scatter_i64, sorted_dense_f64,
    sorted_dense_i32, sorted_dense_i64,
};

/// Unchecked dense scatter variant.
pub const V_DENSE: u32 = 0;
/// Range-checked scatter variant (`BAD_RANGE` on out-of-range key).
pub const V_CHECKED: u32 = 1;
/// Sorted-dense run variant (`UNSORTED` on regressing key).
pub const V_SORTED: u32 = 2;

/// f64 family dispatch. Returns `OK` / `BAD_RANGE` / `UNSORTED` /
/// `BAD_VARIANT`. Unknown variant leaves outputs untouched.
#[inline]
pub fn dispatch_f64(
    keys: &[i32],
    values: &[f64],
    g: usize,
    variant: u32,
    sums: &mut [f64],
    counts: &mut [i64],
) -> i32 {
    match variant {
        V_DENSE => {
            dense_scatter(keys, values, g, sums, counts);
            OK
        }
        V_CHECKED => fused_scatter_soa(keys, values, 1, g, sums, counts),
        V_SORTED => sorted_dense_f64(keys, values, g, sums, counts),
        _ => BAD_VARIANT,
    }
}

/// int64-tick family dispatch. Returns `OK` / `BAD_RANGE` / `UNSORTED` /
/// `OVERFLOW` / `BAD_VARIANT`. Unknown variant leaves outputs untouched.
#[inline]
pub fn dispatch_i64(
    keys: &[i32],
    ticks: &[i64],
    g: usize,
    variant: u32,
    sums: &mut [i64],
    counts: &mut [i64],
) -> i32 {
    match variant {
        V_DENSE => dense_scatter_i64(keys, ticks, g, sums, counts),
        V_CHECKED => checked_scatter_i64(keys, ticks, g, sums, counts),
        V_SORTED => sorted_dense_i64(keys, ticks, g, sums, counts),
        _ => BAD_VARIANT,
    }
}

/// int32 family dispatch (overflow-demo lanes). Returns `OK` /
/// `BAD_RANGE` / `UNSORTED` / `OVERFLOW` / `BAD_VARIANT`.
/// No `V_CHECKED` lane here: `V_DENSE` is already range-checked
/// (int32 lanes exist to prove the overflow contract, not speed).
#[inline]
pub fn dispatch_i32(
    keys: &[i32],
    ticks: &[i32],
    g: usize,
    variant: u32,
    sums: &mut [i32],
    counts: &mut [i64],
) -> i32 {
    match variant {
        V_DENSE => checked_scatter_i32(keys, ticks, g, sums, counts),
        V_SORTED => sorted_dense_i32(keys, ticks, g, sums, counts),
        _ => BAD_VARIANT,
    }
}
