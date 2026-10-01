// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `fused_scatter` mechanic: SoA multi-aggregate over dense codes —
//! shared counts + one sums array per value column, single pass.
//! Mirrors prod `_fused_soa_by_code` dense path (counts computed once,
//! mean derived sum/count by caller — never a second pass here).
//!
//! Layout (all contiguous, caller-owned):
//! - `keys[0..n]`: int32 dense codes in `[0, g)`
//! - `values[0..ncols*n]`: SoA value columns, column c at `[c*n..(c+1)*n]`
//! - `sums[0..ncols*g]`: outputs, column c at `[c*g..(c+1)*g]`, overwritten
//! - `counts[0..g]`: shared output, overwritten
//!
//! Unlike [`crate::groupby::dense::dense_scatter`], out-of-range keys
//! return `BAD_RANGE` (explicit hot bounds check, no panic via FFI).

use crate::core::checks::key_in_range;
use crate::core::errors::BAD_RANGE;
use crate::core::numeric::widen_usize;
use crate::core::validity::fill_zero;
use crate::series::reduce::{add_f64, bump};

/// Returns 0 ok, `BAD_RANGE` on out-of-range key.
#[inline]
pub fn fused_scatter_soa(
    keys: &[i32],
    values: &[f64],
    ncols: usize,
    g: usize,
    sums: &mut [f64],
    counts: &mut [i64],
) -> i32 {
    debug_assert_eq!(values.len(), ncols * keys.len());
    debug_assert!(sums.len() >= ncols * g);
    debug_assert!(counts.len() >= g);
    fill_zero(&mut sums[..ncols * g]);
    fill_zero(&mut counts[..g]);
    let n = keys.len();
    for i in 0..n {
        let k = widen_usize(keys[i]);
        if !key_in_range(k, g) {
            return BAD_RANGE;
        }
        bump(&mut counts[k]);
        for c in 0..ncols {
            add_f64(&mut sums[c * g + k], values[c * n + i]);
        }
    }
    0
}

/// Previous module path kept for internal import compat.
#[inline]
pub fn fused_multi_aggregate_soa(
    keys: &[i32],
    values: &[f64],
    ncols: usize,
    g: usize,
    sums: &mut [f64],
    counts: &mut [i64],
) -> i32 {
    fused_scatter_soa(keys, values, ncols, g, sums, counts)
}
