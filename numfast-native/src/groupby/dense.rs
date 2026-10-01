// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `dense_scatter` mechanic: sums + counts in ONE pass, O(n).
//! `keys` are dense codes in `[0, g)` (factorize happens caller-side).
//! Direct indexing — no hashmap, no sort, no second pass. Sequential
//! `i = 0..n` order matches `np.bincount` accumulation order, so float
//! results are bit-exact. Out-of-range key = precondition violation
//! (safe Rust traps, never UB) — the checked sibling lives in
//! [`crate::groupby::fused`].

use crate::core::numeric::widen_usize;
use crate::core::validity::fill_zero;
use crate::series::reduce::{add_f64, bump};

/// keys: &[i32] dense codes, values: &[f64], g: group count.
/// sums/counts must have len >= g; first g lanes overwritten.
#[inline]
pub fn dense_scatter(
    keys: &[i32],
    values: &[f64],
    g: usize,
    sums: &mut [f64],
    counts: &mut [i64],
) {
    debug_assert_eq!(keys.len(), values.len());
    debug_assert!(sums.len() >= g);
    debug_assert!(counts.len() >= g);
    // Zero output lanes (caller passes uninit/zeroed buffers).
    fill_zero(&mut sums[..g]);
    fill_zero(&mut counts[..g]);
    // Single fused pass.
    for i in 0..keys.len() {
        let k = widen_usize(keys[i]); // contract: k < g
        add_f64(&mut sums[k], values[i]);
        bump(&mut counts[k]);
    }
}

/// Previous module path kept for internal import compat.
#[inline]
pub fn fused_sum_count(
    keys: &[i32],
    values: &[f64],
    g: usize,
    sums: &mut [f64],
    counts: &mut [i64],
) {
    dense_scatter(keys, values, g, sums, counts)
}
