// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `mixed_sum_only` mechanic (P2, accepted): single-pass group-by sums
//! over heterogeneous lanes — `n_i32` int32 columns + `n_f64` float64
//! columns, WITHOUT the counts lane (P2). Occupancy comes from one
//! keys-only count pass ([`count_only`]).
//!
//! Motivation (measured 100M/16T, exact): production `_native_mixed`
//! runs ONE key pass PER column with a counts RMW in every pass (Q5 =
//! 3 key passes + 3 counts arrays, 2 discarded). Decoupled counts +
//! one sum-only pass moves less traffic at bandwidth saturation:
//! Q5 +18%, Q3 +14% end-to-end (see tests/heavy/bench_p12_100M.py +
//! scratch/bench_p12_100M.json). The single-pass WITH-counts variant
//! (P1) was measured too and REJECTED (marginal at 16T, regresses at
//! low thread counts) — it is not shipped here.
//!
//! Generic over lane COUNTS (`n_i32`, `n_f64`), never over queries:
//! Q5 is (2, 1), Q3 is (1, 1). int64 columns are NOT covered — the
//! caller falls back to the proven per-column path (same as before,
//! never a silent lossy route). Zero of a family is legal (loops skip).
//!
//! Layout (all contiguous, caller-owned, SoA exactly like `fused_scatter_soa`):
//! - `keys[0..n]`: int32 dense codes in `[0, g)`
//! - `i32_vals[0..n_i32*n]`: column c at `[c*n..(c+1)*n]`
//! - `f64_vals[0..n_f64*n]`: column c at `[c*n..(c+1)*n]`
//! - `i32_sums[0..n_i32*g]`, `f64_sums[0..n_f64*g]`: outputs, overwritten
//!
//! Contracts: out-of-range key returns `BAD_RANGE` (explicit hot bounds
//! check, no panic via FFI); int32 overflow returns `OVERFLOW` (checked
//! add, never wraps). Sequential `i = 0..n` order matches `dense_scatter`
//! accumulation order, so float results are bit-exact.

use crate::core::checks::key_in_range;
use crate::core::errors::{BAD_RANGE, OK, OVERFLOW};
use crate::core::numeric::widen_usize;
use crate::core::validity::fill_zero;
use crate::series::reduce::{add_f64, add_i32_checked, bump};

/// Single-pass mixed sums WITHOUT the counts lane (P2). Same contract
/// as the layout above; returns `OK` / `BAD_RANGE` / `OVERFLOW`.
/// The caller owns occupancy (one keys-only [`count_only`] pass).
#[inline]
pub fn mixed_sum_only_i32_f64(
    keys: &[i32],
    i32_vals: &[i32],
    f64_vals: &[f64],
    n_i32: usize,
    n_f64: usize,
    g: usize,
    i32_sums: &mut [i32],
    f64_sums: &mut [f64],
) -> i32 {
    let n = keys.len();
    debug_assert_eq!(i32_vals.len(), n_i32 * n);
    debug_assert_eq!(f64_vals.len(), n_f64 * n);
    debug_assert!(i32_sums.len() >= n_i32 * g);
    debug_assert!(f64_sums.len() >= n_f64 * g);
    fill_zero(&mut i32_sums[..n_i32 * g]);
    fill_zero(&mut f64_sums[..n_f64 * g]);
    for i in 0..n {
        let k = widen_usize(keys[i]);
        if !key_in_range(k, g) {
            return BAD_RANGE;
        }
        for c in 0..n_i32 {
            if !add_i32_checked(&mut i32_sums[c * g + k], i32_vals[c * n + i]) {
                return OVERFLOW;
            }
        }
        for c in 0..n_f64 {
            add_f64(&mut f64_sums[c * g + k], f64_vals[c * n + i]);
        }
    }
    OK
}

/// Keys-only count pass (P2 companion: occupancy for sum-only end-to-end).
/// Returns `OK` / `BAD_RANGE`.
#[inline]
pub fn count_only(keys: &[i32], g: usize, counts: &mut [i64]) -> i32 {
    debug_assert!(counts.len() >= g);
    fill_zero(&mut counts[..g]);
    for i in 0..keys.len() {
        let k = widen_usize(keys[i]);
        if !key_in_range(k, g) {
            return BAD_RANGE;
        }
        bump(&mut counts[k]);
    }
    OK
}

/// Owner-shard 2xi32 + 1xf64 sums WITH counts (Q5-P3 experiment: each
/// thread owns a disjoint group range `[lo, hi)` of SHARED global dense
/// state — no private states, no merge, no atomics, no partition pass).
/// Every owner thread scans all rows; rows outside its range are skipped
/// (predicated, no materialization). The caller zeroes nothing: each
/// thread zeroes ONLY its owned lanes (disjoint across threads, so the
/// full state is zeroed exactly once with zero redundancy). Specialized
/// by lane signature (2, 1) + large-M dispatch on the caller side, never
/// by query. Contracts: out-of-range key (outside `[0, g)`) returns
/// `BAD_RANGE`, int32 overflow returns `OVERFLOW` (checked add, never
/// wraps); accumulation order per group matches `dense_scatter`
/// (bit-exact: each group is touched by exactly one thread, in order).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn owner_2i32_1f64(
    keys: &[i32],
    a: &[i32],
    b: &[i32],
    c: &[f64],
    lo: usize,
    hi: usize,
    g: usize,
    s1: &mut [i32],
    s2: &mut [i32],
    s3: &mut [f64],
    counts: &mut [i64],
) -> i32 {
    let n = keys.len();
    debug_assert_eq!(a.len(), n);
    debug_assert_eq!(b.len(), n);
    debug_assert_eq!(c.len(), n);
    debug_assert!(s1.len() >= g);
    debug_assert!(s2.len() >= g);
    debug_assert!(s3.len() >= g);
    debug_assert!(counts.len() >= g);
    debug_assert!(lo <= hi && hi <= g);
    fill_zero(&mut s1[lo..hi]);
    fill_zero(&mut s2[lo..hi]);
    fill_zero(&mut s3[lo..hi]);
    fill_zero(&mut counts[lo..hi]);
    for i in 0..n {
        let k = widen_usize(keys[i]);
        if !key_in_range(k, g) {
            return BAD_RANGE;
        }
        if k >= lo && k < hi {
            bump(&mut counts[k]);
            if !add_i32_checked(&mut s1[k], a[i]) {
                return OVERFLOW;
            }
            if !add_i32_checked(&mut s2[k], b[i]) {
                return OVERFLOW;
            }
            add_f64(&mut s3[k], c[i]);
        }
    }
    OK
}
