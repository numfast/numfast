// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `scaled` mechanic (research-only): scaled-integer dense group-by.
//!
//! Schema semantics (SPEC v0.2 untouched):
//! logical decimal (v3, 6 digits) -> physical int64 ticks,
//! `tick = round((logical - offset) * M)`, `M = 1_000_000`, `offset = 0`
//! in this research. Native sums ticks with integer arithmetic;
//! `mean = int_sum / count` (`/M` at the API boundary only).
//! Production ABI untouched: existing 8 symbols byte-identical;
//! new symbols live in `variants.rs` + `lib.rs` (additive).
//!
//! Hot loops stay flat and specialized (same rule as dense/fused);
//! sharing happens through the `series::reduce` checked-add atoms.

use crate::core::checks::key_in_range;
use crate::core::errors::{BAD_RANGE, OK, OVERFLOW, UNSORTED};
use crate::core::numeric::widen_usize;
use crate::core::validity::fill_zero;
use crate::series::reduce::{add_i32_checked, add_i64_checked, bump};

/// Unchecked dense scatter over int64 ticks. Mirrors
/// [`crate::groupby::dense::dense_scatter`] contract: out-of-range key
/// traps via safe Rust (never UB); integer overflow is an explicit
/// `OVERFLOW` error (never wraps). Returns `OK` / `OVERFLOW`.
#[inline]
pub fn dense_scatter_i64(
    keys: &[i32],
    ticks: &[i64],
    g: usize,
    sums: &mut [i64],
    counts: &mut [i64],
) -> i32 {
    debug_assert_eq!(keys.len(), ticks.len());
    debug_assert!(sums.len() >= g);
    debug_assert!(counts.len() >= g);
    fill_zero(&mut sums[..g]);
    fill_zero(&mut counts[..g]);
    for i in 0..keys.len() {
        let k = widen_usize(keys[i]); // contract: k < g
        if !add_i64_checked(&mut sums[k], ticks[i]) {
            return OVERFLOW;
        }
        bump(&mut counts[k]);
    }
    OK
}

/// Range-checked sibling: out-of-range key returns `BAD_RANGE`
/// (no panic via FFI); overflow returns `OVERFLOW`.
#[inline]
pub fn checked_scatter_i64(
    keys: &[i32],
    ticks: &[i64],
    g: usize,
    sums: &mut [i64],
    counts: &mut [i64],
) -> i32 {
    debug_assert_eq!(keys.len(), ticks.len());
    debug_assert!(sums.len() >= g);
    debug_assert!(counts.len() >= g);
    fill_zero(&mut sums[..g]);
    fill_zero(&mut counts[..g]);
    for i in 0..keys.len() {
        let k = widen_usize(keys[i]);
        if !key_in_range(k, g) {
            return BAD_RANGE;
        }
        if !add_i64_checked(&mut sums[k], ticks[i]) {
            return OVERFLOW;
        }
        bump(&mut counts[k]);
    }
    OK
}

/// int32-accumulator variant: same traversal, `sums` are int32 lanes.
/// v3-scale ticks overflow int32 by design on mass data — the point is
/// the explicit `OVERFLOW` contract (range proof in SEMANTICS.md).
/// Returns `OK` / `BAD_RANGE` / `OVERFLOW`.
#[inline]
pub fn checked_scatter_i32(
    keys: &[i32],
    ticks: &[i32],
    g: usize,
    sums: &mut [i32],
    counts: &mut [i64],
) -> i32 {
    debug_assert_eq!(keys.len(), ticks.len());
    debug_assert!(sums.len() >= g);
    debug_assert!(counts.len() >= g);
    fill_zero(&mut sums[..g]);
    fill_zero(&mut counts[..g]);
    for i in 0..keys.len() {
        let k = widen_usize(keys[i]);
        if !key_in_range(k, g) {
            return BAD_RANGE;
        }
        if !add_i32_checked(&mut sums[k], ticks[i]) {
            return OVERFLOW;
        }
        bump(&mut counts[k]);
    }
    OK
}

/// Sorted-dense over int64 ticks: one pass over nondecreasing keys,
/// per-run accumulation, dense-slot writes (`sums[cur] = run_sum`,
/// each key occurs in exactly one run). Unsorted input aborts with
/// `UNSORTED` (outputs past the abort point are caller-owned garbage,
/// same rule as `sorted_runs`). Returns `OK` / `UNSORTED` / `OVERFLOW`.
#[inline]
pub fn sorted_dense_i64(
    keys: &[i32],
    ticks: &[i64],
    g: usize,
    sums: &mut [i64],
    counts: &mut [i64],
) -> i32 {
    debug_assert_eq!(keys.len(), ticks.len());
    debug_assert!(sums.len() >= g);
    debug_assert!(counts.len() >= g);
    fill_zero(&mut sums[..g]);
    fill_zero(&mut counts[..g]);
    let n = keys.len();
    if n == 0 {
        return OK;
    }
    let mut cur = keys[0];
    let mut acc = ticks[0];
    let mut cnt: i64 = 1;
    for i in 1..n {
        let k = keys[i];
        if k < cur {
            return UNSORTED;
        }
        if k != cur {
            let w = widen_usize(cur); // sorted path: caller guarantees k < g
            sums[w] = acc;
            counts[w] = cnt;
            cur = k;
            acc = ticks[i];
            cnt = 1;
        } else {
            match acc.checked_add(ticks[i]) {
                Some(v) => acc = v,
                None => return OVERFLOW,
            }
            bump(&mut cnt);
        }
    }
    let w = widen_usize(cur);
    sums[w] = acc;
    counts[w] = cnt;
    OK
}

/// Sorted-dense over int32 ticks (same contract as
/// [`sorted_dense_i64`], int32 lanes). Returns `OK` / `UNSORTED` /
/// `OVERFLOW`.
#[inline]
pub fn sorted_dense_i32(
    keys: &[i32],
    ticks: &[i32],
    g: usize,
    sums: &mut [i32],
    counts: &mut [i64],
) -> i32 {
    debug_assert_eq!(keys.len(), ticks.len());
    debug_assert!(sums.len() >= g);
    debug_assert!(counts.len() >= g);
    fill_zero(&mut sums[..g]);
    fill_zero(&mut counts[..g]);
    let n = keys.len();
    if n == 0 {
        return OK;
    }
    let mut cur = keys[0];
    let mut acc = ticks[0];
    let mut cnt: i64 = 1;
    for i in 1..n {
        let k = keys[i];
        if k < cur {
            return UNSORTED;
        }
        if k != cur {
            let w = widen_usize(cur);
            sums[w] = acc;
            counts[w] = cnt;
            cur = k;
            acc = ticks[i];
            cnt = 1;
        } else {
            match acc.checked_add(ticks[i]) {
                Some(v) => acc = v,
                None => return OVERFLOW,
            }
            bump(&mut cnt);
        }
    }
    let w = widen_usize(cur);
    sums[w] = acc;
    counts[w] = cnt;
    OK
}

/// Sorted-dense over float64: same traversal as [`sorted_dense_i64`],
/// sequential accumulation within runs (bit-exact vs
/// `sorted_run_aggregate_f64` compact sums). No overflow concept
/// (`inf` on extreme inputs, same as the existing float kernels).
/// Returns `OK` / `UNSORTED`.
#[inline]
pub fn sorted_dense_f64(
    keys: &[i32],
    values: &[f64],
    g: usize,
    sums: &mut [f64],
    counts: &mut [i64],
) -> i32 {
    use crate::series::reduce::add_f64;
    debug_assert_eq!(keys.len(), values.len());
    debug_assert!(sums.len() >= g);
    debug_assert!(counts.len() >= g);
    fill_zero(&mut sums[..g]);
    fill_zero(&mut counts[..g]);
    let n = keys.len();
    if n == 0 {
        return OK;
    }
    let mut cur = keys[0];
    let mut acc = values[0];
    let mut cnt: i64 = 1;
    for i in 1..n {
        let k = keys[i];
        if k < cur {
            return UNSORTED;
        }
        if k != cur {
            let w = widen_usize(cur);
            sums[w] = acc;
            counts[w] = cnt;
            cur = k;
            acc = values[i];
            cnt = 1;
        } else {
            add_f64(&mut acc, values[i]);
            bump(&mut cnt);
        }
    }
    let w = widen_usize(cur);
    sums[w] = acc;
    counts[w] = cnt;
    OK
}
