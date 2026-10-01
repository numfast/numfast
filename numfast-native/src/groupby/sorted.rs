// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `sorted_runs` mechanic: single pass over nondecreasing int32 keys.
//! Mirrors prod `sorted_fused_aggregate` / `_sorted_1i` / `_sorted_1f`
//! (numba kernels): runs flush directly to dense outputs — no M-sized
//! state, no second traversal, no sortedness probe pass. Unsorted input
//! aborts with `UNSORTED_ABORT` (speculative abort; outputs past the
//! abort point are garbage and caller-owned, same as the numba
//! contract); empty input yields ng=0.
//!
//! Single generic lane param `S` (monomorphizes to the same two flat
//! loops as before). Float accumulation order matches the numba kernel
//! (sequential within run) → bit-exact vs the prod kernel.

use core::ops::AddAssign;

use crate::core::errors::UNSORTED_ABORT;
use crate::core::numeric::widen_i64;
use crate::series::reduce::bump;
use crate::series::scan::{flush_run, key_changed, key_regressed};

/// Returns ng (>= 0), `UNSORTED_ABORT` on unsorted input.
#[inline]
pub fn sorted_runs<S: Copy + AddAssign>(
    keys: &[i32],
    vals: &[S],
    ukeys: &mut [i64],
    sums: &mut [S],
    counts: &mut [i64],
) -> i64 {
    debug_assert_eq!(keys.len(), vals.len());
    let n = keys.len();
    if n == 0 {
        return 0;
    }
    let mut cur = widen_i64(keys[0]);
    let mut acc = vals[0];
    let mut cnt: i64 = 1;
    let mut ng: i64 = 0;
    for i in 1..n {
        let k = widen_i64(keys[i]);
        if key_regressed(k, cur) {
            return UNSORTED_ABORT;
        }
        if key_changed(k, cur) {
            flush_run(ukeys, sums, counts, ng, cur, acc, cnt);
            ng += 1;
            cur = k;
            acc = vals[i];
            cnt = 1;
        } else {
            acc += vals[i];
            bump(&mut cnt);
        }
    }
    flush_run(ukeys, sums, counts, ng, cur, acc, cnt);
    ng + 1
}

/// Previous module paths kept for internal import compat.
#[inline]
pub fn sorted_run_aggregate_i64(
    keys: &[i32],
    vals: &[i64],
    ukeys: &mut [i64],
    sums: &mut [i64],
    counts: &mut [i64],
) -> i64 {
    sorted_runs(keys, vals, ukeys, sums, counts)
}

/// Float variant: same traversal, sequential f64 accumulation within runs.
#[inline]
pub fn sorted_run_aggregate_f64(
    keys: &[i32],
    vals: &[f64],
    ukeys: &mut [i64],
    sums: &mut [f64],
    counts: &mut [i64],
) -> i64 {
    sorted_runs(keys, vals, ukeys, sums, counts)
}
