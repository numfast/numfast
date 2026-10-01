// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `buffer_compact` mechanic: forward compaction of M-sized dense
//! aggregation state into `(ukeys, counts, sums)` in one pass,
//! ascending code order by construction.
//!
//! Single generic lane param `S` (monomorphizes to the same flat loop
//! as the two hand-written copies it replaces). Write frontier never
//! overtakes unread input. Hot — this fn IS the loop.

use crate::core::validity::is_present;

/// Returns `ng` (number of present groups emitted).
/// `counts_m`/`sums_m` have equal len; outputs need `m` lanes worst-case.
#[inline]
pub fn buffer_compact<S: Copy>(
    counts_m: &[i64],
    sums_m: &[S],
    ukeys: &mut [i64],
    counts: &mut [i64],
    sums: &mut [S],
) -> i64 {
    debug_assert_eq!(counts_m.len(), sums_m.len());
    let mut ng: i64 = 0;
    for j in 0..counts_m.len() {
        let c = counts_m[j];
        if is_present(c) {
            let w = ng as usize;
            ukeys[w] = j as i64;
            counts[w] = c;
            sums[w] = sums_m[j];
            ng += 1;
        }
    }
    ng
}
