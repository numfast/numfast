// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `scan_kernel` atoms: ordered key-run traversal for sorted inputs.
//! Consumed by [`crate::groupby::sorted::sorted_runs`]. Hot.

/// True when the key moved backwards — speculative abort point.
#[inline(always)]
pub fn key_regressed(k: i64, cur: i64) -> bool {
    k < cur
}

/// True when the key advanced — flush the finished run.
#[inline(always)]
pub fn key_changed(k: i64, cur: i64) -> bool {
    k != cur
}

/// Flush one finished run to the dense outputs at slot `ng`.
/// Single lane-type param; monomorphizes flat for i64/f64.
#[inline(always)]
pub fn flush_run<S: Copy>(
    ukeys: &mut [i64],
    sums: &mut [S],
    counts: &mut [i64],
    ng: i64,
    key: i64,
    acc: S,
    cnt: i64,
) {
    let w = ng as usize;
    ukeys[w] = key;
    sums[w] = acc;
    counts[w] = cnt;
}
