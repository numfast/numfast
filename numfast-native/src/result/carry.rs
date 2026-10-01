// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Columnar result carry: compact (ukeys, counts, sums) from M-sized
//! dense aggregation state via [`crate::core::compact::buffer_compact`].
//! Native counterpart of the `ColumnCarry` concept (O(1) Python objects,
//! no per-group dict): the carrier holds only columnar buffers;
//! materialization (`to_dict`) stays an explicit off-hot-path step.
//!
//! Mirrors prod `dense_by_code_present` forward-compaction (flatnonzero
//! to sorted present codes in one forward pass): ukeys are emitted in
//! ascending code order by construction.
//!
//! Layout (all contiguous, caller-owned):
//! - `counts_m[0..m]`: int64 dense counts (0 = absent group)
//! - `sums_m[0..m]`: int64 / float64 dense sums (0 at absent lanes)
//! - `ukeys[0..ng]`, `counts[0..ng]`, `sums[0..ng]`: compact outputs
//!   (caller allocates m lanes worst-case; returns exact ng)

use crate::core::compact::buffer_compact;

/// Returns ng (>= 0): number of present groups emitted.
#[inline]
pub fn carry_build_i64(
    counts_m: &[i64],
    sums_m: &[i64],
    ukeys: &mut [i64],
    counts: &mut [i64],
    sums: &mut [i64],
) -> i64 {
    buffer_compact(counts_m, sums_m, ukeys, counts, sums)
}

/// Float variant: same forward compaction, f64 sums.
#[inline]
pub fn carry_build_f64(
    counts_m: &[i64],
    sums_m: &[f64],
    ukeys: &mut [i64],
    counts: &mut [i64],
    sums: &mut [f64],
) -> i64 {
    buffer_compact(counts_m, sums_m, ukeys, counts, sums)
}

/// Previous module paths kept for internal import compat.
#[inline]
pub fn carry_result_build_i64(
    counts_m: &[i64],
    sums_m: &[i64],
    ukeys: &mut [i64],
    counts: &mut [i64],
    sums: &mut [i64],
) -> i64 {
    carry_build_i64(counts_m, sums_m, ukeys, counts, sums)
}

/// Previous module path kept for internal import compat.
#[inline]
pub fn carry_result_build_f64(
    counts_m: &[i64],
    sums_m: &[f64],
    ukeys: &mut [i64],
    counts: &mut [i64],
    sums: &mut [f64],
) -> i64 {
    carry_build_f64(counts_m, sums_m, ukeys, counts, sums)
}
