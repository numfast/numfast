// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `segmented` mechanic: P1 bounds-only reduce (spec variant A).
//!
//! Frozen semantics (parity with `Relational/Segmented/_lib/segmented.py`):
//! `values[n]` (f32/i32) + `bounds[M+1]` (u32, nondecreasing,
//! `bounds[M] == n`) -> `out[M]`. Ops: sum | count | min | max
//! (`mean` is not an op). `count` lands in u32 lanes (own symbol);
//! f32 sums accumulate lane-by-lane in `s..e` order (IEEE-propagate:
//! NaN/Inf ride as-is, empty group -> `+0.0`); i32 sums accumulate
//! in i64 and clamp to `[I32MIN, I32MAX]` (saturate, never wrap);
//! min/max over an empty group is an error; `n == 0` requires
//! `M == 0`; `n <= 4194240` per dispatch (chunkable outside).
//! Traversal is `g = 0..m` with sequential lanes inside each group:
//! strided order, deterministic across runs and threads.
//!
//! Same no-alloc `core`-only discipline as the other kernels: the
//! caller owns all buffers; on success the `m` lanes are overwritten,
//! on pre-validated errors (min/max empty, bad op) outputs are
//! untouched, on streaming-validation errors lanes before the abort
//! point may be written (same rule as the sorted-run speculative
//! abort). No graph/SSSP/router vocabulary here: this is a
//! generic bounds reduce.

use crate::core::errors::{BAD_RANGE, MALFORMED, OK};

/// Max rows per dispatch (frozen P1 limit, chunkable outside).
pub const MAX_DISPATCH_N: usize = 4194240;

/// Op codes for [`segment_reduce_f32`] / [`segment_reduce_i32`]:
/// 0 sum, 2 min, 3 max. Code 1 (`count`) is rejected here with
/// `BAD_RANGE` (count lands in u32 lanes, use [`segment_count`]);
/// any other code (including a `mean` request) is `BAD_RANGE`.
pub const SEG_SUM: u32 = 0;
pub const SEG_MIN: u32 = 2;
pub const SEG_MAX: u32 = 3;

const I32MIN_I64: i64 = -2147483648;
const I32MAX_I64: i64 = 2147483647;

/// NaN-propagating f32 min (`np.minimum` parity: NaN in -> NaN out;
/// +-0 resolved by total order, so `min(-0, +0) == -0`).
/// (`f32::minimum` stays unstable on this toolchain, hence the atom.)
#[inline]
fn fmin_nan(a: f32, b: f32) -> f32 {
    use core::cmp::Ordering;
    if a.is_nan() {
        return a;
    }
    if b.is_nan() {
        return b;
    }
    if a.total_cmp(&b) != Ordering::Greater {
        a
    } else {
        b
    }
}

/// NaN-propagating f32 max (`np.maximum` parity; `max(-0, +0) == +0`).
#[inline]
fn fmax_nan(a: f32, b: f32) -> f32 {
    use core::cmp::Ordering;
    if a.is_nan() {
        return a;
    }
    if b.is_nan() {
        return b;
    }
    if a.total_cmp(&b) != Ordering::Less {
        a
    } else {
        b
    }
}

/// Shared bounds gate: `n <= MAX_DISPATCH_N`, lanes inside `[0, n]`,
/// nondecreasing, `bounds[m] == n`, `n == 0` requires `m == 0`.
/// Caller guarantees `bounds.len() == m + 1`.
#[inline]
fn check_bounds(bounds: &[u32], n: usize) -> i32 {
    if n > MAX_DISPATCH_N {
        return MALFORMED;
    }
    let m = bounds.len() - 1;
    if n == 0 {
        return if m == 0 { OK } else { BAD_RANGE };
    }
    for i in 0..bounds.len() {
        if (bounds[i] as usize) > n {
            return BAD_RANGE;
        }
        if i > 0 && bounds[i] < bounds[i - 1] {
            return BAD_RANGE;
        }
    }
    if (bounds[m] as usize) != n {
        return BAD_RANGE;
    }
    OK
}

/// `count` lane: `out[g] = bounds[g+1] - bounds[g]` (u32, exact:
/// the gate pins `bounds[m] == n <= MAX_DISPATCH_N`, so every diff
/// fits). Caller guarantees `bounds.len() == out.len() + 1`.
#[inline]
pub fn segment_count(bounds: &[u32], n: usize, out: &mut [u32]) -> i32 {
    debug_assert_eq!(bounds.len(), out.len() + 1);
    let rc = check_bounds(bounds, n);
    if rc != OK {
        return rc;
    }
    for g in 0..out.len() {
        out[g] = bounds[g + 1] - bounds[g];
    }
    OK
}

/// f32 reduce: `op` is `SEG_SUM` / `SEG_MIN` / `SEG_MAX`, anything
/// else (including 1 = count) is `BAD_RANGE`. Sum accumulates in f32
/// lane order (IEEE-propagate, empty -> `+0.0`); min/max use the
/// NaN-propagating lane (`np.minimum`/`np.maximum` parity) and reject
/// empty groups with `BAD_RANGE` leaving `out` untouched.
/// Caller guarantees `bounds.len() == out.len() + 1`,
/// `values.len() == n`.
#[inline]
pub fn segment_reduce_f32(
    values: &[f32],
    bounds: &[u32],
    op: u32,
    out: &mut [f32],
) -> i32 {
    debug_assert_eq!(bounds.len(), out.len() + 1);
    let rc = check_bounds(bounds, values.len());
    if rc != OK {
        return rc;
    }
    match op {
        SEG_SUM => {
            for g in 0..out.len() {
                let s = bounds[g] as usize;
                let e = bounds[g + 1] as usize;
                let mut acc: f32 = 0.0;
                for i in s..e {
                    acc += values[i];
                }
                out[g] = acc;
            }
            OK
        }
        SEG_MIN | SEG_MAX => {
            for g in 0..out.len() {
                if bounds[g] == bounds[g + 1] {
                    return BAD_RANGE;
                }
            }
            for g in 0..out.len() {
                let s = bounds[g] as usize;
                let e = bounds[g + 1] as usize;
                let mut acc = values[s];
                for i in (s + 1)..e {
                    acc = if op == SEG_MIN {
                        fmin_nan(acc, values[i])
                    } else {
                        fmax_nan(acc, values[i])
                    };
                }
                out[g] = acc;
            }
            OK
        }
        _ => BAD_RANGE,
    }
}

/// i32 reduce: same gate and op table as [`segment_reduce_f32`].
/// Sum accumulates in i64 and saturates to
/// `[I32MIN, I32MAX]` (never wraps); min/max are plain lane
/// comparisons and reject empty groups with `BAD_RANGE` leaving
/// `out` untouched. Caller guarantees `bounds.len() == out.len() + 1`,
/// `values.len() == n`.
#[inline]
pub fn segment_reduce_i32(
    values: &[i32],
    bounds: &[u32],
    op: u32,
    out: &mut [i32],
) -> i32 {
    debug_assert_eq!(bounds.len(), out.len() + 1);
    let rc = check_bounds(bounds, values.len());
    if rc != OK {
        return rc;
    }
    match op {
        SEG_SUM => {
            for g in 0..out.len() {
                let s = bounds[g] as usize;
                let e = bounds[g + 1] as usize;
                let mut acc: i64 = 0;
                for i in s..e {
                    acc += values[i] as i64;
                }
                out[g] = acc.clamp(I32MIN_I64, I32MAX_I64) as i32;
            }
            OK
        }
        SEG_MIN | SEG_MAX => {
            for g in 0..out.len() {
                if bounds[g] == bounds[g + 1] {
                    return BAD_RANGE;
                }
            }
            for g in 0..out.len() {
                let s = bounds[g] as usize;
                let e = bounds[g + 1] as usize;
                let mut acc = values[s];
                for i in (s + 1)..e {
                    let v = values[i];
                    if (op == SEG_MIN && v < acc) || (op == SEG_MAX && v > acc) {
                        acc = v;
                    }
                }
                out[g] = acc;
            }
            OK
        }
        _ => BAD_RANGE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_all_ops() {
        let v32: Vec<i32> = vec![];
        let vf: Vec<f32> = vec![];
        let b: Vec<u32> = vec![0];
        let mut oi: Vec<i32> = vec![];
        let mut of: Vec<f32> = vec![];
        let mut oc: Vec<u32> = vec![];
        assert_eq!(segment_count(&b, 0, &mut oc), OK);
        assert_eq!(segment_reduce_i32(&v32, &b, SEG_SUM, &mut oi), OK);
        assert_eq!(segment_reduce_i32(&v32, &b, SEG_MIN, &mut oi), OK);
        assert_eq!(segment_reduce_i32(&v32, &b, SEG_MAX, &mut oi), OK);
        assert_eq!(segment_reduce_f32(&vf, &b, SEG_SUM, &mut of), OK);
        assert_eq!(segment_reduce_f32(&vf, &b, SEG_MIN, &mut of), OK);
        assert_eq!(segment_reduce_f32(&vf, &b, SEG_MAX, &mut of), OK);
    }

    #[test]
    fn empty_requires_m0() {
        let v: Vec<i32> = vec![];
        let b: Vec<u32> = vec![0, 0];
        let mut o: Vec<i32> = vec![0];
        let mut c: Vec<u32> = vec![0];
        assert_eq!(segment_count(&b, 0, &mut c), BAD_RANGE);
        assert_eq!(segment_reduce_i32(&v, &b, SEG_SUM, &mut o), BAD_RANGE);
    }

    #[test]
    fn monotone_and_terminal() {
        let v: Vec<i32> = vec![1, 2, 3];
        let mut o: Vec<i32> = vec![0];
        let mut o2: Vec<i32> = vec![0; 2];
        let mut c: Vec<u32> = vec![0];
        // Regression: bounds[1] < bounds[0].
        assert_eq!(
            segment_reduce_i32(&v, &[2, 1, 3], SEG_SUM, &mut o2),
            BAD_RANGE
        );
        // Terminal lane != n.
        assert_eq!(
            segment_reduce_i32(&v, &[0, 2], SEG_SUM, &mut o),
            BAD_RANGE
        );
        assert_eq!(segment_count(&[0, 2], 3, &mut c), BAD_RANGE);
        // Lane outside [0, n].
        assert_eq!(
            segment_reduce_i32(&v, &[0, 4], SEG_SUM, &mut o),
            BAD_RANGE
        );
    }

    #[test]
    fn limit_is_malformed() {
        // No multi-megabyte buffers: count carries `n` as a scalar,
        // so the dispatch gate is observable without allocation.
        let b: Vec<u32> = vec![0];
        let mut c: Vec<u32> = vec![];
        assert_eq!(segment_count(&b, MAX_DISPATCH_N + 1, &mut c), MALFORMED);
        assert_eq!(segment_count(&b, 0, &mut c), OK);
    }

    #[test]
    fn count_lane() {
        let b: Vec<u32> = vec![0, 0, 3, 5];
        let mut o: Vec<u32> = vec![9, 9, 9];
        assert_eq!(segment_count(&b, 5, &mut o), OK);
        assert_eq!(o, vec![0, 3, 2]);
    }

    #[test]
    fn i32_sum_saturates_both_sides() {
        let v: Vec<i32> = vec![i32::MAX, 1, i32::MIN, -1, 7];
        let b: Vec<u32> = vec![0, 2, 4, 5];
        let mut o: Vec<i32> = vec![0; 3];
        assert_eq!(segment_reduce_i32(&v, &b, SEG_SUM, &mut o), OK);
        assert_eq!(o, vec![i32::MAX, i32::MIN, 7]);
    }

    #[test]
    fn f32_sum_ieee_propagates() {
        let v: Vec<f32> = vec![1.0, f32::NAN, f32::INFINITY, 1.0, 5.0];
        let b: Vec<u32> = vec![0, 2, 4, 4, 5];
        let mut o: Vec<f32> = vec![0.0; 4];
        assert_eq!(segment_reduce_f32(&v, &b, SEG_SUM, &mut o), OK);
        assert!(o[0].is_nan());
        assert_eq!(o[1], f32::INFINITY);
        assert_eq!(o[2], 0.0);
        assert_eq!(o[3], 5.0);
    }

    #[test]
    fn min_max_and_empty_rejects_untouched() {
        let v: Vec<i32> = vec![3, 1, 2];
        let b: Vec<u32> = vec![0, 2, 3];
        let mut o: Vec<i32> = vec![0; 2];
        assert_eq!(segment_reduce_i32(&v, &b, SEG_MIN, &mut o), OK);
        assert_eq!(o, vec![1, 2]);
        assert_eq!(segment_reduce_i32(&v, &b, SEG_MAX, &mut o), OK);
        assert_eq!(o, vec![3, 2]);
        // Empty group: error, outputs untouched.
        let be: Vec<u32> = vec![0, 0, 3];
        let mut oe: Vec<i32> = vec![-7, -7];
        assert_eq!(segment_reduce_i32(&v, &be, SEG_MIN, &mut oe), BAD_RANGE);
        assert_eq!(oe, vec![-7, -7]);
        assert_eq!(segment_reduce_i32(&v, &be, SEG_MAX, &mut oe), BAD_RANGE);
        assert_eq!(oe, vec![-7, -7]);
        let mut of: Vec<f32> = vec![-7.0, -7.0];
        let vf: Vec<f32> = vec![3.0, 1.0, 2.0];
        assert_eq!(segment_reduce_f32(&vf, &be, SEG_MIN, &mut of), BAD_RANGE);
        assert_eq!(of, vec![-7.0, -7.0]);
    }

    #[test]
    fn f32_min_max_nan_propagates() {
        let v: Vec<f32> = vec![1.0, f32::NAN, 2.0, 0.5];
        let b: Vec<u32> = vec![0, 3, 4];
        let mut o: Vec<f32> = vec![0.0; 2];
        assert_eq!(segment_reduce_f32(&v, &b, SEG_MIN, &mut o), OK);
        assert!(o[0].is_nan());
        assert_eq!(segment_reduce_f32(&v, &b, SEG_MAX, &mut o), OK);
        assert!(o[0].is_nan());
        assert_eq!(o[1], 0.5);
    }

    #[test]
    fn bad_ops_rejected() {
        let v: Vec<i32> = vec![1, 2];
        let b: Vec<u32> = vec![0, 2];
        let mut o: Vec<i32> = vec![0];
        // 1 = count lives in the u32 lane; 4/99 stand in for mean/unknown.
        for op in [1, 4, 99] {
            assert_eq!(segment_reduce_i32(&v, &b, op, &mut o), BAD_RANGE);
        }
        let vf: Vec<f32> = vec![1.0, 2.0];
        let mut of: Vec<f32> = vec![0.0];
        for op in [1, 4, 99] {
            assert_eq!(segment_reduce_f32(&vf, &b, op, &mut of), BAD_RANGE);
        }
    }

    #[test]
    fn strided_order_deterministic() {
        // Interleaved magnitudes: the expected lanes pin the exact
        // accumulation order (sequential within group, groups ascending).
        let v: Vec<i32> = vec![10, -3, 4, 100, -50, 7, 0, 9];
        let b: Vec<u32> = vec![0, 3, 3, 6, 8];
        let mut a: Vec<i32> = vec![0; 4];
        let mut c: Vec<u32> = vec![0; 4];
        assert_eq!(segment_reduce_i32(&v, &b, SEG_SUM, &mut a), OK);
        assert_eq!(a, vec![11, 0, 57, 9]);
        assert_eq!(segment_count(&b, 8, &mut c), OK);
        assert_eq!(c, vec![3, 0, 3, 2]);
        // Second pass over the same buffers: bit-identical.
        let mut a2: Vec<i32> = vec![0; 4];
        assert_eq!(segment_reduce_i32(&v, &b, SEG_SUM, &mut a2), OK);
        assert_eq!(a, a2);
    }

    #[test]
    fn nonzero_origin_bounds_allowed() {
        // Python parity: bounds[0] > 0 is legal (leading rows uncovered).
        let v: Vec<i32> = vec![9, 1, 2];
        let b: Vec<u32> = vec![1, 3];
        let mut o: Vec<i32> = vec![0];
        assert_eq!(segment_reduce_i32(&v, &b, SEG_SUM, &mut o), OK);
        assert_eq!(o, vec![3]);
    }
}
