// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `reduce_kernel` atoms: lane accumulation for scatter/reduce loops.
//! Consumed by dense, fused and sorted kernels. Hot — and order matters:
//! callers iterate `i = 0..n` sequentially, so float sums stay bit-exact
//! vs `np.bincount` / the prod numba kernels.

/// `dst += v` for f64 lanes.
#[inline(always)]
pub fn add_f64(dst: &mut f64, v: f64) {
    *dst += v;
}

/// `cnt += 1` for count lanes.
#[inline(always)]
pub fn bump(cnt: &mut i64) {
    *cnt += 1;
}

/// `*dst = checked_add(*dst, v)` for int64 tick lanes (research-only
/// scaled path). Returns false on overflow — caller maps to
/// `errors::OVERFLOW`. Hot.
#[inline(always)]
pub fn add_i64_checked(dst: &mut i64, v: i64) -> bool {
    match dst.checked_add(v) {
        Some(s) => {
            *dst = s;
            true
        }
        None => false,
    }
}

/// int32 sibling of [`add_i64_checked`] (research-only overflow-demo
/// lanes). Returns false on overflow.
#[inline(always)]
pub fn add_i32_checked(dst: &mut i32, v: i32) -> bool {
    match dst.checked_add(v) {
        Some(s) => {
            *dst = s;
            true
        }
        None => false,
    }
}
