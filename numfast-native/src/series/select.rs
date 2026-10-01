// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `select_kernel` mechanics: boolean selection (IR `filter`) + boolean
//! combine (IR `mask` and/or/not).
//!
//! Single-pass, caller-allocated output, order preserved by construction.
//! Masks are u8 lanes (`0` = drop, nonzero = keep); outputs normalized to
//! `0/1`. Same no-alloc `core`-only discipline as the other kernels:
//! no collections, no syscalls, `std` only for ctypes ergonomics at the FFI.

/// Count kept rows (pure; sizing probe, same number `select_scatter` returns).
#[inline]
pub fn select_count(mask: &[u8]) -> i64 {
    let mut m: i64 = 0;
    for &b in mask.iter() {
        m += (b != 0) as i64;
    }
    m
}

/// Compact `src[i]` where `mask[i] != 0`, order preserved.
/// Returns `m` (rows written). Caller guarantees `out.len() >= src.len()`.
/// Single generic lane param `S` (monomorphizes to the same flat loop as
/// hand-written per-dtype copies). Write frontier never overtakes unread
/// input (compaction, same argument as `core::compact::buffer_compact`).
#[inline]
pub fn select_scatter<S: Copy>(src: &[S], mask: &[u8], out: &mut [S]) -> i64 {
    debug_assert_eq!(src.len(), mask.len());
    debug_assert!(out.len() >= src.len());
    let mut m: usize = 0;
    for i in 0..src.len() {
        if mask[i] != 0 {
            out[m] = src[i];
            m += 1;
        }
    }
    m as i64
}

/// Elementwise AND of two masks, output normalized `0/1`.
#[inline]
pub fn mask_combine_and(a: &[u8], b: &[u8], out: &mut [u8]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(out.len() >= a.len());
    for i in 0..a.len() {
        out[i] = ((a[i] != 0) & (b[i] != 0)) as u8;
    }
}

/// Elementwise OR of two masks, output normalized `0/1`.
#[inline]
pub fn mask_combine_or(a: &[u8], b: &[u8], out: &mut [u8]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(out.len() >= a.len());
    for i in 0..a.len() {
        out[i] = ((a[i] != 0) | (b[i] != 0)) as u8;
    }
}

/// Elementwise NOT of one mask, output normalized `0/1`.
#[inline]
pub fn mask_negate(a: &[u8], out: &mut [u8]) {
    debug_assert!(out.len() >= a.len());
    for i in 0..a.len() {
        out[i] = (a[i] == 0) as u8;
    }
}
