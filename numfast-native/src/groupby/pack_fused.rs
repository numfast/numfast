// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `pack_fused` mechanic (P4, experimental): composite key packing +
//! aggregation in ONE pass, no intermediate packed-key buffer.
//!
//! Generic over value dtype (i32 / f64 lanes), never over queries:
//! any (k1, k2, m2) mixed-radix composite `pk = k1*m2 + k2` feeding a
//! dense scatter. Reuses the `pack_elem` / `widen_usize` / `fill_zero` /
//! `add_*` / `bump` atoms; sequential `i = 0..n` order matches
//! `dense_scatter`, so float results are bit-exact.
//!
//! Contracts: negative k1/k2 or int32 pack overflow -> `BAD_RANGE`
//! (same as `pack_codes`); packed key out of `[0, g)` -> `BAD_RANGE`
//! (explicit hot bounds check, no panic via FFI); int32 lane overflow
//! -> `OVERFLOW` (checked add, never wraps). Returns `OK` / `BAD_RANGE`
//! / `OVERFLOW`.

use crate::core::checks::key_in_range;
use crate::core::errors::{BAD_RANGE, OK, OVERFLOW};
use crate::core::numeric::{pack_elem, widen_usize};
use crate::core::validity::fill_zero;
use crate::series::reduce::{add_f64, add_i32_checked, bump};

/// Fused pack + i32 sum+count. Returns `OK` / `BAD_RANGE` / `OVERFLOW`.
#[inline]
pub fn pack_fused_i32(
    k1: &[i32],
    k2: &[i32],
    m2: i32,
    vals: &[i32],
    g: usize,
    sums: &mut [i32],
    counts: &mut [i64],
) -> i32 {
    debug_assert_eq!(k1.len(), k2.len());
    debug_assert_eq!(k1.len(), vals.len());
    debug_assert!(sums.len() >= g);
    debug_assert!(counts.len() >= g);
    fill_zero(&mut sums[..g]);
    fill_zero(&mut counts[..g]);
    let m = m2 as i64;
    for i in 0..k1.len() {
        let pk = match pack_elem(k1[i], k2[i], m) {
            Some(v) => v,
            None => return BAD_RANGE,
        };
        let k = widen_usize(pk);
        if !key_in_range(k, g) {
            return BAD_RANGE;
        }
        if !add_i32_checked(&mut sums[k], vals[i]) {
            return OVERFLOW;
        }
        bump(&mut counts[k]);
    }
    OK
}

/// Fused pack + f64 sum+count. Returns `OK` / `BAD_RANGE`.
#[inline]
pub fn pack_fused_f64(
    k1: &[i32],
    k2: &[i32],
    m2: i32,
    vals: &[f64],
    g: usize,
    sums: &mut [f64],
    counts: &mut [i64],
) -> i32 {
    debug_assert_eq!(k1.len(), k2.len());
    debug_assert_eq!(k1.len(), vals.len());
    debug_assert!(sums.len() >= g);
    debug_assert!(counts.len() >= g);
    fill_zero(&mut sums[..g]);
    fill_zero(&mut counts[..g]);
    let m = m2 as i64;
    for i in 0..k1.len() {
        let pk = match pack_elem(k1[i], k2[i], m) {
            Some(v) => v,
            None => return BAD_RANGE,
        };
        let k = widen_usize(pk);
        if !key_in_range(k, g) {
            return BAD_RANGE;
        }
        add_f64(&mut sums[k], vals[i]);
        bump(&mut counts[k]);
    }
    OK
}
