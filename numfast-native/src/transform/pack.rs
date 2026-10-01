// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `pack_codes` mechanic: int32-direct mixed-radix composite
//! out[i] = k1[i]*m2 + k2[i]. Mirrors prod `_pack_i32_direct`
//! (production Q2 path): single pass, caller-allocated int32 output,
//! overflow-safe (i64 arithmetic + explicit bound check per element,
//! never wraps). Bit-identical to the int64 radix composite by
//! construction when the caller bound k1max*m2+k2max <= 2^31-1 holds.

use crate::core::errors::BAD_RANGE;
use crate::core::numeric::pack_elem;

/// Returns 0 ok, `BAD_RANGE` on negative input or int32 overflow.
#[inline]
pub fn pack_codes(k1: &[i32], k2: &[i32], m2: i32, out: &mut [i32]) -> i32 {
    debug_assert!(out.len() >= k1.len());
    debug_assert_eq!(k1.len(), k2.len());
    let m = m2 as i64;
    for i in 0..k1.len() {
        match pack_elem(k1[i], k2[i], m) {
            Some(v) => out[i] = v,
            None => return BAD_RANGE,
        }
    }
    0
}

/// Previous module path kept for internal import compat.
#[inline]
pub fn pack_i32_direct(k1: &[i32], k2: &[i32], m2: i32, out: &mut [i32]) -> i32 {
    pack_codes(k1, k2, m2, out)
}
