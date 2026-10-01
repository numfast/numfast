// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Small numerics shared by kernels: widening casts, overflow-checked
//! radix element, ASCII digit test, signed int32 limit.
//!
//! All hot and single-branch — `#[inline(always)]`; the fused loops
//! that consume them stay flat in their own kernels.

/// Widen a dense code for direct indexing. Hot, zero-cost.
#[inline(always)]
pub fn widen_usize(k: i32) -> usize {
    k as usize
}

/// Widen a key/code into an i64 lane. Hot, zero-cost.
#[inline(always)]
pub fn widen_i64(k: i32) -> i64 {
    k as i64
}

/// One mixed-radix element `a * m + b` with negativity + int32-overflow
/// rejection. Hot. `None` maps to `errors::BAD_RANGE` at the kernel.
#[inline(always)]
pub fn pack_elem(a: i32, b: i32, m: i64) -> Option<i32> {
    if a < 0 || b < 0 {
        return None;
    }
    let v = a as i64 * m + b as i64;
    if v > i32::MAX as i64 {
        return None;
    }
    Some(v as i32)
}

/// ASCII decimal digit (`0-9` only; non-ASCII is invalid, same as the
/// prod `ascii_is_decimal` branch). Hot: range form (LLVM folds the two
/// compares; measured faster than the wrapping_sub single-compare form in
/// the pattern digit loop on this toolchain — see recon matrix).
#[inline(always)]
pub fn is_digit_byte(b: u8) -> bool {
    (b'0'..=b'9').contains(&b)
}

/// Parse magnitude limit for the pattern body: `-2^31` allowed for the
/// negative branch, `2^31 - 1` otherwise. Hot.
#[inline(always)]
pub fn int_limit(neg: bool) -> i64 {
    if neg {
        1i64 << 31
    } else {
        i32::MAX as i64
    }
}
