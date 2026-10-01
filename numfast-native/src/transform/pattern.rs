// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Prefix+int string encode: bytes + offsets -> int32 codes + validity.
//! Mirrors prod `_encode_pattern_vec` contract (both Arrow and no-Arrow
//! branches): exact prefix byte match; numeric body = optional single '-'
//! + non-empty ASCII digits (0-9 only; non-ASCII -> invalid, same as
//! `ascii_is_decimal`); int32 overflow -> explicit error (never wraps).
//! Overflow is VALUE-based (saturating i64 magnitude vs int32 limit, same
//! as the Arrow `safe cast to int32`): leading zeros never fake an
//! overflow. Non-matching rows -> invalid (None-equivalent), never error.
//! Width sidecar = max digit-core width over valid rows (-1 if none valid);
//! lossless restore `f"{prefix}{code:0{width}d}"` for fixed-width patterns.
//!
//! The fused digit loop stays in this kernel (digit validity, magnitude,
//! width and err_row are one intertwined scan); shared atoms come from
//! [`crate::core::numeric`] and [`crate::core::validity`].

use crate::core::checks::offset_in_bounds;
use crate::core::errors::{BAD_RANGE, MALFORMED, OK};
use crate::core::numeric::{int_limit, is_digit_byte};
use crate::core::validity::clear_slot;

/// Rare path: 19+-digit bodies. Saturating magnitude keeps VALUE semantics
/// exact (leading zeros never fake an overflow, same as the Arrow
/// safe-cast). `None` on the first non-digit (invalid row, never an error).
/// Cold + never inlined: keeps the hot short loop small and separately
/// laid out (icache/alignment), out of the per-row fast path.
#[cold]
#[inline(never)]
fn scan_long(body: &[u8]) -> Option<(i64, i32)> {
    let mut mag: i64 = 0;
    let mut w: i32 = 0;
    for &b in body {
        if !is_digit_byte(b) {
            return None;
        }
        mag = mag.saturating_mul(10).saturating_add((b - b'0') as i64);
        w += 1;
    }
    Some((mag, w))
}

/// Returns 0 ok, `BAD_RANGE` int32 overflow, `MALFORMED` bad offsets.
/// On overflow, rows before `err_row` are written; row `err_row` and
/// later are untouched. Invalid rows are data (valid=0, code=0), never
/// an error.
#[inline]
pub fn pattern_scan_parse(
    data: &[u8],
    offs: &[i32],
    prefix: &[u8],
    codes: &mut [i32],
    valid: &mut [u8],
    width_out: &mut i32,
    err_row_out: &mut i32,
) -> i32 {
    let n = codes.len();
    debug_assert_eq!(valid.len(), n);
    debug_assert!(offs.len() >= n + 1);
    *width_out = -1;
    *err_row_out = -1;
    let total = data.len();
    // Explicit bounds-checks: offsets well-formed and inside data.
    let mut prev = offs[0];
    if !offset_in_bounds(prev, total) {
        return MALFORMED;
    }
    for i in 0..n {
        clear_slot(codes, valid, i);
        let end = offs[i + 1];
        if end < prev || !offset_in_bounds(end, total) {
            return MALFORMED;
        }
        let row = &data[prev as usize..end as usize];
        prev = end;
        if row.len() < prefix.len() || &row[..prefix.len()] != prefix {
            continue;
        }
        let mut body = &row[prefix.len()..];
        let neg = body.first() == Some(&b'-');
        if neg {
            body = &body[1..];
        }
        if body.is_empty() {
            continue;
        }
        // Digit scan, versioned on body length (no per-byte length
        // branch): <=18 digits run the plain Horner loop (exact — 18
        // digits always fit i64, no wrap possible); 19+ digit bodies take
        // the cold saturating path above. A non-digit anywhere -> invalid
        // row (never an error), even past saturation.
        let (mag, w) = if body.len() > 18 {
            match scan_long(body) {
                Some(v) => v,
                None => continue,
            }
        } else {
            let mut mag: i64 = 0;
            let mut w: i32 = 0;
            let mut ok = true;
            for &b in body {
                if !is_digit_byte(b) {
                    ok = false;
                    break;
                }
                mag = mag * 10 + (b - b'0') as i64;
                w += 1;
            }
            if !ok {
                continue;
            }
            (mag, w)
        };
        if mag > int_limit(neg) {
            *err_row_out = i as i32;
            return BAD_RANGE;
        }
        codes[i] = if neg { (-(mag as i64)) as i32 } else { mag as i32 };
        valid[i] = 1;
        if w > *width_out {
            *width_out = w;
        }
    }
    // Validate final offset sentinel.
    if (offs[n] as usize) > total {
        return MALFORMED;
    }
    OK
}

/// Previous module path kept for internal import compat.
#[inline]
pub fn pattern_encode(
    data: &[u8],
    offs: &[i32],
    prefix: &[u8],
    codes: &mut [i32],
    valid: &mut [u8],
    width_out: &mut i32,
    err_row_out: &mut i32,
) -> i32 {
    pattern_scan_parse(data, offs, prefix, codes, valid, width_out, err_row_out)
}
