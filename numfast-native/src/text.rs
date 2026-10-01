// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! TEXT scalar kernels: length (code points) + contains/startswith/
//! endswith/equals (affix + equality).
//!
//! Semantics (frozen):
//! - `length` = Unicode code points per row (validated UTF-8 walk),
//!   NOT bytes. Matches `np.char.str_len` + Arrow `utf8_length`.
//!   Wasm rule: never `str::chars().count()` / `str::from_utf8` in
//!   the row loop — both miscompile on wasm32-unknown-unknown with
//!   the pinned toolchain (rustc 1.98.1): correct bytes in,
//!   spurious MALFORMED out at n >= ~263 rows. The indexed byte
//!   walk below (`utf8_count_validated`, same shape as
//!   `memmem_first`) is bit-exact on all targets (native + wasm
//!   parity green).
//! - `contains` = UTF-8 substring search, case-sensitive. Empty needle
//!   matches every row (matches `dict_contains_impl` + `np.char.find`).
//!   Row validity uses the same byte-walk validator (never
//!   `from_utf8` in the loop); the needle is checked once up front.
//! - `startswith`/`endswith` = anchored affix match, case-sensitive
//!   (`str.startswith` / `str.endswith` + `np.char.startswith` /
//!   `np.char.endswith` parity). Empty affix matches every row. Needle
//!   checked once up front (non-UTF8 -> BAD_RANGE, same as `contains`).
//! - `equals` = byte-exact row equality (== valid UTF-8 bytes, hence ==
//!   code points; `str ==` + `np.char.equal` parity). Non-UTF8 needle
//!   -> BAD_RANGE, same surface as `contains`.
//! - Validity travels host-side (DELTA-3 sidecar): these symbols never
//!   see validity; invalid rows are masked by the caller, never here.
//! - Offsets must be nondecreasing int32 boundaries inside data
//!   (same wire as `nf_pattern_encode`); violations -> MALFORMED.
//! - Invalid UTF-8 in a row -> MALFORMED (explicit, never silent;
//!   unreachable from the Python `str.encode("utf-8")` path).
//! - `length` output fits int32 by construction (points <= bytes <=
//!   i32::MAX); no overflow code exists on this symbol.
//! - `contains` with a non-UTF8 needle -> BAD_RANGE (explicit).
//!
//! Columnar: one forward pass over rows, row-independent (chunkable).

use crate::core::checks::offset_in_bounds;
use crate::core::errors::{BAD_RANGE, MALFORMED, OK};

/// Validated code-point count. `None` on invalid UTF-8.
///
/// Wasm-safe: indexed `while` over bytes, no iterators, no
/// `str` methods. Accepts exactly the UTF-8 range
/// (`from_utf8`-equivalent: rejects overlongs, surrogates,
/// 0xF5..0xFF leads, truncated tails).
#[inline(always)]
pub(crate) fn utf8_count_validated(row: &[u8]) -> Option<i32> {
    let mut c: i32 = 0;
    let mut j: usize = 0;
    let len = row.len();
    while j < len {
        let b0 = row[j];
        if b0 < 0x80 {
            c += 1;
            j += 1;
        } else if b0 >> 5 == 0b110 {
            if j + 1 >= len {
                return None;
            }
            let b1 = row[j + 1];
            if b1 & 0xC0 != 0x80 {
                return None;
            }
            // Overlong: 0xC0/0xC1 never valid; 0xC2+ always 2-byte min.
            if b0 < 0xC2 {
                return None;
            }
            c += 1;
            j += 2;
        } else if b0 >> 4 == 0b1110 {
            if j + 2 >= len {
                return None;
            }
            let b1 = row[j + 1];
            let b2 = row[j + 2];
            if b1 & 0xC0 != 0x80 || b2 & 0xC0 != 0x80 {
                return None;
            }
            let cp = ((b0 as u32 & 0x0F) << 12)
                | ((b1 as u32 & 0x3F) << 6)
                | (b2 as u32 & 0x3F);
            // Overlong (< 0x800) or surrogate (D800-DFFF).
            if cp < 0x800 || (0xD800..0xE000).contains(&cp) {
                return None;
            }
            c += 1;
            j += 3;
        } else if b0 >> 3 == 0b11110 {
            if j + 3 >= len {
                return None;
            }
            let b1 = row[j + 1];
            let b2 = row[j + 2];
            let b3 = row[j + 3];
            if b1 & 0xC0 != 0x80 || b2 & 0xC0 != 0x80 || b3 & 0xC0 != 0x80 {
                return None;
            }
            let cp = ((b0 as u32 & 0x07) << 18)
                | ((b1 as u32 & 0x3F) << 12)
                | ((b2 as u32 & 0x3F) << 6)
                | (b3 as u32 & 0x3F);
            // Overlong (< 0x10000) or out of range (> 0x10FFFF).
            if cp < 0x10000 || cp > 0x10FFFF {
                return None;
            }
            c += 1;
            j += 4;
        } else {
            // Continuation byte or 0xF5..0xFF lead: never valid.
            return None;
        }
    }
    Some(c)
}

/// Row lengths in code points. Returns 0 ok, MALFORMED on bad
/// offsets / invalid UTF-8. Rows before the bad row are written.
#[inline(never)]
pub fn text_length_scan(data: &[u8], offs: &[i32], out: &mut [i32]) -> i32 {
    let n = out.len();
    debug_assert!(offs.len() >= n + 1);
    let total = data.len();
    if n == 0 {
        return OK;
    }
    let mut prev = offs[0];
    if !offset_in_bounds(prev, total) {
        return MALFORMED;
    }
    for i in 0..n {
        let end = offs[i + 1];
        if end < prev || !offset_in_bounds(end, total) {
            return MALFORMED;
        }
        let row = &data[prev as usize..end as usize];
        prev = end;
        match utf8_count_validated(row) {
            Some(count) => out[i] = count,
            None => return MALFORMED,
        }
    }
    OK
}

/// Substring hit per row (u8 0/1). Empty needle matches all rows.
/// Returns 0 ok, BAD_RANGE on non-UTF8 needle, MALFORMED on bad
/// offsets / invalid UTF-8. Rows before the bad row are written.
#[inline]
pub fn text_contains_scan(
    data: &[u8],
    offs: &[i32],
    needle: &[u8],
    out: &mut [u8],
) -> i32 {
    let n = out.len();
    debug_assert!(offs.len() >= n + 1);
    if utf8_count_validated(needle).is_none() {
        return BAD_RANGE;
    }
    let total = data.len();
    if n == 0 {
        return OK;
    }
    let mut prev = offs[0];
    if !offset_in_bounds(prev, total) {
        return MALFORMED;
    }
    // First-byte prefilter: skips memcmp on rows that cannot match.
    let first = if needle.is_empty() { 0 } else { needle[0] };
    for i in 0..n {
        out[i] = 0;
        let end = offs[i + 1];
        if end < prev || !offset_in_bounds(end, total) {
            return MALFORMED;
        }
        let row = &data[prev as usize..end as usize];
        prev = end;
        if utf8_count_validated(row).is_none() {
            return MALFORMED;
        }
        if needle.is_empty() {
            out[i] = 1;
        } else if row.len() >= needle.len()
            && memmem_first(row, needle, first)
        {
            out[i] = 1;
        }
    }
    OK
}

/// Naive substring with a first-byte skip. `row.len() >= needle.len()`
/// and non-empty needle are caller guarantees. Hot: single forward
/// pass, `==` on slices lowers to memcmp.
#[inline(always)]
fn memmem_first(row: &[u8], needle: &[u8], first: u8) -> bool {
    let last_start = row.len() - needle.len();
    let mut i = 0;
    while i <= last_start {
        if row[i] == first && &row[i..i + needle.len()] == needle {
            return true;
        }
        i += 1;
    }
    false
}

/// Anchored affix hit per row (u8 0/1). Empty needle matches all rows.
/// `at_start`: true = startswith, false = endswith. Returns 0 ok,
/// BAD_RANGE on non-UTF8 needle, MALFORMED on bad offsets / invalid
/// UTF-8. Rows before the bad row are written. Same offset/validity
/// wire as `text_contains_scan` (row-independent, chunkable).
#[inline]
pub fn text_affix_scan(
    data: &[u8],
    offs: &[i32],
    needle: &[u8],
    at_start: bool,
    out: &mut [u8],
) -> i32 {
    let n = out.len();
    debug_assert!(offs.len() >= n + 1);
    if utf8_count_validated(needle).is_none() {
        return BAD_RANGE;
    }
    let total = data.len();
    if n == 0 {
        return OK;
    }
    let mut prev = offs[0];
    if !offset_in_bounds(prev, total) {
        return MALFORMED;
    }
    for i in 0..n {
        out[i] = 0;
        let end = offs[i + 1];
        if end < prev || !offset_in_bounds(end, total) {
            return MALFORMED;
        }
        let row = &data[prev as usize..end as usize];
        prev = end;
        if utf8_count_validated(row).is_none() {
            return MALFORMED;
        }
        if needle.is_empty() {
            out[i] = 1;
        } else if row.len() >= needle.len() {
            let hit = if at_start {
                &row[..needle.len()] == needle
            } else {
                &row[row.len() - needle.len()..] == needle
            };
            if hit {
                out[i] = 1;
            }
        }
    }
    OK
}

/// Byte-exact row equality per row (u8 0/1). Valid UTF-8 bytes == valid
/// UTF-8 code points, so `==` on slices is `str ==` parity. Returns 0
/// ok, BAD_RANGE on non-UTF8 needle, MALFORMED on bad offsets /
/// invalid UTF-8. Same wire as `text_contains_scan`.
#[inline]
pub fn text_equals_scan(data: &[u8], offs: &[i32], needle: &[u8], out: &mut [u8]) -> i32 {
    let n = out.len();
    debug_assert!(offs.len() >= n + 1);
    if utf8_count_validated(needle).is_none() {
        return BAD_RANGE;
    }
    let total = data.len();
    if n == 0 {
        return OK;
    }
    let mut prev = offs[0];
    if !offset_in_bounds(prev, total) {
        return MALFORMED;
    }
    for i in 0..n {
        out[i] = 0;
        let end = offs[i + 1];
        if end < prev || !offset_in_bounds(end, total) {
            return MALFORMED;
        }
        let row = &data[prev as usize..end as usize];
        prev = end;
        if utf8_count_validated(row).is_none() {
            return MALFORMED;
        }
        if row == needle {
            out[i] = 1;
        }
    }
    OK
}
