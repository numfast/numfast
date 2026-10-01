// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Precondition checks: null pointers, dense-code ranges, offsets.
//!
//! Null checks are cold (error path); the dense bounds check is hot
//! (inside the fused multi-aggregate row loop) and must stay a single
//! predictable branch — hence `#[inline(always)]` everywhere here.

/// True when the pointer is usable (non-null). Cold.
#[inline(always)]
pub fn non_null<T>(p: *const T) -> bool {
    !p.is_null()
}

/// Dense code `k` (already widened) inside `[0, g)`. Hot.
/// `dense_scatter` deliberately does NOT use this: its contract leaves
/// out-of-range keys to the safe-Rust trap (same as before).
#[inline(always)]
pub fn key_in_range(k: usize, g: usize) -> bool {
    k < g
}

/// Row-boundary offset inside `[0, total]`. Warm/hot (pattern walk).
#[inline(always)]
pub fn offset_in_bounds(off: i32, total: usize) -> bool {
    off >= 0 && (off as usize) <= total
}
