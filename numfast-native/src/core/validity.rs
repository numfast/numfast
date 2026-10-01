// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Output-state + validity helpers: zeroing prologues, per-row slot
//! reset, absent-lane predicate.
//!
//! Zeroing is a warm per-call prologue (mem-bound, once per call);
//! `clear_slot` / `is_present` sit in hot loops and stay 1–2 stores.

/// Overwrite the whole lane range with zero (`0.0` / `0` — `Default`
/// is the zero value for every lane type used here). Warm prologue.
#[inline]
pub fn fill_zero<T: Copy + Default>(lanes: &mut [T]) {
    for s in lanes.iter_mut() {
        *s = T::default();
    }
}

/// Per-row reset: invalid data, never an error (pattern contract).
/// Hot — compiles to two stores.
#[inline(always)]
pub fn clear_slot(codes: &mut [i32], valid: &mut [u8], i: usize) {
    codes[i] = 0;
    valid[i] = 0;
}

/// Dense count lane present (`0 = absent group`). Hot.
#[inline(always)]
pub fn is_present(count: i64) -> bool {
    count != 0
}
