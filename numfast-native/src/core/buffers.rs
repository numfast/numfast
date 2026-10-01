// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Raw pointer + length ABI: single unsafe wrap point for all FFI.
//!
//! Caller contract (same as before): pointers are non-null, aligned,
//! and valid for `len` contiguous lanes of `T`; outputs are
//! caller-owned and overwritten. Null is rejected by the caller via
//! [`crate::core::checks::non_null`] before borrowing — never here.

use core::slice;

/// Borrow `len` lanes. Cold per-call setup.
/// SAFETY: upheld by the caller contract above.
/// Zero-length borrows never touch the pointer (null + 0 is legal input
/// from empty-family FFI calls): return an empty slice via a dangling
/// pointer instead of `from_raw_parts(null, 0)` (formally UB).
#[inline]
pub unsafe fn borrow<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 {
        return &[];
    }
    // SAFETY: caller guarantees validity (see module docs).
    unsafe { slice::from_raw_parts(ptr, len) }
}

/// Borrow `len` lanes mutably. Cold per-call setup.
/// SAFETY: upheld by the caller contract above.
/// Same zero-length dangling rule as [`borrow`].
#[inline]
pub unsafe fn borrow_mut<'a, T>(ptr: *mut T, len: usize) -> &'a mut [T] {
    if len == 0 {
        return &mut [];
    }
    // SAFETY: caller guarantees validity (see module docs).
    unsafe { slice::from_raw_parts_mut(ptr, len) }
}
