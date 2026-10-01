// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `shift_kernel` mechanic: positional right-shift N->N (IR `shift`).
//!
//! Data-only memcpy semantics: `out[i] = 0` for `i < periods`, else
//! `out[i] = src[i - periods]`. NaN/Inf ride bit-exact (plain copy,
//! never arithmetic). Validity sidecar travels host-side (same carry
//! rule as CPU `_shift_ref` and GPU `shift_take`): this kernel never
//! sees validity. Empty/identity/all-zero shapes are handled here
//! (no caller branch needed, same as the other kernels).
//!
//! Same no-alloc `core`-only discipline as the other kernels.

/// Right-shift copy. Caller guarantees `out.len() >= src.len()`.
/// Single generic lane param `S` (monomorphizes to the same flat loop
/// as hand-written per-dtype copies). `S: Default` is the zero value
/// (`0` / `0.0`) for every lane type used here.
#[inline]
pub fn shift_scatter<S: Copy + Default>(src: &[S], periods: usize, out: &mut [S]) {
    debug_assert!(out.len() >= src.len());
    let n = src.len();
    if n == 0 {
        return;
    }
    if periods == 0 {
        out[..n].copy_from_slice(&src[..n]);
        return;
    }
    if periods >= n {
        for s in out[..n].iter_mut() {
            *s = S::default();
        }
        return;
    }
    for s in out[..periods].iter_mut() {
        *s = S::default();
    }
    out[periods..n].copy_from_slice(&src[..n - periods]);
}
