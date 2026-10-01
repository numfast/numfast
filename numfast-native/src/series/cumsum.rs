// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `cumsum_kernel` mechanic: inclusive prefix sum N->N (IR `cumsum`).
//!
//! Data-only scan semantics: `out[i] = sum(src[0..=i])` in the lane
//! dtype. NaN/Inf ride IEEE-forward (plain accumulation, never a
//! validity signal). Validity sidecar travels host-side (0-fill of
//! invalid rows + per-row carry, same rule as CPU `_cumsum_ref`):
//! this kernel never sees validity. Empty is handled here (no caller
//! branch needed, same as the other kernels).
//!
//! Integer policy (frozen IR contract, same as the GPU scan lanes):
//! int32 wraps mod 2**32 (`wrapping_add`, never saturate/trap).
//! Floats accumulate in their own dtype (f32 stays f32, f64 stays f64),
//! sequential `i = 0..n` order — the CPU `_cumsum_ref` oracle order,
//! so conformance is the existing numerical contract from
//! `specs-rebuilt/conformance-profile.toml` vs an f64 oracle.
//!
//! Same no-alloc `core`-only discipline as the other kernels.

/// Inclusive prefix sum over int32 lanes (wrapping mod 2**32).
/// Caller guarantees `out.len() >= src.len()`.
#[inline]
pub fn cumsum_scatter_i32(src: &[i32], out: &mut [i32]) {
    debug_assert!(out.len() >= src.len());
    let mut acc: i32 = 0;
    for i in 0..src.len() {
        acc = acc.wrapping_add(src[i]);
        out[i] = acc;
    }
}

/// Inclusive prefix sum over float32 lanes (own-dtype accumulation).
/// Caller guarantees `out.len() >= src.len()`.
#[inline]
pub fn cumsum_scatter_f32(src: &[f32], out: &mut [f32]) {
    debug_assert!(out.len() >= src.len());
    let mut acc: f32 = 0.0;
    for i in 0..src.len() {
        acc += src[i];
        out[i] = acc;
    }
}

/// Inclusive prefix sum over float64 lanes (own-dtype accumulation).
/// Caller guarantees `out.len() >= src.len()`.
#[inline]
pub fn cumsum_scatter_f64(src: &[f64], out: &mut [f64]) {
    debug_assert!(out.len() >= src.len());
    let mut acc: f64 = 0.0;
    for i in 0..src.len() {
        acc += src[i];
        out[i] = acc;
    }
}
