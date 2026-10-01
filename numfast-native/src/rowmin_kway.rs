// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Canonical: generic K-way rowwise MIN over int32 selector
//! lanes + float32 payload gather (N->N, one pass).
//!
//! Data semantics per row `i` over lanes `k = 0..K` (K runtime, `1..=256`):
//! `m[i] = argmin_k(t[k][i])` (strict `<`, ties keep smallest index),
//! `t_best[i] = t[m[i]][i]` (int32 exact copy),
//! `d_best[i] = d[m[i]][i]` (float32 bit-exact copy, never arithmetic).
//!
//! Ordering rule: strict `<` chain from lane 0 over the int32 T lanes.
//! Consequences: ties keep the smallest index; INT32_MIN/MAX ride as
//! ordinary ordered values. The D lanes NEVER select the mode -- they
//! are pure payload (NaN/Inf/-0.0 ride bit-exact through the gather).
//! Validity sidecars travel host-side; this kernel never sees validity.
//! Empty is handled by the caller (`n == 0` returns before deref).
//! Row-independent: chunkable=true (any `[lo, hi)` slice of the outputs
//! is a pure function of the same input slices).
//!
//! Same no-alloc `core`-only discipline as the other kernels.

/// Generic K-way rowwise MIN over int32 selector lanes + f32 gather.
/// `t_lanes.len() == d_lanes.len() == k` with `1 <= k <= 256`.
/// Caller guarantees every lane has `len() >= n`, outputs `len() >= n`.
#[inline]
pub fn rowwise_kway_time_argmin_gather(
    t_lanes: &[&[i32]],
    d_lanes: &[&[f32]],
    n: usize,
    t_best: &mut [i32],
    d_best: &mut [f32],
    m_best: &mut [u8],
) {
    let k = t_lanes.len();
    debug_assert!(k >= 1 && k <= 256);
    debug_assert_eq!(d_lanes.len(), k);
    debug_assert!(t_best.len() >= n);
    debug_assert!(d_best.len() >= n);
    debug_assert!(m_best.len() >= n);
    for i in 0..n {
        let mut m: usize = 0;
        let mut best: i32 = t_lanes[0][i];
        for lane in 1..k {
            let v = t_lanes[lane][i];
            if v < best {
                m = lane;
                best = v;
            }
        }
        m_best[i] = m as u8;
        t_best[i] = best;
        // Gather: verbatim copy of the winning lane's D payload
        // (float32 bit-exact, NaN/Inf/-0.0 included).
        d_best[i] = d_lanes[m][i];
    }
}
