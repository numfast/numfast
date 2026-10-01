// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `rowmin` mechanic: generic rowwise MIN over 4 int32 value lanes keyed
//! by 4 float32 key lanes, with argmin + gather (N->N, one pass).
//!
//! Data semantics per row `i` over lanes `k = 0..4`:
//! `m[i] = argmin_k(d[k][i])` (ties keep the smallest index),
//! `t_best[i] = t[m[i]][i]` (int32 exact copy),
//! `d_best[i] = d[m[i]][i]` (float32 bit-exact copy, never arithmetic).
//!
//! Ordering rule: strict `<` chain from lane 0 (`if d[k] < best` takes
//! the lane). Consequences, all deterministic: ties keep the smallest
//! index; `-0.0` vs `0.0` is a tie (index wins); `-Inf` beats anything
//! finite; `+Inf` loses to anything finite; a NaN lane never wins by
//! `<` (comparisons with NaN are false), so NaN wins only when every
//! lane ahead of it also failed to beat lane 0 -- i.e. all-NaN rows
//! (and NaN-leading rows) resolve to the smallest such index, exactly
//! the `np.argmin` strict-less scan order the parity harness checks.
//! NaN/Inf payloads ride bit-exact through the gather (plain copy).
//!
//! Validity sidecars travel host-side; this kernel never sees validity.
//! Empty is handled here (no caller branch needed, same as the other
//! kernels). Row-independent: chunkable=true (any `[lo, hi)` slice of
//! the outputs is a pure function of the same input slices).
//!
//! Same no-alloc `core`-only discipline as the other kernels.

/// Rowwise MIN4 + argmin + gather over 4 int32 lanes keyed by 4 f32 lanes.
/// Caller guarantees every slice has `len() >= n`.
#[inline]
pub fn rowwise_min4_argmin_gather(
    t0: &[i32],
    t1: &[i32],
    t2: &[i32],
    t3: &[i32],
    d0: &[f32],
    d1: &[f32],
    d2: &[f32],
    d3: &[f32],
    n: usize,
    t_best: &mut [i32],
    d_best: &mut [f32],
    m_best: &mut [u8],
) {
    debug_assert!(t0.len() >= n);
    debug_assert!(t1.len() >= n);
    debug_assert!(t2.len() >= n);
    debug_assert!(t3.len() >= n);
    debug_assert!(d0.len() >= n);
    debug_assert!(d1.len() >= n);
    debug_assert!(d2.len() >= n);
    debug_assert!(d3.len() >= n);
    debug_assert!(t_best.len() >= n);
    debug_assert!(d_best.len() >= n);
    debug_assert!(m_best.len() >= n);
    for i in 0..n {
        let mut m: usize = 0;
        let mut best: f32 = d0[i];
        if d1[i] < best {
            m = 1;
            best = d1[i];
        }
        if d2[i] < best {
            m = 2;
            best = d2[i];
        }
        if d3[i] < best {
            m = 3;
            best = d3[i];
        }
        m_best[i] = m as u8;
        // Gather: verbatim copies of the winning lane (int32 exact,
        // float32 bit-exact, NaN payload included).
        t_best[i] = match m {
            0 => t0[i],
            1 => t1[i],
            2 => t2[i],
            _ => t3[i],
        };
        d_best[i] = best;
    }
}

/// Rowwise MIN4 over int32 time lanes + float32 payload gather (generic).
/// Kept specialized for compat; canonical generic kernel lives in
/// `rowmin_kway` (`K = 4` agrees bit-exactly, see K-way parity).
/// Caller guarantees every slice has `len() >= n`.
///
/// Data semantics per row `i` over lanes `k = 0..4`:
/// `m[i] = argmin_k(t[k][i])` (strict `<`, ties keep smallest index),
/// `t_best[i] = t[m[i]][i]` (int32 exact copy),
/// `d_best[i] = d[m[i]][i]` (float32 bit-exact copy, never arithmetic).
///
/// Ordering rule: strict `<` chain from lane 0 over the int32 T lanes.
/// Consequences: ties keep the smallest index; INT32_MIN/MAX ride as
/// ordinary ordered values. The D lanes NEVER select the mode -- they
/// are pure payload (NaN/Inf/-0.0 ride bit-exact through the gather).
/// Validity sidecars travel host-side; this kernel never sees validity.
/// Empty is handled by the caller (`n == 0` returns before deref).
/// Row-independent: chunkable=true (any `[lo, hi)` slice of the outputs
/// is a pure function of the same input slices).
#[inline]
pub fn rowwise_min4_time_argmin_gather(
    t0: &[i32],
    t1: &[i32],
    t2: &[i32],
    t3: &[i32],
    d0: &[f32],
    d1: &[f32],
    d2: &[f32],
    d3: &[f32],
    n: usize,
    t_best: &mut [i32],
    d_best: &mut [f32],
    m_best: &mut [u8],
) {
    debug_assert!(t0.len() >= n);
    debug_assert!(t1.len() >= n);
    debug_assert!(t2.len() >= n);
    debug_assert!(t3.len() >= n);
    debug_assert!(d0.len() >= n);
    debug_assert!(d1.len() >= n);
    debug_assert!(d2.len() >= n);
    debug_assert!(d3.len() >= n);
    debug_assert!(t_best.len() >= n);
    debug_assert!(d_best.len() >= n);
    debug_assert!(m_best.len() >= n);
    for i in 0..n {
        let mut m: usize = 0;
        let mut best: i32 = t0[i];
        if t1[i] < best {
            m = 1;
            best = t1[i];
        }
        if t2[i] < best {
            m = 2;
            best = t2[i];
        }
        if t3[i] < best {
            m = 3;
            best = t3[i];
        }
        m_best[i] = m as u8;
        t_best[i] = best;
        // Gather: verbatim copy of the winning lane's D payload
        // (float32 bit-exact, NaN/Inf/-0.0 included).
        d_best[i] = match m {
            0 => d0[i],
            1 => d1[i],
            2 => d2[i],
            _ => d3[i],
        };
    }
}

/// Rowwise MIN4 over int32 time lanes, time-only (no D payload).
/// Same `m`/`t_best` semantics as [`rowwise_min4_time_argmin_gather`];
/// caller guarantees every slice has `len() >= n`.
#[inline]
pub fn rowwise_min4_time_argmin(
    t0: &[i32],
    t1: &[i32],
    t2: &[i32],
    t3: &[i32],
    n: usize,
    t_best: &mut [i32],
    m_best: &mut [u8],
) {
    debug_assert!(t0.len() >= n);
    debug_assert!(t1.len() >= n);
    debug_assert!(t2.len() >= n);
    debug_assert!(t3.len() >= n);
    debug_assert!(t_best.len() >= n);
    debug_assert!(m_best.len() >= n);
    for i in 0..n {
        let mut m: usize = 0;
        let mut best: i32 = t0[i];
        if t1[i] < best {
            m = 1;
            best = t1[i];
        }
        if t2[i] < best {
            m = 2;
            best = t2[i];
        }
        if t3[i] < best {
            m = 3;
            best = t3[i];
        }
        m_best[i] = m as u8;
        t_best[i] = best;
    }
}
