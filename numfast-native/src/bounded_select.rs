// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Generic native bounded top-K selection over paired int32 keys.
//!
//! Generic surface: no domain vocabulary (no slack/rank/request/route
//! words here — inputs are `a` (primary) and `b` (secondary)).
//! Integer-only, deterministic, no alloc, no query branches.
//!
//! Contract per group `g` over CSR `offs[0..G+1]` into `a/b[0..N]`:
//! output lanes `out_a/out_b/out_idx[G*K]` hold the per-group top-K
//! pairs sorted ascending by `(a, b, idx)` with strict `<` on `(a, b)`
//! and the smaller original index winning full ties. `K` is runtime
//! (`>= 1`, caller allocates `G*K` lanes). Short groups pad the tail
//! with `(INF, INF, -1)` where `INF = i32::MAX`; infeasible rows are
//! exactly `a == INF` lanes (no mask input — they sink to the tail by
//! order). All-`INF` group => all lanes `INF` (caller reads it as empty).
//! Empty groups => all lanes `INF`. `G == 1` is the flat case.
//!
//! Properties: full-sort parity (top-K == prefix of full sort by the
//! same comparator), prefix invariant (top-(K-1) == prefix of top-K),
//! input-order invariance (comparator carries `idx`, so the selected
//! set never depends on scan order). Row-groups independent:
//! chunkable=true over disjoint `[g0, g1)` group ranges.
//!
//! Same buffers in/out discipline as the other kernels: caller-owned
//! lanes, no alloc, no collections, `std` only at the FFI wrap.

/// Sentinel for infeasible / padding lanes.
pub const BOUND_INF: i32 = i32::MAX;

/// Core kernel: per-group bounded top-K over `(a, b)` with `idx` tiebreak.
/// Caller guarantees: `a/b` len >= `n`, `offs` len >= `g+1`, outputs each
/// len >= `g*k` with `k >= 1`. Returns 0 ok, -2 bad `k`, -3 bad offsets.
#[inline]
pub fn bounded_select_2i32(
    a: &[i32],
    b: &[i32],
    offs: &[i32],
    g: usize,
    k: usize,
    out_a: &mut [i32],
    out_b: &mut [i32],
    out_idx: &mut [i32],
) -> i32 {
    if k == 0 {
        return crate::core::errors::BAD_RANGE;
    }
    if offs.len() < g + 1 {
        return crate::core::errors::MALFORMED;
    }
    let n = a.len().min(b.len());
    // Validate CSR: nondecreasing, inside [0, n], offs[g] == n, offs[0] == 0.
    // Empty (g == 0) requires n == 0.
    if g == 0 {
        if n != 0 {
            return crate::core::errors::MALFORMED;
        }
        return crate::core::errors::OK;
    }
    if offs[0] != 0 {
        return crate::core::errors::MALFORMED;
    }
    let mut prev: i64 = 0;
    for gi in 0..=g {
        let v = offs[gi] as i64;
        if v < prev || v > n as i64 {
            return crate::core::errors::MALFORMED;
        }
        prev = v;
    }
    if offs[g] as usize != n {
        return crate::core::errors::MALFORMED;
    }
    debug_assert!(out_a.len() >= g * k);
    debug_assert!(out_b.len() >= g * k);
    debug_assert!(out_idx.len() >= g * k);
    for gi in 0..g {
        let lo = offs[gi] as usize;
        let hi = offs[gi + 1] as usize;
        let base = gi * k;
        let (oa, ob, oi) = (&mut out_a[base..base + k], &mut out_b[base..base + k], &mut out_idx[base..base + k]);
        let mut len: usize = 0;
        for idx in lo..hi {
            let ka = a[idx];
            let kb = b[idx];
            let ii = idx as i32;
            // Linear locate: first p with (ka,kb,ii) < (oa[p],ob[p],oi[p]).
            // Linear (not binary): K <= 32 hot path, no N/query branches.
            let mut p: usize = 0;
            while p < len {
                let pa = oa[p];
                let pb = ob[p];
                let pi = oi[p];
                if ka < pa || (ka == pa && (kb < pb || (kb == pb && ii < pi))) {
                    break;
                }
                p += 1;
            }
            if len < k {
                // Shift [p..len) right by one, insert at p.
                let mut q = len;
                while q > p {
                    oa[q] = oa[q - 1];
                    ob[q] = ob[q - 1];
                    oi[q] = oi[q - 1];
                    q -= 1;
                }
                oa[p] = ka;
                ob[p] = kb;
                oi[p] = ii;
                len += 1;
            } else if p < k {
                // Full: insert at p, drop worst at k-1.
                let mut q = k - 1;
                while q > p {
                    oa[q] = oa[q - 1];
                    ob[q] = ob[q - 1];
                    oi[q] = oi[q - 1];
                    q -= 1;
                }
                oa[p] = ka;
                ob[p] = kb;
                oi[p] = ii;
            }
        }
        // Pad tail with (INF, INF, -1).
        let mut q = len;
        while q < k {
            oa[q] = BOUND_INF;
            ob[q] = BOUND_INF;
            oi[q] = -1;
            q += 1;
        }
    }
    crate::core::errors::OK
}
