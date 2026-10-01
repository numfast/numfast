// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `sssp` mechanic: canonical single-source shortest path over flat CSR
//! with u32 lanes (generic graph/weights/sources, no domain vocabulary).
//!
//! Frozen semantics (parity with the Python heapq reference in
//! `src/Relational/Sssp/_lib/sssp.py::_fb_sssp`):
//! - `dist[source] = 0`, min-heap of `(dist, vertex)` (tie-break by
//!   vertex id, same as `heapq` on `(d, u)` tuples);
//! - stale-pop skip (`d != dist[u]`);
//! - full run to heap exhaustion (no target set, no early exit);
//! - edge relax `nd = d.saturating_add(w)`, update iff
//!   `nd < dist[v]` (saturating overflow lands on INF, never wins);
//! - `INF (UINT32_MAX)` reserved: rejected in `indptr`/`indices`/
//!   `sources`, skipped in `weights` (edge excluded, never relaxed);
//! - unreachable vertices keep `INF` in `dist`.
//!
//! Same no-alloc `core`-only discipline as the other kernels: the
//! caller owns all buffers; `V == 0` is an error (no empty-graph
//! encoding on this surface: `indptr` must hold `V + 1` lanes).

use core::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::core::errors::{BAD_RANGE, MALFORMED, OK};

/// Reserved lane value (INF/UNREACHABLE): rejected in index lanes,
/// skipped in weight lanes, written for unreachable vertices.
pub const SSSP_INF: u32 = u32::MAX;

/// Max fan-out for the batch MT pass (same budget as the join probe:
/// clamped, never unbounded).
const SSSP_THREADS_MAX: usize = 64;

/// Validate CSR geometry. `indptr.len() == v + 1`, nondecreasing,
/// inside `[0, e]`, `indptr[v] == e`; `indices` in `[0, v)`, no
/// reserved lanes. Returns `OK` / `BAD_RANGE` / `MALFORMED`.
#[inline]
fn check_csr(indptr: &[u32], indices: &[u32]) -> i32 {
    let e = indices.len();
    if indptr.is_empty() {
        return if e == 0 { BAD_RANGE } else { MALFORMED };
    }
    let v = indptr.len() - 1;
    if v == 0 {
        return BAD_RANGE;
    }
    for w in indices.iter() {
        if *w == SSSP_INF || (*w as usize) >= v {
            return BAD_RANGE;
        }
    }
    let mut prev: u32 = 0;
    for (i, &x) in indptr.iter().enumerate() {
        if x == SSSP_INF || (x as usize) > e {
            return MALFORMED;
        }
        if i > 0 && x < prev {
            return BAD_RANGE;
        }
        prev = x;
    }
    if (indptr[v] as usize) != e {
        return MALFORMED;
    }
    OK
}

/// Dijkstra core over validated CSR. `dist.len() == v`, overwritten
/// (`INF` init, `dist[source] = 0`). `pred` optionally overwritten
/// (`-1` init, `pred[source] = -1`, `pred[v] = u` on relax).
#[inline]
fn dijkstra(
    indptr: &[u32],
    indices: &[u32],
    weights: &[u32],
    source: usize,
    dist: &mut [u32],
    pred: Option<&mut [u32]>,
) {
    let v = indptr.len() - 1;
    debug_assert_eq!(dist.len(), v);
    dist.fill(SSSP_INF);
    dist[source] = 0;
    // Caller passes i32 lanes via u32 view; -1 sentinel = u32::MAX.
    let mut pred_u: Option<&mut [u32]> = pred;
    if let Some(p) = pred_u.as_mut() {
        p.fill(SSSP_INF);
        p[source] = SSSP_INF;
    }
    let mut pq: BinaryHeap<Reverse<(u32, u32)>> =
        BinaryHeap::with_capacity(v.min(1024).max(1));
    pq.push(Reverse((0u32, source as u32)));
    while let Some(Reverse((d, u))) = pq.pop() {
        let uu = u as usize;
        if d != dist[uu] {
            continue;
        }
        let beg = indptr[uu] as usize;
        let end = indptr[uu + 1] as usize;
        for e in beg..end {
            let w = weights[e];
            if w == SSSP_INF {
                continue;
            }
            let vv = indices[e] as usize;
            let nd = d.saturating_add(w);
            if nd < dist[vv] {
                dist[vv] = nd;
                if let Some(p) = pred_u.as_mut() {
                    p[vv] = u;
                }
                pq.push(Reverse((nd, indices[e])));
            }
        }
    }
}

/// Single source: `indptr[V+1]` + `indices[E]` + `weights[E]` +
/// `source` -> `dist[V]`. Caller guarantees lane lengths.
/// Errors: `BAD_RANGE` (reserved/out-of-range index, non-monotone
/// indptr, bad source), `MALFORMED` (lane outside `[0, E]`,
/// `indptr[V] != E`). `pred` follows the same call with `pred[V]`
/// i32 lanes (`-1` root/unreached).
#[inline]
pub fn sssp_csr(
    indptr: &[u32],
    indices: &[u32],
    weights: &[u32],
    source: u32,
    dist: &mut [u32],
    pred: Option<&mut [u32]>,
) -> i32 {
    debug_assert_eq!(weights.len(), indices.len());
    let rc = check_csr(indptr, indices);
    if rc != OK {
        return rc;
    }
    let v = indptr.len() - 1;
    if dist.len() < v {
        return MALFORMED;
    }
    if source == SSSP_INF || (source as usize) >= v {
        return BAD_RANGE;
    }
    if let Some(p) = pred.as_ref() {
        if p.len() < v {
            return MALFORMED;
        }
    }
    // Reborrow exact lanes (caller may over-allocate).
    let dist = &mut dist[..v];
    match pred {
        Some(p) => {
            // i32 lanes viewed as u32 (bit-identical, `-1` == INF).
            let pu: &mut [u32] = unsafe { std::slice::from_raw_parts_mut(p.as_mut_ptr() as *mut u32, v) };
            dijkstra(indptr, indices, weights, source as usize, dist, Some(pu));
        }
        None => dijkstra(indptr, indices, weights, source as usize, dist, None),
    }
    OK
}

/// One batch shard: rows `sources` -> `out` lanes `[r*V..(r+1)*V]`.
/// Flat loop, no allocation, read-only graph (thread-shareable).
#[inline]
fn sssp_batch_shard(
    indptr: &[u32],
    indices: &[u32],
    weights: &[u32],
    v: usize,
    sources: &[u32],
    out: &mut [u32],
) {
    debug_assert!(out.len() >= sources.len() * v);
    for (r, &s) in sources.iter().enumerate() {
        let row = &mut out[r * v..(r + 1) * v];
        dijkstra(indptr, indices, weights, s as usize, row, None);
    }
}

/// Batch: `sources[K]` -> `out[K*V]` row-major (`out[r*V..(r+1)*V]`
/// holds `dist` from `sources[r]`). Each row independent: output
/// order = source order (thread-count invariant). `nthreads`
/// clamped to `[1, 64]` (0 -> 1). Validated first (outputs
/// untouched on error).
#[inline]
pub fn sssp_batch(
    indptr: &[u32],
    indices: &[u32],
    weights: &[u32],
    sources: &[u32],
    k: usize,
    out: &mut [u32],
    nthreads: usize,
) -> i32 {
    debug_assert_eq!(sources.len(), k);
    let rc = check_csr(indptr, indices);
    if rc != OK {
        return rc;
    }
    let v = indptr.len() - 1;
    if k == 0 {
        return OK;
    }
    if out.len() < k.saturating_mul(v) {
        return MALFORMED;
    }
    for &s in &sources[..k] {
        if s == SSSP_INF || (s as usize) >= v {
            return BAD_RANGE;
        }
    }
    let t = nthreads.clamp(1, SSSP_THREADS_MAX).min(k);
    let cs = k.div_ceil(t);
    let src_chunks: Vec<&[u32]> = sources[..k].chunks(cs).collect();
    let mut out_rest: &mut [u32] = &mut out[..k * v];
    let mut out_parts: Vec<&mut [u32]> = Vec::with_capacity(src_chunks.len());
    for sc in &src_chunks {
        let (a, ra) = out_rest.split_at_mut(sc.len() * v);
        out_parts.push(a);
        out_rest = ra;
    }
    // wasm32-unknown-unknown: no threads — same shards, inline, same order.
    #[cfg(target_arch = "wasm32")]
    {
        for (sc, oc) in src_chunks.into_iter().zip(out_parts) {
            sssp_batch_shard(indptr, indices, weights, v, sc, oc);
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::scope(|scope| {
        for (sc, oc) in src_chunks.into_iter().zip(out_parts) {
            scope.spawn(move || {
                sssp_batch_shard(indptr, indices, weights, v, sc, oc);
            });
        }
    });
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::errors::{BAD_RANGE, MALFORMED, OK};

    fn graph() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        // V=4: 0->[1(2),2(5)], 1->[2(1),3(4)], 2->[3(1)], 3->[].
        (vec![0, 2, 4, 5, 5], vec![1, 2, 2, 3, 3], vec![2, 5, 1, 4, 1])
    }

    #[test]
    fn single_source_dist() {
        let (ip, ix, w) = graph();
        let mut d = vec![0u32; 4];
        assert_eq!(sssp_csr(&ip, &ix, &w, 0, &mut d, None), OK);
        assert_eq!(d, vec![0, 2, 3, 4]);
        let mut d1 = vec![0u32; 4];
        assert_eq!(sssp_csr(&ip, &ix, &w, 1, &mut d1, None), OK);
        assert_eq!(d1, vec![SSSP_INF, 0, 1, 2]);
        let mut d3 = vec![0u32; 4];
        assert_eq!(sssp_csr(&ip, &ix, &w, 3, &mut d3, None), OK);
        assert_eq!(d3, vec![SSSP_INF, SSSP_INF, SSSP_INF, 0]);
    }

    #[test]
    fn tie_break_vertex_id() {
        // 0->[1(1),2(1)]: both dist 1, heap pops 1 first (smaller id).
        let ip = vec![0, 2, 2, 2];
        let ix = vec![1, 2];
        let w = vec![1, 1];
        let mut d = vec![0u32; 3];
        let mut p = vec![0i32; 3];
        assert_eq!(sssp_csr(&ip, &ix, &w, 0, &mut d, Some(unsafe {
            std::slice::from_raw_parts_mut(p.as_mut_ptr() as *mut u32, 3)
        })), OK);
        assert_eq!(d, vec![0, 1, 1]);
        assert_eq!(p[1], 0);
        assert_eq!(p[2], 0);
    }

    #[test]
    fn inf_weight_skipped() {
        let (ip, ix, w) = (vec![0, 1, 1], vec![1], vec![SSSP_INF]);
        let mut d = vec![0u32; 2];
        assert_eq!(sssp_csr(&ip, &ix, &w, 0, &mut d, None), OK);
        assert_eq!(d, vec![0, SSSP_INF]);
    }

    #[test]
    fn guards() {
        let (ip, ix, w) = graph();
        let mut d = vec![0u32; 4];
        assert_eq!(sssp_csr(&ip, &ix, &w, 9, &mut d, None), BAD_RANGE);
        assert_eq!(sssp_csr(&ip, &ix, &w, SSSP_INF, &mut d, None), BAD_RANGE);
        let mut bad = ip.clone();
        bad[2] = 1;
        assert_eq!(sssp_csr(&bad, &ix, &w, 0, &mut d, None), BAD_RANGE);
        let mut short = ip.clone();
        short[4] = 4;
        assert_eq!(sssp_csr(&short, &ix, &w, 0, &mut d, None), MALFORMED);
        let mut small = vec![0u32; 2];
        assert_eq!(sssp_csr(&ip, &ix, &w, 0, &mut small, None), MALFORMED);
    }

    #[test]
    fn batch_rows_equal_single() {
        let (ip, ix, w) = graph();
        let src = vec![0u32, 1, 3];
        let mut out = vec![0u32; 3 * 4];
        assert_eq!(sssp_batch(&ip, &ix, &w, &src, 3, &mut out, 4), OK);
        let mut d0 = vec![0u32; 4];
        let mut d1 = vec![0u32; 4];
        let mut d3 = vec![0u32; 4];
        assert_eq!(sssp_csr(&ip, &ix, &w, 0, &mut d0, None), OK);
        assert_eq!(sssp_csr(&ip, &ix, &w, 1, &mut d1, None), OK);
        assert_eq!(sssp_csr(&ip, &ix, &w, 3, &mut d3, None), OK);
        assert_eq!(&out[0..4], &d0[..]);
        assert_eq!(&out[4..8], &d1[..]);
        assert_eq!(&out[8..12], &d3[..]);
        // Thread-count invariant: single vs MT identical.
        let mut out1 = vec![0u32; 3 * 4];
        assert_eq!(sssp_batch(&ip, &ix, &w, &src, 3, &mut out1, 1), OK);
        assert_eq!(out, out1);
    }
}
