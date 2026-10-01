// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `router` mechanic: multi-source Dijkstra (SSSP) over a flat CSR graph
//! with early exit on a target set. Port of the query hot path in
//! `roadgraph/ClusterRouter/_lib/router.py::route_nodes`
//! (portal graph, V=235734, E=922245, avg_deg 3.9).
//!
//! Frozen semantics (parity with `route_nodes`):
//! - all sources at dist 0, min-heap of `(dist, vertex)` (tie-break by
//!   vertex id, same as CPython `heapq` on `(d, u)` tuples);
//! - stale-pop skip (`d != dist[u]`);
//! - first popped vertex in target set wins, its dist is returned;
//! - edge relax `nd = d + w`, update iff `nd < dist[v]`;
//! - unreachable -> no target popped.
//! INF edges (`0xFFFFFFFF`) are excluded at flatten time in Python
//! (same `continue` as the Python build); negative weights are skipped.
//! No heuristics, no GPU/WGSL, no whole-RoadGraph port: this one kernel
//! only. Python owns orchestration/data/ABI.

use core::cmp::Reverse;
use core::i64;
use std::collections::BinaryHeap;

use crate::core::buffers::{borrow, borrow_mut};
use crate::core::checks::non_null;
use crate::core::errors::{BAD_RANGE, NULL_OR_ABORT, OK};

/// Unreachable marker written to `out_dist` (also FFI return `1`).
pub const ROUTER_UNREACHABLE: i32 = 1;

/// Multi-source Dijkstra core. Caller owns all buffers.
/// `indptr` len `n+1` (nondecreasing, `indptr[n] == m`), `indices`/`weights`
/// len `m`, `sources` len `nsrc`, `is_target` len `n` (0/1).
/// Returns `(dist, target_or_-1, popped)`.
pub fn router_route_csr(
    indptr: &[i64],
    indices: &[i32],
    weights: &[i64],
    sources: &[i32],
    is_target: &[u8],
) -> (i64, i32, i64) {
    let n = is_target.len();
    const INF: i64 = i64::MAX / 4;
    let mut dist = vec![INF; n];
    let mut pq: BinaryHeap<Reverse<(i64, i32)>> =
        BinaryHeap::with_capacity(sources.len().max(1) * 2);
    for &s in sources {
        if s < 0 || (s as usize) >= n {
            continue;
        }
        let v = s as usize;
        if dist[v] > 0 {
            dist[v] = 0;
            pq.push(Reverse((0, s)));
        }
    }
    let mut popped: i64 = 0;
    while let Some(Reverse((d, u))) = pq.pop() {
        let uu = u as usize;
        if d != dist[uu] {
            continue;
        }
        popped += 1;
        if is_target[uu] != 0 {
            return (d, u, popped);
        }
        let beg = indptr[uu] as usize;
        let end = indptr[uu + 1] as usize;
        for e in beg..end {
            let w = weights[e];
            if w < 0 {
                continue;
            }
            let v = indices[e] as usize;
            let nd = d.saturating_add(w);
            if nd < dist[v] {
                dist[v] = nd;
                pq.push(Reverse((nd, indices[e])));
            }
        }
    }
    (i64::MAX, -1, popped)
}

/// FFI: multi-source Dijkstra over flat CSR, early exit on target set.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `n` vertices; `indptr[0..n+1]` int64 nondecreasing, `indptr[n] == m`;
/// - `indices[0..m]` int32 in `[0, n)`; `weights[0..m]` int64 ms (`<0` skip);
/// - `sources[0..nsrc]` int32 (out-of-range entries ignored);
/// - `is_target[0..n]` u8 0/1;
/// - `out_dist[0..1]` int64, `out_target[0..1]` int32, `out_visited[0..1]` int64.
/// - returns 0 found, 1 unreachable (`out_dist = INT64_MAX`, `out_target = -1`),
///   -1 null pointer, -2 bad range (monotone/indptr/empty-shape violation).
#[no_mangle]
pub unsafe extern "C" fn nf_router_route(
    n: usize,
    indptr: *const i64,
    m: usize,
    indices: *const i32,
    weights: *const i64,
    sources: *const i32,
    nsrc: usize,
    is_target: *const u8,
    out_dist: *mut i64,
    out_target: *mut i32,
    out_visited: *mut i64,
) -> i32 {
    if n == 0 || m == 0 || nsrc == 0 {
        return BAD_RANGE;
    }
    if !non_null(indptr)
        || !non_null(indices)
        || !non_null(weights)
        || !non_null(sources)
        || !non_null(is_target)
        || !non_null(out_dist)
        || !non_null(out_target)
        || !non_null(out_visited)
    {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let indptr = unsafe { borrow(indptr, n + 1) };
    let indices = unsafe { borrow(indices, m) };
    let weights = unsafe { borrow(weights, m) };
    let sources = unsafe { borrow(sources, nsrc) };
    let is_target = unsafe { borrow(is_target, n) };
    let out_dist = unsafe { borrow_mut(out_dist, 1) };
    let out_target = unsafe { borrow_mut(out_target, 1) };
    let out_visited = unsafe { borrow_mut(out_visited, 1) };
    if indptr[n] as usize != m {
        return BAD_RANGE;
    }
    let mut prev: i64 = 0;
    for i in 0..=n {
        let v = indptr[i];
        if v < 0 || v as usize > m || v < prev {
            return BAD_RANGE;
        }
        prev = v;
    }
    for e in 0..m {
        let v = indices[e];
        if v < 0 || (v as usize) >= n {
            return BAD_RANGE;
        }
    }
    let (d, t, popped) = router_route_csr(indptr, indices, weights, sources, is_target);
    out_dist[0] = d;
    out_target[0] = t;
    out_visited[0] = popped;
    if t < 0 {
        ROUTER_UNREACHABLE
    } else {
        OK
    }
}
