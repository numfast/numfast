// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `adjacency` mechanic: P2 CSR slice + flat gather (generic CSR,
//! no graph-router vocabulary: no SSSP/TBRS/router concepts here).
//!
//! Frozen semantics (parity with
//! `Relational/Segmented/_lib/adjacency.py`): `indptr[V+1]` (u32,
//! nondecreasing, inside `[0, E]`, `indptr[V] == E`) + `indices[E]`
//! (u32) + `query[k]` (u32) -> `begins[k]` / `ends[k]` (u32) with
//! `begins[g] = indptr[q]`, `ends[g] = indptr[q+1]`; the flat lane is
//! the concatenation of `indices[begins[g]..ends[g]]` in query order
//! (storage order inside each slice, deterministic). `UINT32_MAX` is
//! reserved (INF/INVALID) in every input lane. `V/E/k == 0` is safe:
//! the empty-graph encoding is `indptr.len() == 0` (or `== 1` with
//! `E == 0`); a non-empty query on an empty graph is an error, while
//! `k == 0` only enforces the reserved-lane rule (same as the Python
//! early return, which runs after the id coercion).
//!
//! Same no-alloc `core`-only discipline as the other kernels: the
//! caller owns all buffers; on success the `k` lanes are overwritten,
//! on validation errors outputs hold at most lanes before the abort
//! point (same rule as the sorted-run speculative abort).

use crate::core::errors::{BAD_RANGE, MALFORMED, OK};

/// Reserved lane value (INF/INVALID): rejected in every input lane.
pub const RESERVED: u32 = u32::MAX;

#[inline]
fn has_reserved(lane: &[u32]) -> bool {
    lane.contains(&RESERVED)
}

/// CSR slice: `begins[g]/ends[g]` for `query[g]`.
/// Caller guarantees `begins.len() == ends.len() == query.len()`.
/// Errors: `BAD_RANGE` on any reserved lane, non-monotone indptr,
/// out-of-range query, or a non-empty query on an empty graph;
/// `MALFORMED` on `indptr[V] != E`, a lane outside `[0, E]`, or a
/// non-empty index block with empty indptr. With `query` empty only
/// the reserved-lane rule is enforced (outputs untouched).
#[inline]
pub fn adjacency_slice(
    indptr: &[u32],
    indices: &[u32],
    query: &[u32],
    begins: &mut [u32],
    ends: &mut [u32],
) -> i32 {
    debug_assert_eq!(begins.len(), query.len());
    debug_assert_eq!(ends.len(), query.len());
    if has_reserved(indptr) || has_reserved(indices) || has_reserved(query) {
        return BAD_RANGE;
    }
    let e = indices.len();
    if indptr.is_empty() {
        if e != 0 {
            return MALFORMED;
        }
        if !query.is_empty() {
            return BAD_RANGE;
        }
        return OK;
    }
    let v = indptr.len() - 1;
    for i in 0..indptr.len() {
        if (indptr[i] as usize) > e {
            return MALFORMED;
        }
        if i > 0 && indptr[i] < indptr[i - 1] {
            return BAD_RANGE;
        }
    }
    if (indptr[v] as usize) != e {
        return MALFORMED;
    }
    if query.is_empty() {
        return OK;
    }
    for g in 0..query.len() {
        if (query[g] as usize) >= v {
            return BAD_RANGE;
        }
    }
    for g in 0..query.len() {
        let q = query[g] as usize;
        begins[g] = indptr[q];
        ends[g] = indptr[q + 1];
    }
    OK
}

/// Flat gather: concatenate `indices[begins[g]..ends[g]]` in query
/// order into `out` (storage order inside each slice). Caller
/// guarantees `begins.len() == ends.len() == k` and
/// `out.len() >= sum(ends - begins)`; the pair table is validated
/// first (outputs untouched on error). Errors: `BAD_RANGE` on
/// `begins[g] > ends[g]` or `ends[g] > E`; `MALFORMED` when `out`
/// is short. Index lanes are carried verbatim (the slice gate owns
/// the reserved-lane rule); `k == 0` returns `OK`.
#[inline]
pub fn adjacency_gather(
    indices: &[u32],
    begins: &[u32],
    ends: &[u32],
    out: &mut [u32],
) -> i32 {
    debug_assert_eq!(begins.len(), ends.len());
    let e = indices.len();
    let mut total: usize = 0;
    for g in 0..begins.len() {
        if begins[g] > ends[g] {
            return BAD_RANGE;
        }
        if (ends[g] as usize) > e {
            return BAD_RANGE;
        }
        total += (ends[g] - begins[g]) as usize;
    }
    if out.len() < total {
        return MALFORMED;
    }
    let mut p: usize = 0;
    for g in 0..begins.len() {
        let s = begins[g] as usize;
        let t = ends[g] as usize;
        for i in s..t {
            out[p] = indices[i];
            p += 1;
        }
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> (Vec<u32>, Vec<u32>) {
        // V=4: v0 -> [10, 11], v1 -> [], v2 -> [12], v3 -> [13, 14, 15].
        (vec![0, 2, 2, 3, 6], vec![10, 11, 12, 13, 14, 15])
    }

    #[test]
    fn empty_graph_safe() {
        let ie: Vec<u32> = vec![];
        let ix: Vec<u32> = vec![];
        let q: Vec<u32> = vec![];
        let mut b: Vec<u32> = vec![];
        let mut e: Vec<u32> = vec![];
        // V/E/k = 0: ok, nothing written.
        assert_eq!(adjacency_slice(&ie, &ix, &q, &mut b, &mut e), OK);
        // Single-slot indptr with E = 0 is the same empty graph.
        let ie1: Vec<u32> = vec![0];
        assert_eq!(adjacency_slice(&ie1, &ix, &q, &mut b, &mut e), OK);
        // Non-empty query on an empty graph: error.
        let q1: Vec<u32> = vec![0];
        let mut b1: Vec<u32> = vec![7];
        let mut e1: Vec<u32> = vec![7];
        assert_eq!(adjacency_slice(&ie, &ix, &q1, &mut b1, &mut e1), BAD_RANGE);
        assert_eq!(adjacency_slice(&ie1, &ix, &q1, &mut b1, &mut e1), BAD_RANGE);
        // Non-empty indices with empty indptr: malformed.
        let ix1: Vec<u32> = vec![3];
        assert_eq!(adjacency_slice(&ie, &ix1, &q, &mut b, &mut e), MALFORMED);
        // Gather over zero slices: ok.
        let mut o: Vec<u32> = vec![];
        assert_eq!(adjacency_gather(&ix, &b, &e, &mut o), OK);
    }

    #[test]
    fn reserved_rejected_every_lane() {
        let (mut ip, mut ix) = graph();
        let q: Vec<u32> = vec![0];
        let mut b: Vec<u32> = vec![0];
        let mut e: Vec<u32> = vec![0];
        ip[1] = RESERVED;
        assert_eq!(adjacency_slice(&ip, &ix, &q, &mut b, &mut e), BAD_RANGE);
        let (ip2, mut ix2) = graph();
        ix2[0] = RESERVED;
        assert_eq!(adjacency_slice(&ip2, &ix2, &q, &mut b, &mut e), BAD_RANGE);
        let (ip3, ix3) = graph();
        let q3: Vec<u32> = vec![RESERVED];
        assert_eq!(adjacency_slice(&ip3, &ix3, &q3, &mut b, &mut e), BAD_RANGE);
        // Reserved surfaces even with k == 0 (Python coercion parity).
        let q0: Vec<u32> = vec![];
        let mut b0: Vec<u32> = vec![];
        let mut e0: Vec<u32> = vec![];
        assert_eq!(adjacency_slice(&ip, &ix, &q0, &mut b0, &mut e0), BAD_RANGE);
    }

    #[test]
    fn monotone_terminal_and_range() {
        let (ip, ix) = graph();
        let mut b: Vec<u32> = vec![0];
        let mut e: Vec<u32> = vec![0];
        // Query vertex out of range.
        assert_eq!(
            adjacency_slice(&ip, &ix, &[4], &mut b, &mut e),
            BAD_RANGE
        );
        // Regression in indptr.
        let mut bad = ip.clone();
        bad[3] = 1;
        assert_eq!(adjacency_slice(&bad, &ix, &[0], &mut b, &mut e), BAD_RANGE);
        // Terminal lane != E.
        let mut short = ip.clone();
        short[4] = 5;
        assert_eq!(
            adjacency_slice(&short, &ix, &[0], &mut b, &mut e),
            MALFORMED
        );
        // Lane outside [0, E].
        let mut wide = ip.clone();
        wide[2] = 99;
        assert_eq!(
            adjacency_slice(&wide, &ix, &[0], &mut b, &mut e),
            MALFORMED
        );
    }

    #[test]
    fn slice_then_gather_storage_order() {
        let (ip, ix) = graph();
        // Query order [3, 0, 2, 1]: begins/ends follow the query,
        // gather preserves storage order inside each slice.
        let q: Vec<u32> = vec![3, 0, 2, 1];
        let mut b: Vec<u32> = vec![9; 4];
        let mut e: Vec<u32> = vec![9; 4];
        assert_eq!(adjacency_slice(&ip, &ix, &q, &mut b, &mut e), OK);
        assert_eq!(b, vec![3, 0, 2, 2]);
        assert_eq!(e, vec![6, 2, 3, 2]);
        let mut flat: Vec<u32> = vec![0; 6];
        assert_eq!(adjacency_gather(&ix, &b, &e, &mut flat), OK);
        assert_eq!(flat, vec![13, 14, 15, 10, 11, 12]);
        // Repeat: deterministic.
        let mut b2: Vec<u32> = vec![0; 4];
        let mut e2: Vec<u32> = vec![0; 4];
        let mut f2: Vec<u32> = vec![0; 6];
        assert_eq!(adjacency_slice(&ip, &ix, &q, &mut b2, &mut e2), OK);
        assert_eq!(adjacency_gather(&ix, &b2, &e2, &mut f2), OK);
        assert_eq!((b, e, flat), (b2, e2, f2));
    }

    #[test]
    fn gather_guards() {
        let (_, ix) = graph();
        // begins > ends.
        let mut o: Vec<u32> = vec![0; 4];
        assert_eq!(
            adjacency_gather(&ix, &[3], &[2], &mut o),
            BAD_RANGE
        );
        // ends > E.
        assert_eq!(
            adjacency_gather(&ix, &[0], &[7], &mut o),
            BAD_RANGE
        );
        // Short output.
        let mut short: Vec<u32> = vec![0; 1];
        assert_eq!(
            adjacency_gather(&ix, &[0], &[2], &mut short),
            MALFORMED
        );
        // Empty slice pair contributes nothing.
        let mut two: Vec<u32> = vec![0; 2];
        assert_eq!(adjacency_gather(&ix, &[2, 0], &[2, 2], &mut two), OK);
        assert_eq!(two, vec![10, 11]);
    }
}
