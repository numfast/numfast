// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `join_build` + `join_probe` mechanics (M1: native JOIN probe path).
//!
//! Open-addressing hash table over int32 keys, caller-owned buffers:
//! - build: right keys (H2O contract: unique) -> table lanes
//!   (`table_keys`, `table_pos`, `table_occ`), capacity `cap` is a
//!   power of two with `cap >= 2*s` (load factor <= 0.5). Duplicate
//!   right key = explicit error, never a silent collapse.
//! - probe: left keys -> `pos_out` (int64 match position, 0 on miss)
//!   + `hit_out` (u8 0/1). Row-independent: thread-count invariant,
//!   output order = input order (same as the searchsorted probe).
//!
//! Hashing reuses the reserved [`crate::core::hash`] probe atoms
//! (`probe_step` Fibonacci mix over key bits, `next_probe` linear
//! step). Hot loops stay flat and specialized (no generic mega-kernel).
//!
//! M3 (fused output): `join_fused_left_i32` (single-pass probe ->
//! direct full-size output, row-independent) and
//! `join_fused_inner_i32` (two-pass count + re-probe fill, no
//! pos/pos[hit]/fancy-indexing temps, order-preserving). Reuses the
//! same table layout and [`probe_step`]/[`next_probe`] atoms.

use crate::core::hash::{next_probe, probe_step};

/// Join-local codes (symbol-local meanings, same convention as the
/// frozen table: `-2` = precondition violation, `-3` = malformed
/// geometry). Frozen `errors.rs` values untouched.
pub const JOIN_DUP: i32 = -2;
pub const JOIN_GEOM: i32 = -3;

/// Max fan-out for the MT probe (mirrors NUMFAST_THREADS=16 bench
/// protocol; clamped, never unbounded).
const PROBE_THREADS_MAX: usize = 64;

/// `cap` valid iff power of two and `cap >= 2*s` (`s > 0`).
#[inline]
fn valid_cap(cap: usize, s: usize) -> bool {
    cap.is_power_of_two() && cap >= s.saturating_mul(2)
}

/// Build the right hash table. `right.len() == s`; table lanes
/// `table_keys/table_pos/table_occ` each `len >= cap`. `occ[..cap]`
/// zeroed first (WARM). Returns `OK`, `JOIN_DUP` (first duplicate
/// left prior lanes written — caller must rebuild), `JOIN_GEOM`.
#[inline]
pub fn join_build(
    right: &[i32],
    s: usize,
    table_keys: &mut [i32],
    table_pos: &mut [i32],
    table_occ: &mut [u8],
    cap: usize,
) -> i32 {
    debug_assert_eq!(right.len(), s);
    debug_assert!(table_keys.len() >= cap);
    debug_assert!(table_pos.len() >= cap);
    debug_assert!(table_occ.len() >= cap);
    if !valid_cap(cap, s) {
        return JOIN_GEOM;
    }
    let mask = cap - 1;
    table_occ[..cap].fill(0);
    for pos in 0..s {
        let k = right[pos];
        let mut slot = probe_step((k as u32) as u64, mask);
        loop {
            if table_occ[slot] == 0 {
                table_keys[slot] = k;
                table_pos[slot] = pos as i32;
                table_occ[slot] = 1;
                break;
            }
            if table_keys[slot] == k {
                return JOIN_DUP;
            }
            slot = next_probe(slot, mask);
        }
    }
    crate::core::errors::OK
}

/// One probe lookup: `Some(right_pos)` on hit, `None` on miss.
/// Shared atom for the M1 probe shard and the M3 fused kernels.
#[inline]
fn join_probe_one(
    table_keys: &[i32],
    table_pos: &[i32],
    table_occ: &[u8],
    mask: usize,
    k: i32,
) -> Option<i32> {
    let mut slot = probe_step((k as u32) as u64, mask);
    loop {
        if table_occ[slot] == 0 {
            return None;
        }
        if table_keys[slot] == k {
            return Some(table_pos[slot]);
        }
        slot = next_probe(slot, mask);
    }
}

/// One probe shard: rows `left` -> `pos_out`/`hit_out` (same lens).
/// Flat loop, no allocation, read-only table (thread-shareable).
#[inline]
fn join_probe_shard(
    table_keys: &[i32],
    table_pos: &[i32],
    table_occ: &[u8],
    mask: usize,
    left: &[i32],
    pos_out: &mut [i64],
    hit_out: &mut [u8],
) {
    debug_assert_eq!(pos_out.len(), left.len());
    debug_assert_eq!(hit_out.len(), left.len());
    for i in 0..left.len() {
        match join_probe_one(table_keys, table_pos, table_occ, mask, left[i]) {
            Some(p) => {
                pos_out[i] = p as i64;
                hit_out[i] = 1;
            }
            None => {
                pos_out[i] = 0;
                hit_out[i] = 0;
            }
        }
    }
}

/// One gather shard: rows `pos` -> `out[i] = payload[pos[i]]`.
/// Flat loop, no allocation, read-only payload (thread-shareable).
/// First out-of-range position aborts the shard with `JOIN_GEOM`
/// (lanes before it written — same partial-write convention as
/// `JOIN_DUP` on build); caller data (probe output) is always in
/// range, so the branch stays cold.
#[inline]
fn join_gather_shard(payload: &[i32], s: usize, pos: &[i64], out: &mut [i32]) -> i32 {
    debug_assert!(out.len() >= pos.len());
    debug_assert!(payload.len() >= s);
    for i in 0..pos.len() {
        let p = pos[i] as usize;
        if p >= s {
            return JOIN_GEOM;
        }
        out[i] = payload[p];
    }
    crate::core::errors::OK
}

/// MT positional gather over `m` positions, `nthreads` clamped to
/// `[1, 64]` (0 -> 1). Payload `len >= s`, `pos.len() == m`, `out`
/// `len >= m`. Read-only payload shared across scoped threads;
/// output shards are disjoint, order = input order. Returns `OK` /
/// `JOIN_GEOM` (first OOB position; `s == 0` with `m > 0` is GEOM).
#[inline]
pub fn join_gather_i32(
    payload: &[i32],
    s: usize,
    pos: &[i64],
    m: usize,
    out: &mut [i32],
    nthreads: usize,
) -> i32 {
    debug_assert_eq!(pos.len(), m);
    debug_assert!(out.len() >= m);
    debug_assert!(payload.len() >= s);
    if m == 0 {
        return crate::core::errors::OK;
    }
    if s == 0 {
        return JOIN_GEOM;
    }
    let t = nthreads.clamp(1, PROBE_THREADS_MAX).min(m);
    let cs = m.div_ceil(t);
    // Disjoint reborrows, one per scoped thread (row-independent).
    let pos_chunks: Vec<&[i64]> = pos[..m].chunks(cs).collect();
    let out_chunks: Vec<&mut [i32]> = out[..m].chunks_mut(cs).collect();
    let mut rc = crate::core::errors::OK;
    std::thread::scope(|scope| {
        let handles: Vec<_> = pos_chunks
            .into_iter()
            .zip(out_chunks)
            .map(|(pc, oc)| scope.spawn(move || join_gather_shard(payload, s, pc, oc)))
            .collect();
        for h in handles {
            if h.join().unwrap() != crate::core::errors::OK {
                rc = JOIN_GEOM;
            }
        }
    });
    rc
}

/// MT probe over `n` rows, `nthreads` clamped to `[1, 64]` (0 -> 1).
/// Table lanes `len >= cap`, outputs `len >= n`. Read-only table is
/// shared across scoped threads; output shards are disjoint.
/// Returns `OK` / `JOIN_GEOM` (cap not power of two; `s` unknown here
/// so only the mask rule is checked — build validated the load).
#[inline]
pub fn join_probe(
    table_keys: &[i32],
    table_pos: &[i32],
    table_occ: &[u8],
    cap: usize,
    left: &[i32],
    n: usize,
    pos_out: &mut [i64],
    hit_out: &mut [u8],
    nthreads: usize,
) -> i32 {
    debug_assert_eq!(left.len(), n);
    debug_assert!(pos_out.len() >= n);
    debug_assert!(hit_out.len() >= n);
    debug_assert!(table_keys.len() >= cap);
    debug_assert!(table_pos.len() >= cap);
    debug_assert!(table_occ.len() >= cap);
    if !cap.is_power_of_two() {
        return JOIN_GEOM;
    }
    if n == 0 {
        return crate::core::errors::OK;
    }
    let mask = cap - 1;
    let t = nthreads.clamp(1, PROBE_THREADS_MAX).min(n);
    let cs = n.div_ceil(t);
    // Disjoint reborrows, one per scoped thread (row-independent).
    let key_chunks: Vec<&[i32]> = left[..n].chunks(cs).collect();
    let pos_chunks: Vec<&mut [i64]> = pos_out[..n].chunks_mut(cs).collect();
    let hit_chunks: Vec<&mut [u8]> = hit_out[..n].chunks_mut(cs).collect();
    std::thread::scope(|scope| {
        for ((kc, pc), hc) in key_chunks.into_iter().zip(pos_chunks).zip(hit_chunks) {
            scope.spawn(move || {
                join_probe_shard(table_keys, table_pos, table_occ, mask, kc, pc, hc);
            });
        }
    });
    crate::core::errors::OK
}

/// One fused-LEFT shard: probe row `i` -> direct full-size output
/// (`out_keys/out_v1` passthrough, `out_v2` payload or 0,
/// `valid_out` 1/0 NULL sidecar). Flat loop, no allocation,
/// read-only table+payload (thread-shareable), disjoint writes.
#[inline]
fn join_fused_left_shard(
    table_keys: &[i32],
    table_pos: &[i32],
    table_occ: &[u8],
    mask: usize,
    left_keys: &[i32],
    left_v1: &[i32],
    payload: &[i32],
    s: usize,
    out_keys: &mut [i32],
    out_v1: &mut [i32],
    out_v2: &mut [i32],
    valid_out: &mut [u8],
) {
    debug_assert_eq!(out_keys.len(), left_keys.len());
    debug_assert_eq!(out_v1.len(), left_keys.len());
    debug_assert_eq!(out_v2.len(), left_keys.len());
    debug_assert_eq!(valid_out.len(), left_keys.len());
    debug_assert!(payload.len() >= s);
    for i in 0..left_keys.len() {
        out_keys[i] = left_keys[i];
        out_v1[i] = left_v1[i];
        match join_probe_one(table_keys, table_pos, table_occ, mask, left_keys[i]) {
            Some(p) => {
                out_v2[i] = payload[p as usize];
                valid_out[i] = 1;
            }
            None => {
                out_v2[i] = 0;
                valid_out[i] = 0;
            }
        }
    }
    let _ = s;
}

/// M3 fused LEFT over `n` rows, `nthreads` clamped to `[1, 64]`
/// (0 -> 1). Table lanes `len >= cap`, payload `len >= s`, inputs
/// `len >= n`, outputs `len >= n`. Single pass: hash probe ->
/// hit/miss -> immediately the full-size compact output
/// (keys/v1/v2 + valid NULL sidecar, miss -> v2 = 0, valid = 0).
/// No pos/pos[hit]/fancy-indexing temps, no separate materialize.
/// Row-independent: thread-count invariant, output order = input
/// order. Returns `OK` / `JOIN_GEOM` (cap not pow2; `s == 0` with
/// `n > 0`).
#[inline]
pub fn join_fused_left_i32(
    table_keys: &[i32],
    table_pos: &[i32],
    table_occ: &[u8],
    cap: usize,
    left_keys: &[i32],
    left_v1: &[i32],
    payload: &[i32],
    s: usize,
    n: usize,
    out_keys: &mut [i32],
    out_v1: &mut [i32],
    out_v2: &mut [i32],
    valid_out: &mut [u8],
    nthreads: usize,
) -> i32 {
    debug_assert_eq!(left_keys.len(), n);
    debug_assert_eq!(left_v1.len(), n);
    debug_assert!(payload.len() >= s);
    debug_assert!(out_keys.len() >= n);
    debug_assert!(out_v1.len() >= n);
    debug_assert!(out_v2.len() >= n);
    debug_assert!(valid_out.len() >= n);
    debug_assert!(table_keys.len() >= cap);
    debug_assert!(table_pos.len() >= cap);
    debug_assert!(table_occ.len() >= cap);
    if !cap.is_power_of_two() {
        return JOIN_GEOM;
    }
    if n == 0 {
        return crate::core::errors::OK;
    }
    if s == 0 {
        return JOIN_GEOM;
    }
    let mask = cap - 1;
    let t = nthreads.clamp(1, PROBE_THREADS_MAX).min(n);
    let cs = n.div_ceil(t);
    // Disjoint reborrows, one per scoped thread (row-independent).
    let key_chunks: Vec<&[i32]> = left_keys[..n].chunks(cs).collect();
    let v1_chunks: Vec<&[i32]> = left_v1[..n].chunks(cs).collect();
    let ok_chunks: Vec<&mut [i32]> = out_keys[..n].chunks_mut(cs).collect();
    let o1_chunks: Vec<&mut [i32]> = out_v1[..n].chunks_mut(cs).collect();
    let o2_chunks: Vec<&mut [i32]> = out_v2[..n].chunks_mut(cs).collect();
    let vv_chunks: Vec<&mut [u8]> = valid_out[..n].chunks_mut(cs).collect();
    std::thread::scope(|scope| {
        for (((((kc, vc), okc), o1c), o2c), vvc) in key_chunks
            .into_iter()
            .zip(v1_chunks)
            .zip(ok_chunks)
            .zip(o1_chunks)
            .zip(o2_chunks)
            .zip(vv_chunks)
        {
            scope.spawn(move || {
                join_fused_left_shard(
                    table_keys, table_pos, table_occ, mask, kc, vc, payload, s, okc, o1c,
                    o2c, vvc,
                );
            });
        }
    });
    crate::core::errors::OK
}

/// Count hits in one chunk (M3 INNER pass 1): probe only, no writes.
#[inline]
fn join_count_shard(
    table_keys: &[i32],
    table_pos: &[i32],
    table_occ: &[u8],
    mask: usize,
    left: &[i32],
) -> usize {
    let mut c = 0usize;
    for &k in left {
        if join_probe_one(table_keys, table_pos, table_occ, mask, k).is_some() {
            c += 1;
        }
    }
    c
}

/// Fill one compact chunk (M3 INNER pass 2): re-probe rows, write
/// hits at `out_*` lanes `[0..hits]` (caller prefix offsets),
/// probe order preserved within the chunk.
#[inline]
fn join_fill_shard(
    table_keys: &[i32],
    table_pos: &[i32],
    table_occ: &[u8],
    mask: usize,
    left_keys: &[i32],
    left_v1: &[i32],
    payload: &[i32],
    out_keys: &mut [i32],
    out_v1: &mut [i32],
    out_v2: &mut [i32],
) {
    debug_assert!(out_keys.len() <= left_keys.len());
    debug_assert!(out_v1.len() <= left_keys.len());
    debug_assert!(out_v2.len() <= left_keys.len());
    let mut w = 0usize;
    for i in 0..left_keys.len() {
        if let Some(p) = join_probe_one(table_keys, table_pos, table_occ, mask, left_keys[i]) {
            out_keys[w] = left_keys[i];
            out_v1[w] = left_v1[i];
            out_v2[w] = payload[p as usize];
            w += 1;
        }
    }
    debug_assert_eq!(w, out_keys.len());
}

/// M3 fused INNER over `n` rows, `nthreads` clamped to `[1, 64]`
/// (0 -> 1). Table lanes `len >= cap`, payload `len >= s`, inputs
/// `len >= n`, outputs `len >= n` (only first `m` lanes written).
/// Two passes with a re-probe (no pos/pos[hit]/fancy-indexing
/// temps, no separate materialize): pass 1 counts hits per chunk,
/// prefix offsets, pass 2 re-probes and writes compact
/// (keys/v1/v2) directly at prefix offsets. Output order = input
/// (probe) order, thread-count invariant. Returns `m` (hits,
/// `>= 0`), `JOIN_GEOM` (-3: cap not pow2; `s == 0` with `n > 0`).
#[inline]
pub fn join_fused_inner_i32(
    table_keys: &[i32],
    table_pos: &[i32],
    table_occ: &[u8],
    cap: usize,
    left_keys: &[i32],
    left_v1: &[i32],
    payload: &[i32],
    s: usize,
    n: usize,
    out_keys: &mut [i32],
    out_v1: &mut [i32],
    out_v2: &mut [i32],
    nthreads: usize,
) -> i64 {
    debug_assert_eq!(left_keys.len(), n);
    debug_assert_eq!(left_v1.len(), n);
    debug_assert!(payload.len() >= s);
    debug_assert!(out_keys.len() >= n);
    debug_assert!(out_v1.len() >= n);
    debug_assert!(out_v2.len() >= n);
    debug_assert!(table_keys.len() >= cap);
    debug_assert!(table_pos.len() >= cap);
    debug_assert!(table_occ.len() >= cap);
    if !cap.is_power_of_two() {
        return JOIN_GEOM as i64;
    }
    if n == 0 {
        return 0;
    }
    if s == 0 {
        return JOIN_GEOM as i64;
    }
    let mask = cap - 1;
    let t = nthreads.clamp(1, PROBE_THREADS_MAX).min(n);
    let cs = n.div_ceil(t);
    // Pass 1: hit counts per chunk (probe only, no output writes).
    let counts: Vec<usize> = std::thread::scope(|scope| {
        let handles: Vec<_> = left_keys[..n]
            .chunks(cs)
            .map(|kc| {
                scope.spawn(move || join_count_shard(table_keys, table_pos, table_occ, mask, kc))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let nc = counts.len();
    let mut off = Vec::with_capacity(nc + 1);
    off.push(0usize);
    for c in &counts {
        off.push(off.last().copied().unwrap() + *c);
    }
    let m = off[nc];
    debug_assert!(m <= n);
    if m == 0 {
        return 0;
    }
    // Pass 2: carve disjoint compact ranges [off[i]..off[i+1]] and
    // re-probe each input chunk directly into its range.
    let key_chunks: Vec<&[i32]> = left_keys[..n].chunks(cs).collect();
    let v1_chunks: Vec<&[i32]> = left_v1[..n].chunks(cs).collect();
    let mut ok_rest: &mut [i32] = &mut out_keys[..m];
    let mut o1_rest: &mut [i32] = &mut out_v1[..m];
    let mut o2_rest: &mut [i32] = &mut out_v2[..m];
    let mut ok_parts: Vec<&mut [i32]> = Vec::with_capacity(nc);
    let mut o1_parts: Vec<&mut [i32]> = Vec::with_capacity(nc);
    let mut o2_parts: Vec<&mut [i32]> = Vec::with_capacity(nc);
    for c in &counts {
        let (a, ra) = ok_rest.split_at_mut(*c);
        let (b, rb) = o1_rest.split_at_mut(*c);
        let (d, rd) = o2_rest.split_at_mut(*c);
        ok_parts.push(a);
        o1_parts.push(b);
        o2_parts.push(d);
        ok_rest = ra;
        o1_rest = rb;
        o2_rest = rd;
    }
    std::thread::scope(|scope| {
        for ((((kc, vc), okc), o1c), o2c) in key_chunks
            .into_iter()
            .zip(v1_chunks)
            .zip(ok_parts)
            .zip(o1_parts)
            .zip(o2_parts)
        {
            scope.spawn(move || {
                join_fill_shard(
                    table_keys, table_pos, table_occ, mask, kc, vc, payload, okc, o1c, o2c,
                );
            });
        }
    });
    m as i64
}
