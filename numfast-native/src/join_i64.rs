// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `join_build_i64` + `join_probe_i64` mechanics (int64-key lane).
//!
//! Additive copy of the frozen int32 `join.rs` lane for 64-bit keys
//! (LDBC ids exceed int32). Existing int32 symbols untouched.
//!
//! Open-addressing hash table over int64 keys, caller-owned buffers:
//! - build: right keys (unique contract) -> table lanes
//!   (`table_keys` i64, `table_pos` i32, `table_occ` u8), capacity
//!   `cap` power of two with `cap >= 2*s` (load <= 0.5). Duplicate
//!   right key = explicit error, never a silent collapse.
//! - probe: left keys -> `pos_out` (int64 right-row position, 0 on
//!   miss) + `hit_out` (u8 0/1). Row-independent: thread-count
//!   invariant, output order = input order (same as searchsorted).
//!
//! Hashing reuses the reserved [`crate::core::hash`] probe atoms.
//! int64 keys hash by bit-preserving `k as u64` (two's complement,
//! so INT64_MIN/MAX are distinct lanes, never a trap).

use crate::core::hash::{next_probe, probe_step};

/// Join-local codes (symbol-local, same convention as the frozen
/// int32 lane: `-2` = duplicate, `-3` = malformed geometry).
pub const JOIN_DUP: i32 = -2;
pub const JOIN_GEOM: i32 = -3;

/// Max fan-out for the MT probe (mirrors the int32 lane).
const PROBE_THREADS_MAX: usize = 64;

/// `cap` valid iff power of two and `cap >= 2*s` (`s > 0`).
#[inline]
fn valid_cap(cap: usize, s: usize) -> bool {
    cap.is_power_of_two() && cap >= s.saturating_mul(2)
}

/// Build the right hash table. `right.len() == s`; table lanes
/// `table_keys`/`table_pos`/`table_occ` each `len >= cap`.
/// `occ[..cap]` zeroed first (WARM). Returns `OK`, `JOIN_DUP`
/// (first duplicate, prior lanes written), `JOIN_GEOM`.
#[inline]
pub fn join_build_i64(
    right: &[i64],
    s: usize,
    table_keys: &mut [i64],
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
        let mut slot = probe_step(k as u64, mask);
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
#[inline]
fn join_probe_one_i64(
    table_keys: &[i64],
    table_pos: &[i32],
    table_occ: &[u8],
    mask: usize,
    k: i64,
) -> Option<i32> {
    let mut slot = probe_step(k as u64, mask);
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
fn join_probe_shard_i64(
    table_keys: &[i64],
    table_pos: &[i32],
    table_occ: &[u8],
    mask: usize,
    left: &[i64],
    pos_out: &mut [i64],
    hit_out: &mut [u8],
) {
    debug_assert_eq!(pos_out.len(), left.len());
    debug_assert_eq!(hit_out.len(), left.len());
    for i in 0..left.len() {
        match join_probe_one_i64(table_keys, table_pos, table_occ, mask, left[i]) {
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

/// MT probe over `n` rows, `nthreads` clamped to `[1, 64]` (0 -> 1).
/// Table lanes `len >= cap`, outputs `len >= n`. Read-only table is
/// shared across scoped threads; output shards are disjoint.
/// Returns `OK` / `JOIN_GEOM` (cap not power of two).
#[inline]
pub fn join_probe_i64(
    table_keys: &[i64],
    table_pos: &[i32],
    table_occ: &[u8],
    cap: usize,
    left: &[i64],
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
    let key_chunks: Vec<&[i64]> = left[..n].chunks(cs).collect();
    let pos_chunks: Vec<&mut [i64]> = pos_out[..n].chunks_mut(cs).collect();
    let hit_chunks: Vec<&mut [u8]> = hit_out[..n].chunks_mut(cs).collect();
    std::thread::scope(|scope| {
        for ((kc, pc), hc) in key_chunks.into_iter().zip(pos_chunks).zip(hit_chunks) {
            scope.spawn(move || {
                join_probe_shard_i64(table_keys, table_pos, table_occ, mask, kc, pc, hc);
            });
        }
    });
    crate::core::errors::OK
}
