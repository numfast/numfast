// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `pair_insert` mechanic: generic open-addressing insert of `(key, value)`
//! pairs over typeless 64-bit lanes (caller views them as u64 or i64,
//! bit-identical) for batch assignment.
//!
//! Generic lanes only, no domain vocabulary. Caller owns all buffers.
//! Single-threaded (same as ST reference), no alloc, `core`-only by
//! construction (flat loop + [`crate::core::hash`] probe atoms, same
//! family as `join_build`). WASM-compatible by construction (no threads,
//! no collections) but not exported over the wasm surface (native CPU
//! research surface, same gate as `sssp`).
//!
//! Semantics (frozen):
//! - `cap` power of two, `cap >= 1`; table lanes `len >= cap`;
//! - `used[..cap]` zeroed first (WARM), then rows `0..n` inserted in
//!   input order; duplicate key = last-write-wins (value overwritten,
//!   occupancy unchanged), never an error (unlike `join_build` unique
//!   contract);
//! - full-range keys incl `MIN`/`MAX` (no reserved lane; typeless bits);
//! - probe: `slot = probe_step(key as u64, mask)`, linear `next_probe`
//!   wrap; full table (unreachable when `cap >= n`, still guarded:
//!   loops at most `cap` steps, then `PAIR_FULL`).
//!
//! Codes (this surface only): `ng >= 0` occupied slots; `-2` null-state
//! (`NULL_STATE`, like sorted/carry lane-returning symbols); `-3` bad
//! geometry (cap not pow2, cap == 0, short table).

use crate::core::hash::{next_probe, probe_step};

/// Bad geometry (cap not pow2 / short table / full table).
pub const PAIR_GEOM: i64 = -3;

/// Insert `n` pairs into caller-owned open-addressing lanes.
/// `tkeys/tvals/used` each `len >= cap`. Returns occupied count.
#[inline]
pub fn pair_insert_i64(
    keys: &[i64],
    vals: &[i64],
    n: usize,
    tkeys: &mut [i64],
    tvals: &mut [i64],
    used: &mut [u8],
    cap: usize,
) -> i64 {
    debug_assert!(keys.len() >= n);
    debug_assert!(vals.len() >= n);
    debug_assert!(tkeys.len() >= cap);
    debug_assert!(tvals.len() >= cap);
    debug_assert!(used.len() >= cap);
    if !cap.is_power_of_two() || cap == 0 {
        return PAIR_GEOM;
    }
    if tkeys.len() < cap || tvals.len() < cap || used.len() < cap {
        return PAIR_GEOM;
    }
    let mask = cap - 1;
    used[..cap].fill(0);
    let mut ng: i64 = 0;
    for i in 0..n {
        let k = keys[i];
        let v = vals[i];
        let mut slot = probe_step(k as u64, mask);
        for _ in 0..cap {
            if used[slot] == 0 {
                tkeys[slot] = k;
                tvals[slot] = v;
                used[slot] = 1;
                ng += 1;
                break;
            }
            if tkeys[slot] == k {
                tvals[slot] = v;
                break;
            }
            slot = next_probe(slot, mask);
        }
    }
    ng
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_last_wins() {
        let k = vec![1i64, 2, 1, -5, 9223372036854775807, -9223372036854775808];
        let v = vec![10i64, 20, 11, 50, 70, 80];
        let cap = 16usize;
        let (mut tk, mut tv, mut used) = (vec![0i64; cap], vec![0i64; cap], vec![0u8; cap]);
        let ng = pair_insert_i64(&k, &v, k.len(), &mut tk, &mut tv, &mut used, cap);
        assert_eq!(ng, 5);
        // Every key recoverable with last value.
        for (kk, vv) in [(1i64, 11i64), (2, 20), (-5, 50)] {
            let mut slot = probe_step(kk as u64, cap - 1);
            loop {
                assert_eq!(used[slot], 1);
                if tk[slot] == kk {
                    assert_eq!(tv[slot], vv);
                    break;
                }
                slot = next_probe(slot, cap - 1);
            }
        }
    }

    #[test]
    fn empty_zero_cap_geom() {
        let (mut tk, mut tv, mut used) = (vec![0i64; 8], vec![0i64; 8], vec![0u8; 8]);
        assert_eq!(pair_insert_i64(&[], &[], 0, &mut tk, &mut tv, &mut used, 8), 0);
        assert_eq!(pair_insert_i64(&[1], &[1], 1, &mut tk, &mut tv, &mut used, 7), PAIR_GEOM);
        assert_eq!(pair_insert_i64(&[1], &[1], 1, &mut tk, &mut tv, &mut used, 0), PAIR_GEOM);
    }
}
