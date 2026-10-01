// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Stable LSD radix argsort permutation (Sort M1 research, additive).
//!
//! Replaces `np.argsort(k, kind="stable")` (and the double-argsort
//! `_stable_desc_idx`) with a native O(n) permutation. Exact stable
//! semantics: LSD byte passes with stable scatter = ties keep input
//! order, bit-for-bit identical to `np.argsort(kind="stable")`.
//! Descending folds into the init copy (`!` of the order-preserving
//! word): ascending on inverted words = descending stable, ties still
//! in input order — identical to the group-reverse in `_stable_desc_idx`.
//!
//! Same buffers in/out discipline as the other kernels: caller-owned
//! scratch, no alloc, no collections, `std` only at the FFI wrap.
//! M1 is single-threaded over the passes (matches ST `np.argsort`);
//! no `nthreads` param (no speculative options).

/// One stable counting-scatter pass on byte `shift` of the words.
/// `nxt_*` fully overwritten; `cur_*` read-only. Caller guarantees
/// `len >= n` on all four lanes.
#[inline]
fn scatter_pass<W: Copy + Into<u64>>(
    cur_w: &[W],
    cur_p: &[i32],
    nxt_w: &mut [W],
    nxt_p: &mut [i32],
    n: usize,
    shift: u32,
) {
    let mut pos = [0usize; 256];
    for i in 0..n {
        pos[((cur_w[i].into() >> shift) & 0xFF) as usize] += 1;
    }
    let mut sum = 0usize;
    for p in pos.iter_mut() {
        let c = *p;
        *p = sum;
        sum += c;
    }
    for i in 0..n {
        let b = ((cur_w[i].into() >> shift) & 0xFF) as usize;
        let d = pos[b];
        pos[b] = d + 1;
        nxt_w[d] = cur_w[i];
        nxt_p[d] = cur_p[i];
    }
}

/// Stable ascending/descending permutation of int32 keys.
/// Pair buffers ping-pong `A=(tmp0, perm_out)` / `B=(tmp1, tmpp)`;
/// 4 passes (even) land the permutation back in `perm_out`.
/// Caller guarantees `perm_out/tmp0/tmp1/tmpp` each `len >= n`,
/// `n <= i32::MAX`.
#[inline]
pub fn sort_perm_i32(
    keys: &[i32],
    descending: bool,
    perm_out: &mut [i32],
    tmp0: &mut [u32],
    tmp1: &mut [u32],
    tmpp: &mut [i32],
) {
    let n = keys.len();
    debug_assert!(perm_out.len() >= n);
    debug_assert!(tmp0.len() >= n);
    debug_assert!(tmp1.len() >= n);
    debug_assert!(tmpp.len() >= n);
    for i in 0..n {
        let mut w = (keys[i] as u32) ^ 0x8000_0000;
        if descending {
            w = !w;
        }
        tmp0[i] = w;
        perm_out[i] = i as i32;
    }
    for pass in 0..4 {
        let shift = (pass * 8) as u32;
        if pass % 2 == 0 {
            scatter_pass(&tmp0[..n], &perm_out[..n], &mut tmp1[..n], &mut tmpp[..n], n, shift);
        } else {
            scatter_pass(&tmp1[..n], &tmpp[..n], &mut tmp0[..n], &mut perm_out[..n], n, shift);
        }
    }
}

/// Stable ascending/descending permutation of int64 keys.
/// Same layout as [`sort_perm_i32`] with 8 passes (even) over u64 words.
#[inline]
pub fn sort_perm_i64(
    keys: &[i64],
    descending: bool,
    perm_out: &mut [i32],
    tmp0: &mut [u64],
    tmp1: &mut [u64],
    tmpp: &mut [i32],
) {
    let n = keys.len();
    debug_assert!(perm_out.len() >= n);
    debug_assert!(tmp0.len() >= n);
    debug_assert!(tmp1.len() >= n);
    debug_assert!(tmpp.len() >= n);
    for i in 0..n {
        let mut w = (keys[i] as u64) ^ 0x8000_0000_0000_0000;
        if descending {
            w = !w;
        }
        tmp0[i] = w;
        perm_out[i] = i as i32;
    }
    for pass in 0..8 {
        let shift = (pass * 8) as u32;
        if pass % 2 == 0 {
            scatter_pass(&tmp0[..n], &perm_out[..n], &mut tmp1[..n], &mut tmpp[..n], n, shift);
        } else {
            scatter_pass(&tmp1[..n], &tmpp[..n], &mut tmp0[..n], &mut perm_out[..n], n, shift);
        }
    }
}
