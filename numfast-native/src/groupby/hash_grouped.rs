// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Generic grouped COUNT DISTINCT pair-hash mechanics (native GroupedHashMT).
//!
//! Native port of the proven Numba `GroupedHashMT` lane
//! (`src/Drivers/GroupedHashMT/_lib/grouped_mt_hash.py`, old lane
//! intact as fallback/reference): partitioned open-addressing hash
//! over composite (key, value) int64 pairs with fingerprint routing,
//! per-partition slot tables, multi-threaded execution, caller-owned
//! reusable buffers.
//!
//! Algorithm (identical to the Numba lane, generic only):
//! - rows are routed by pair fingerprint into `P` partitions, `P` from
//!   `N` only (power of two, `16 <= P <= 256`), no cardinality input;
//! - each partition owns a linear-probing table over (key, value)
//!   pairs with full-key compare on every probe (exact by
//!   construction, independent of fingerprint quality);
//! - occupied keys are collected per partition, then counted by the
//!   caller-side adaptive unique (range-measured dense bincount,
//!   else sort fallback — same rule as the Numba lane).
//!
//! Integer generality: the kernels are int64 lanes. Narrower integer
//! dtypes widen exactly (int32/uint32 -> int64 injective); uint64
//! lanes above `i64::MAX` are rejected caller-side (fallback owns
//! them, never a silent wrap). Non-integer dtypes never reach here.
//! Dtypes are preserved in the widened domain (no int32 narrowing).
//!
//! Threading: `std::thread::scope`, `nthreads` clamped to `[1, 64]`
//! (0 -> 1), same convention as [`crate::join`]. Fingerprint,
//! per-thread counting and scatter shard over row ranges; insert /
//! collect shard over partition ranges (disjoint table lanes, no
//! atomics, no merge — pre-split reborrows, fully safe). The
//! front mirrors the proven lane exactly: per-thread counts ->
//! caller-side prefix (`tpos`) -> reservation scatter, so the
//! per-partition SET (and hence the final map) is bit-identical at
//! any thread count (scatter order inside a partition is not output).
//!
//! Buffers are caller-owned, all contiguous, overwritten as
//! documented; no allocation on the data path. Return codes: `OK` /
//! `GH_GEOM` (-3, bad geometry: `P` not a power of two, `P == 0`;
//! null pointers are rejected FFI-side with -1).

/// Symbol-local geometry code (same convention as `JOIN_GEOM`).
pub const GH_GEOM: i32 = -3;

/// Max fan-out for the MT kernels (same clamp as the join probe).
const GH_THREADS_MAX: usize = 64;

/// Hard cap on partitions (matches the proven lane: `P <= 256`).
pub const GH_PART_MAX: usize = 256;

/// Pair mix constants (identical to the proven Numba lane).
const C1: u64 = 0x9E37_79B9_7F4A_7C15;
const C2: u64 = 0x00BF_5847_6D1F_F973;

/// Pair fingerprint: `uk*C1 + ((uv ^ (uv >> 33))*C2)`, uint64
/// wrapping (bit-identical to the Numba `uint64` arithmetic,
/// including negative inputs reinterpreted as two's complement).
#[inline(always)]
fn pair_fp(k: i64, v: i64) -> u64 {
    let uk = k as u64;
    let uv = v as u64;
    uk.wrapping_mul(C1)
        .wrapping_add((uv ^ (uv >> 33)).wrapping_mul(C2))
}

/// Clamp fan-out to `[1, 64]` (0 -> 1), capped by work items.
#[inline]
fn fanout(nthreads: usize, work: usize) -> usize {
    nthreads.clamp(1, GH_THREADS_MAX).min(work.max(1))
}

/// One fingerprint shard: rows `k/v` -> `fp` (partition ids).
/// Flat loop, no allocation, read-only inputs (thread-shareable).
#[inline]
fn fp_shard(k: &[i64], v: &[i64], pmask: u64, fp: &mut [i64]) {
    debug_assert_eq!(k.len(), v.len());
    debug_assert_eq!(fp.len(), k.len());
    for i in 0..k.len() {
        fp[i] = (pair_fp(k[i], v[i]) & pmask) as i64;
    }
}

/// MT fingerprint over `n` rows. `k/v/fp` each `len >= n`.
/// Returns `OK`.
#[inline]
pub fn gh_fp(k: &[i64], v: &[i64], n: usize, fp: &mut [i64], pmask: u64, nthreads: usize) -> i32 {
    debug_assert_eq!(k.len(), n);
    debug_assert_eq!(v.len(), n);
    debug_assert!(fp.len() >= n);
    if n == 0 {
        return crate::core::errors::OK;
    }
    let t = fanout(nthreads, n);
    let cs = n.div_ceil(t);
    // Owned chunks, one per scoped thread (disjoint reborrows,
    // carved sequentially so ranges tile [0..n) in order).
    let mut k_rest: &[i64] = &k[..n];
    let mut v_rest: &[i64] = &v[..n];
    let mut fp_rest: &mut [i64] = &mut fp[..n];
    let mut k_chunks: Vec<&[i64]> = Vec::with_capacity(t);
    let mut v_chunks: Vec<&[i64]> = Vec::with_capacity(t);
    let mut fp_chunks: Vec<&mut [i64]> = Vec::with_capacity(t);
    for tt in 0..t {
        let s0 = (tt * cs).min(n);
        let s1 = ((tt + 1) * cs).min(n);
        if s0 >= s1 {
            continue;
        }
        let len = s1 - s0;
        let (a, ra) = k_rest.split_at(len);
        k_rest = ra;
        k_chunks.push(a);
        let (b, rb) = v_rest.split_at(len);
        v_rest = rb;
        v_chunks.push(b);
        let (c, rc) = fp_rest.split_at_mut(len);
        fp_rest = rc;
        fp_chunks.push(c);
    }
    std::thread::scope(|scope| {
        for ((kk, vv), ff) in k_chunks.into_iter().zip(v_chunks).zip(fp_chunks) {
            scope.spawn(move || fp_shard(kk, vv, pmask, ff));
        }
    });
    crate::core::errors::OK
}

/// One count shard: rows `fp` -> per-thread histogram row
/// `tc_row[q]` (disjoint across threads, plain adds, no atomics).
/// Flat loop, no allocation.
#[inline]
fn count_shard(fp: &[i64], tc_row: &mut [i64]) {
    for &f in fp {
        tc_row[f as usize] += 1;
    }
}

/// MT per-thread histogram over `n` rows. `fp` `len >= n`,
/// `tc` `len >= t * p` (WARM-zeroed by the caller; each thread
/// owns row `tt`, disjoint, no atomics). `t`/`p` are the fan-out
/// used for the matching scatter pass. Returns `OK` / `GH_GEOM`.
#[inline]
pub fn gh_count(fp: &[i64], n: usize, tc: &mut [i64], t: usize, p: usize, nthreads: usize) -> i32 {
    debug_assert_eq!(fp.len(), n);
    debug_assert!(tc.len() >= t * p);
    if p == 0 || p > GH_PART_MAX || !p.is_power_of_two() {
        return GH_GEOM;
    }
    if t == 0 || t > GH_THREADS_MAX {
        return GH_GEOM;
    }
    if n == 0 {
        return crate::core::errors::OK;
    }
    let nt = fanout(nthreads, n).min(t);
    let cs = n.div_ceil(nt);
    // Owned row chunks + disjoint histogram rows (carved in order).
    let mut fp_rest: &[i64] = &fp[..n];
    let mut tc_rest: &mut [i64] = &mut tc[..t * p];
    let mut fp_chunks: Vec<&[i64]> = Vec::with_capacity(nt);
    let mut tc_chunks: Vec<&mut [i64]> = Vec::with_capacity(nt);
    for tt in 0..nt {
        let s0 = (tt * cs).min(n);
        let s1 = ((tt + 1) * cs).min(n);
        if s0 >= s1 {
            continue;
        }
        let len = s1 - s0;
        let (a, ra) = fp_rest.split_at(len);
        fp_rest = ra;
        fp_chunks.push(a);
        // Histogram rows tile [0..t) in chunk order (idle tail
        // rows stay zero); the scatter pass uses the same order.
        let (b, rb) = tc_rest.split_at_mut(p);
        tc_rest = rb;
        tc_chunks.push(b);
    }
    std::thread::scope(|scope| {
        for (ff, tc_row) in fp_chunks.into_iter().zip(tc_chunks) {
            scope.spawn(move || count_shard(ff, tc_row));
        }
    });
    crate::core::errors::OK
}

/// One reservation-scatter shard: rows `k/v/fp` -> `pk/pv` lanes
/// from the private cursor row (`tpos_row`, copied on the stack).
/// Reservations are disjoint by prefix construction (same rule as
/// the proven lane). Flat loop, no allocation.
#[inline]
fn scatter_shard(
    k: &[i64],
    v: &[i64],
    fp: &[i64],
    tpos_row: &[i64],
    pk_base: usize,
    pv_base: usize,
    pk_len: usize,
    pv_len: usize,
) {
    debug_assert_eq!(k.len(), v.len());
    debug_assert_eq!(fp.len(), k.len());
    // SAFETY: lanes are disjoint by prefix reservation (caller
    // prefix over per-thread counts); bases outlive the scope.
    let pk: &mut [i64] = unsafe { std::slice::from_raw_parts_mut(pk_base as *mut i64, pk_len) };
    let pv: &mut [i64] = unsafe { std::slice::from_raw_parts_mut(pv_base as *mut i64, pv_len) };
    let mut cur = [0i64; GH_PART_MAX];
    cur[..tpos_row.len()].copy_from_slice(tpos_row);
    for i in 0..k.len() {
        let p = fp[i] as usize;
        let lane = cur[p] as usize;
        cur[p] += 1;
        pk[lane] = k[i];
        pv[lane] = v[i];
    }
}

/// MT reservation scatter over `n` rows into `p` partitions.
/// `k/v/fp` `len >= n`, `tpos` `len >= t * p` (caller prefix:
/// thread `tt` owns `tpos[tt * p + q]` bases), `pk/pv`
/// `len >= n` (overwritten). Row chunks tile `[0..n)` in the same
/// order as the matching count pass, so reservations line up.
/// Returns `OK` / `GH_GEOM`.
#[inline]
pub fn gh_scatter(
    k: &[i64],
    v: &[i64],
    fp: &[i64],
    n: usize,
    tpos: &[i64],
    t: usize,
    p: usize,
    pk: &mut [i64],
    pv: &mut [i64],
    nthreads: usize,
) -> i32 {
    debug_assert_eq!(k.len(), n);
    debug_assert_eq!(v.len(), n);
    debug_assert_eq!(fp.len(), n);
    debug_assert!(tpos.len() >= t * p);
    debug_assert!(pk.len() >= n);
    debug_assert!(pv.len() >= n);
    if p == 0 || p > GH_PART_MAX || !p.is_power_of_two() {
        return GH_GEOM;
    }
    if t == 0 || t > GH_THREADS_MAX {
        return GH_GEOM;
    }
    if n == 0 {
        return crate::core::errors::OK;
    }
    let nt = fanout(nthreads, n).min(t);
    let cs = n.div_ceil(nt);
    let pk_base = pk.as_mut_ptr() as usize;
    let pv_base = pv.as_mut_ptr() as usize;
    let pk_len = pk.len();
    let pv_len = pv.len();
    // Owned row chunks + matching tpos rows (same tiling as count).
    let mut k_rest: &[i64] = &k[..n];
    let mut v_rest: &[i64] = &v[..n];
    let mut fp_rest: &[i64] = &fp[..n];
    let mut k_chunks: Vec<&[i64]> = Vec::with_capacity(nt);
    let mut v_chunks: Vec<&[i64]> = Vec::with_capacity(nt);
    let mut fp_chunks: Vec<&[i64]> = Vec::with_capacity(nt);
    let mut tp_chunks: Vec<&[i64]> = Vec::with_capacity(nt);
    let mut ti = 0usize;
    for tt in 0..nt {
        let s0 = (tt * cs).min(n);
        let s1 = ((tt + 1) * cs).min(n);
        if s0 >= s1 {
            continue;
        }
        let len = s1 - s0;
        let (a, ra) = k_rest.split_at(len);
        k_rest = ra;
        k_chunks.push(a);
        let (b, rb) = v_rest.split_at(len);
        v_rest = rb;
        v_chunks.push(b);
        let (c, rc) = fp_rest.split_at(len);
        fp_rest = rc;
        fp_chunks.push(c);
        tp_chunks.push(&tpos[ti * p..(ti + 1) * p]);
        ti += 1;
    }
    std::thread::scope(|scope| {
        for (((kk, vv), ff), tp) in k_chunks
            .into_iter()
            .zip(v_chunks)
            .zip(fp_chunks)
            .zip(tp_chunks)
        {
            scope.spawn(move || {
                scatter_shard(kk, vv, ff, tp, pk_base, pv_base, pk_len, pv_len);
            });
        }
    });
    crate::core::errors::OK
}

/// One insert shard over `tm_loc.len()` local partitions: each
/// partition serially inserts its rows with full (key, value)
/// compare (exact by construction). `pk_chunk/pv_chunk` hold
/// exactly the shard rows; `starts`/`toff` are the GLOBAL ranges
/// (`plo` = first global partition); `r0`/`b0` rebase chunk-local
/// row/table indices; `tm_loc/tk_chunk/tv_chunk/used_chunk` are
/// disjoint pre-split lanes. Flat loops, no allocation.
#[inline]
#[allow(clippy::too_many_arguments)]
fn pins_shard(
    pk_chunk: &[i64],
    pv_chunk: &[i64],
    starts: &[i64],
    plo: usize,
    toff: &[i64],
    tm_loc: &[i64],
    tk_chunk: &mut [i64],
    tv_chunk: &mut [i64],
    used_chunk: &mut [u8],
    r0: usize,
    b0: usize,
) {
    let np = tm_loc.len();
    for q in 0..np {
        let tbase = (toff[plo + q] as usize) - b0;
        let mask = tm_loc[q] as u64;
        let mask_us = mask as usize;
        let slots = mask_us + 1;
        let j0 = (starts[plo + q] as usize) - r0;
        let j1 = (starts[plo + q + 1] as usize) - r0;
        debug_assert!(j1 <= pk_chunk.len());
        debug_assert!(j1 <= pv_chunk.len());
        debug_assert!(tbase + slots <= tk_chunk.len());
        debug_assert!(tbase + slots <= tv_chunk.len());
        debug_assert!(tbase + slots <= used_chunk.len());
        for j in j0..j1 {
            // SAFETY: `j` in the asserted row window; `s = tbase + h`
            // with `h < slots` by mask construction (`mask = slots - 1`,
            // power-of-two minus one: initial `fp & mask` and every
            // `(h + 1) & mask` stay in `[0..slots)`), and the table
            // window covers `tbase + slots` (asserted). Bounds checks
            // are hoisted out of the probe loop.
            let rk = unsafe { *pk_chunk.get_unchecked(j) };
            let vv = unsafe { *pv_chunk.get_unchecked(j) };
            let mut h = (pair_fp(rk, vv) & mask) as usize;
            loop {
                let s = tbase + h;
                if unsafe { *used_chunk.get_unchecked(s) } == 0 {
                    unsafe {
                        *used_chunk.get_unchecked_mut(s) = 1;
                        *tk_chunk.get_unchecked_mut(s) = rk;
                        *tv_chunk.get_unchecked_mut(s) = vv;
                    }
                    break;
                } else if unsafe {
                    *tk_chunk.get_unchecked(s) == rk && *tv_chunk.get_unchecked(s) == vv
                } {
                    break;
                }
                h = (h + 1) & mask_us;
            }
        }
    }
}

/// MT pair-hash insert over `p` partitions. `pk/pv` `len >= n`
/// (`n == starts[p]`), `starts/toff` `len >= p + 1`, `tm`
/// `len >= p` (`tm[q] = slots - 1`, power-of-two minus one),
/// `tk/tv/used` `len >= m` (`m == toff[p]`, `used` zeroed first —
/// WARM). Partition ranges are contiguous in row space and table
/// space, so shards take pre-split disjoint reborrows (no atomics,
/// no merge, fully safe). Returns `OK` / `GH_GEOM`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn gh_pins(
    pk: &[i64],
    pv: &[i64],
    starts: &[i64],
    toff: &[i64],
    tm: &[i64],
    tk: &mut [i64],
    tv: &mut [i64],
    used: &mut [u8],
    p: usize,
    nthreads: usize,
) -> i32 {
    debug_assert!(starts.len() >= p + 1);
    debug_assert!(toff.len() >= p + 1);
    debug_assert!(tm.len() >= p);
    if p == 0 || p > GH_PART_MAX || !p.is_power_of_two() {
        return GH_GEOM;
    }
    let n = starts[p] as usize;
    let m = toff[p] as usize;
    debug_assert!(pk.len() >= n);
    debug_assert!(pv.len() >= n);
    debug_assert!(tk.len() >= m);
    debug_assert!(tv.len() >= m);
    debug_assert!(used.len() >= m);
    let t = fanout(nthreads, p);
    let cs = p.div_ceil(t);
    // Contiguous carve in issue order: each range starts where the
    // previous ended (ranges tile [0..n) / [0..m) / [0..p]).
    let mut pk_rest: &[i64] = &pk[..n];
    let mut pv_rest: &[i64] = &pv[..n];
    let mut tk_rest: &mut [i64] = &mut tk[..m];
    let mut tv_rest: &mut [i64] = &mut tv[..m];
    let mut used_rest: &mut [u8] = &mut used[..m];
    // Borrow the disjoint pieces first (NLL: sequential splits).
    let mut pk_parts: Vec<&[i64]> = Vec::with_capacity(t);
    let mut pv_parts: Vec<&[i64]> = Vec::with_capacity(t);
    let mut tk_parts: Vec<&mut [i64]> = Vec::with_capacity(t);
    let mut tv_parts: Vec<&mut [i64]> = Vec::with_capacity(t);
    let mut used_parts: Vec<&mut [u8]> = Vec::with_capacity(t);
    let mut ranges: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(t);
    for tt in 0..t {
        let plo = (tt * cs).min(p);
        let phi = ((tt + 1) * cs).min(p);
        if plo >= phi {
            continue;
        }
        let r0 = starts[plo] as usize;
        let r1 = starts[phi] as usize;
        let b0 = toff[plo] as usize;
        let b1 = toff[phi] as usize;
        let (a, ra) = pk_rest.split_at(r1 - r0);
        pk_rest = ra;
        pk_parts.push(a);
        let (b, rb) = pv_rest.split_at(r1 - r0);
        pv_rest = rb;
        pv_parts.push(b);
        let (c, rc) = tk_rest.split_at_mut(b1 - b0);
        tk_rest = rc;
        tk_parts.push(c);
        let (d, rd) = tv_rest.split_at_mut(b1 - b0);
        tv_rest = rd;
        tv_parts.push(d);
        let (e, re) = used_rest.split_at_mut(b1 - b0);
        used_rest = re;
        used_parts.push(e);
        ranges.push((plo, phi, r0, b0));
    }
    std::thread::scope(|scope| {
        let shards = pk_parts
            .into_iter()
            .zip(pv_parts)
            .zip(tk_parts)
            .zip(tv_parts)
            .zip(used_parts)
            .zip(ranges);
        for (((((pk_c, pv_c), tk_c), tv_c), used_c), (plo, phi, r0, b0)) in shards {
            let tm_c: &[i64] = &tm[plo..phi];
            scope.spawn(move || {
                pins_shard(pk_c, pv_c, starts, plo, toff, tm_c, tk_c, tv_c, used_c, r0, b0);
            });
        }
    });
    crate::core::errors::OK
}

/// One occupancy-count shard: local partitions `[0..np)` ->
/// `cnt_loc[q]` = used slots in the local table window.
/// `toff` GLOBAL (`plo` = first global partition), `b0` rebases
/// the table window. Flat loop, no allocation.
#[inline]
fn occ_count_shard(
    used_chunk: &[u8],
    toff: &[i64],
    plo: usize,
    cnt_loc: &mut [i64],
    b0: usize,
) {
    let np = cnt_loc.len();
    for q in 0..np {
        let s0 = (toff[plo + q] as usize) - b0;
        let s1 = (toff[plo + q + 1] as usize) - b0;
        let mut c: i64 = 0;
        for s in s0..s1 {
            if used_chunk[s] != 0 {
                c += 1;
            }
        }
        cnt_loc[q] = c;
    }
}

/// MT occupancy count over `p` partitions. `used` `len >= m`
/// (`m == toff[p]`), `toff` `len >= p + 1`, `cnt` `len >= p`
/// (overwritten). Disjoint pre-split reborrows, fully safe.
/// Returns `OK` / `GH_GEOM`.
#[inline]
pub fn gh_occ_count(used: &[u8], toff: &[i64], p: usize, cnt: &mut [i64], nthreads: usize) -> i32 {
    debug_assert!(toff.len() >= p + 1);
    debug_assert!(cnt.len() >= p);
    if p == 0 || p > GH_PART_MAX || !p.is_power_of_two() {
        return GH_GEOM;
    }
    let m = toff[p] as usize;
    debug_assert!(used.len() >= m);
    let t = fanout(nthreads, p);
    let cs = p.div_ceil(t);
    let mut used_rest: &[u8] = &used[..m];
    let mut cnt_rest: &mut [i64] = &mut cnt[..p];
    let mut used_parts: Vec<&[u8]> = Vec::with_capacity(t);
    let mut cnt_parts: Vec<&mut [i64]> = Vec::with_capacity(t);
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(t);
    for tt in 0..t {
        let plo = (tt * cs).min(p);
        let phi = ((tt + 1) * cs).min(p);
        if plo >= phi {
            continue;
        }
        let b0 = toff[plo] as usize;
        let b1 = toff[phi] as usize;
        let (a, ra) = used_rest.split_at(b1 - b0);
        used_rest = ra;
        used_parts.push(a);
        let (b, rb) = cnt_rest.split_at_mut(phi - plo);
        cnt_rest = rb;
        cnt_parts.push(b);
        ranges.push((plo, b0));
    }
    std::thread::scope(|scope| {
        let shards = used_parts.into_iter().zip(cnt_parts).zip(ranges);
        for ((u_c, c_c), (plo, b0)) in shards {
            scope.spawn(move || {
                occ_count_shard(u_c, toff, plo, c_c, b0);
            });
        }
    });
    crate::core::errors::OK
}

/// One occupancy-fill shard: `np` local partitions -> occupied
/// `tk` keys at `ok_chunk[w++]` (caller prefix offsets make the
/// chunk ranges disjoint across shards). `toff` GLOBAL (`plo` =
/// first global partition), `b0` rebases the table window.
/// Flat loop, no allocation.
#[inline]
fn occ_fill_shard(
    tk_chunk: &[i64],
    used_chunk: &[u8],
    toff: &[i64],
    plo: usize,
    np: usize,
    ok_chunk: &mut [i64],
    b0: usize,
) {
    debug_assert_eq!(tk_chunk.len(), used_chunk.len());
    let mut w = 0usize;
    for q in 0..np {
        let s0 = (toff[plo + q] as usize) - b0;
        let s1 = (toff[plo + q + 1] as usize) - b0;
        debug_assert!(s1 <= tk_chunk.len());
        for s in s0..s1 {
            if used_chunk[s] != 0 {
                ok_chunk[w] = tk_chunk[s];
                w += 1;
            }
        }
    }
    debug_assert!(w <= ok_chunk.len());
}

/// MT occupancy fill over `p` partitions. `tk/used` `len >= m`
/// (`m == toff[p]`), `toff/ostart` `len >= p + 1`, `ok`
/// `len >= nocc` (`nocc == ostart[p]`, overwritten). Output
/// ranges are contiguous prefix slices, pre-split disjoint across
/// scoped threads (fully safe). Returns `OK` / `GH_GEOM`.
#[inline]
pub fn gh_occ_fill(
    tk: &[i64],
    used: &[u8],
    toff: &[i64],
    ostart: &[i64],
    ok: &mut [i64],
    p: usize,
    nthreads: usize,
) -> i32 {
    debug_assert!(toff.len() >= p + 1);
    debug_assert!(ostart.len() >= p + 1);
    if p == 0 || p > GH_PART_MAX || !p.is_power_of_two() {
        return GH_GEOM;
    }
    let m = toff[p] as usize;
    let nocc = ostart[p] as usize;
    debug_assert!(tk.len() >= m);
    debug_assert!(used.len() >= m);
    debug_assert!(ok.len() >= nocc);
    let t = fanout(nthreads, p);
    let cs = p.div_ceil(t);
    let mut tk_rest: &[i64] = &tk[..m];
    let mut used_rest: &[u8] = &used[..m];
    let mut ok_rest: &mut [i64] = &mut ok[..nocc];
    let mut tk_parts: Vec<&[i64]> = Vec::with_capacity(t);
    let mut used_parts: Vec<&[u8]> = Vec::with_capacity(t);
    let mut ok_parts: Vec<&mut [i64]> = Vec::with_capacity(t);
    let mut ranges: Vec<(usize, usize, usize)> = Vec::with_capacity(t);
    for tt in 0..t {
        let plo = (tt * cs).min(p);
        let phi = ((tt + 1) * cs).min(p);
        if plo >= phi {
            continue;
        }
        let b0 = toff[plo] as usize;
        let b1 = toff[phi] as usize;
        let w0 = ostart[plo] as usize;
        let w1 = ostart[phi] as usize;
        let (a, ra) = tk_rest.split_at(b1 - b0);
        tk_rest = ra;
        tk_parts.push(a);
        let (b, rb) = used_rest.split_at(b1 - b0);
        used_rest = rb;
        used_parts.push(b);
        let (c, rc) = ok_rest.split_at_mut(w1 - w0);
        ok_rest = rc;
        ok_parts.push(c);
        ranges.push((plo, phi - plo, b0));
    }
    std::thread::scope(|scope| {
        let shards = tk_parts
            .into_iter()
            .zip(used_parts)
            .zip(ok_parts)
            .zip(ranges);
        for (((tk_c, u_c), ok_c), (plo, np, b0)) in shards {
            scope.spawn(move || {
                occ_fill_shard(tk_c, u_c, toff, plo, np, ok_c, b0);
            });
        }
    });
    crate::core::errors::OK
}
