# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Candidates A-G, one contract: run(keys, vals, g, p=16, threads=T).

A row-parallel flat dense + sum-merge (baseline, TxG replication).
B local flat preagg per worker -> partial stream -> ONE single-T Fshift
  over partial records (measures partial_rows; ~N => scheme pointless).
C per-worker Fshift over row blocks -> replica-sum single merge (TxPxW).
D partition-parallel full-scan, contiguous ownership, fused pid, no pid
  array, no replication (total acc = G). Cost: T x N reads.
E same as D but LPT ownership on sampled (100K) partition loads.
F shared single (PxW) accumulator, per-partition locks, sub-block batching
  (no replication, no pid array, exact). Fiserver block=8K; Fbig block=64K.
G radix-partitioned (count+scatter pass) -> parallel per-partition flat
  agg. Has explicit partition pass (reference for its honest price).

Golden-exact (bit-identical int64) in all. Research-only.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .contract import State, empty_state, info
from .kernels import (HAS_NUMBA, geom, k_flat_part, k_flat_slice,
                      k_fshift_owned, k_fshift_ownerpos, k_fshift_slice,
                      k_fshift_weighted, k_pid_hist, k_scatter_fused,
                      k_sub_apply_perm, k_sub_fill, k_sub_hist)

P_DEF = 16


def _key_of_shift(q, li, w):
    return q * np.int64(w) + li


def _compact_shift(sums, counts, w, g, p):
    uk, ss, cc = [], [], []
    for q in range(p):
        nz = np.flatnonzero(counts[q] > 0)
        if nz.size:
            uk.append((_key_of_shift(np.int64(q), nz.astype(np.int64),
                                     np.int64(w))).astype(np.int64))
            ss.append(sums[q][nz])
            cc.append(counts[q][nz])
    if not uk:
        return empty_state()
    ukeys = np.concatenate(uk)
    # shift partitions cover increasing ranges -> already sorted; keep guard
    if bool((np.diff(ukeys) < 0).any()):
        o = np.argsort(ukeys, kind="stable")
        return State(ukeys[o], np.concatenate(ss)[o], np.concatenate(cc)[o])
    return State(ukeys, np.concatenate(ss), np.concatenate(cc))


def _compact_owned(sums_list, counts_list, plo_list, w, g, p):
    # sums_list[w]: (npo_w, W); concat in partition order -> sorted keys.
    uk, ss, cc = [], [], []
    for sums, counts, plo in zip(sums_list, counts_list, plo_list):
        for r in range(sums.shape[0]):
            q = plo + r
            nz = np.flatnonzero(counts[r] > 0)
            if nz.size:
                uk.append((_key_of_shift(np.int64(q), nz.astype(np.int64),
                                         np.int64(w))).astype(np.int64))
                ss.append(sums[r][nz])
                cc.append(counts[r][nz])
    if not uk:
        return empty_state()
    return State(np.concatenate(uk), np.concatenate(ss), np.concatenate(cc))


def _blocks(n, t):
    return np.linspace(0, n, t + 1).astype(np.int64)


def run_A(keys, vals, g, p=P_DEF, threads=1):
    k = np.asarray(keys).astype(np.int32, copy=False)
    v = np.asarray(vals).astype(np.int32, copy=False)
    n = k.size
    if n == 0:
        return empty_state(), info(0, 0, 0, 0, 0, {"cand": "A"})
    assert HAS_NUMBA
    bounds = _blocks(n, threads)
    t0 = time.perf_counter()
    part_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    sums3 = [np.zeros(g, dtype=np.int64) for _ in range(threads)]
    cnts3 = [np.zeros(g, dtype=np.int64) for _ in range(threads)]

    def _w(w):
        a, b = int(bounds[w]), int(bounds[w + 1])
        k_flat_slice(k[a:b], v[a:b], sums3[w], cnts3[w])

    if threads == 1:
        _w(0)
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(_w, range(threads)))
    agg_ms = (time.perf_counter() - t1) * 1000.0
    t2 = time.perf_counter()
    gs = sums3[0]
    gc = cnts3[0]
    for w in range(1, threads):
        gs += sums3[w]
        gc += cnts3[w]
    mask = gc > 0
    st = State(np.flatnonzero(mask).astype(np.int64), gs[mask], gc[mask])
    merge_ms = (time.perf_counter() - t2) * 1000.0
    rep = int(sum(s.nbytes + c.nbytes for s, c in zip(sums3, cnts3)))
    traffic = int(k.nbytes + v.nbytes + 8 * g * 2 * threads
                  + 8 * st.ngroups * 2)
    return st, info(part_ms, agg_ms, merge_ms, rep, traffic,
                    {"cand": "A", "threads": threads,
                     "ws_bytes": int(8 * g * 2),
                     "partial_rows": 0, "route_ms": 0.0})


def run_B(keys, vals, g, p=P_DEF, threads=1):
    # Phase1: like A (per-worker flat local). Phase2: single Fshift over
    # concatenated partials (weighted). partial_rows = sum distinct_w.
    k = np.asarray(keys).astype(np.int32, copy=False)
    v = np.asarray(vals).astype(np.int32, copy=False)
    n = k.size
    if n == 0:
        return empty_state(), info(0, 0, 0, 0, 0, {"cand": "B"})
    assert HAS_NUMBA
    gm = geom(g, p)
    shift, w = gm["shift"], gm["width"]
    bounds = _blocks(n, threads)
    t0 = time.perf_counter()
    part_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    sums3 = [np.zeros(g, dtype=np.int64) for _ in range(threads)]
    cnts3 = [np.zeros(g, dtype=np.int64) for _ in range(threads)]

    def _w(wi):
        a, b = int(bounds[wi]), int(bounds[wi + 1])
        k_flat_slice(k[a:b], v[a:b], sums3[wi], cnts3[wi])

    if threads == 1:
        _w(0)
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(_w, range(threads)))
    # emit partials
    pks, pss, pcs = [], [], []
    for wi in range(threads):
        nz = np.flatnonzero(cnts3[wi] > 0)
        if nz.size:
            pks.append(nz.astype(np.int64))
            pss.append(sums3[wi][nz])
            pcs.append(cnts3[wi][nz])
    pk = np.concatenate(pks) if pks else np.empty(0, dtype=np.int64)
    psv = np.concatenate(pss) if pss else np.empty(0, dtype=np.int64)
    pcv = np.concatenate(pcs) if pcs else np.empty(0, dtype=np.int64)
    pre_ms = (time.perf_counter() - t1) * 1000.0
    t2 = time.perf_counter()
    sums = np.zeros((p, w), dtype=np.int64)
    counts = np.zeros((p, w), dtype=np.int64)
    if pk.size:
        k_fshift_weighted(np.ascontiguousarray(pk), np.ascontiguousarray(psv),
                          np.ascontiguousarray(pcv), sums, counts,
                          shift, w, p)
    st = _compact_shift(sums, counts, w, g, p)
    shift_ms = (time.perf_counter() - t2) * 1000.0
    partial_rows = int(pk.size)
    partial_bytes = int(pk.nbytes + psv.nbytes + pcv.nbytes)
    rep = int(sum(s.nbytes + c.nbytes for s, c in zip(sums3, cnts3)))
    traffic = int(k.nbytes + v.nbytes + 8 * g * 2 * threads
                  + partial_bytes + 8 * p * w * 2 + 8 * st.ngroups * 2)
    return st, info(part_ms, pre_ms, shift_ms, rep + partial_bytes, traffic,
                    {"cand": "B", "threads": threads, "ws_bytes": int(8 * w * 2),
                     "partial_rows": partial_rows,
                     "partial_bytes": partial_bytes,
                     "partial_ratio": partial_rows / max(1, n),
                     "route_ms": 0.0})


def run_C(keys, vals, g, p=P_DEF, threads=1):
    k = np.asarray(keys).astype(np.int32, copy=False)
    v = np.asarray(vals).astype(np.int32, copy=False)
    n = k.size
    if n == 0:
        return empty_state(), info(0, 0, 0, 0, 0, {"cand": "C"})
    assert HAS_NUMBA
    gm = geom(g, p)
    shift, w = gm["shift"], gm["width"]
    bounds = _blocks(n, threads)
    t0 = time.perf_counter()
    part_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    if threads == 1:
        sums = np.zeros((p, w), dtype=np.int64)
        counts = np.zeros((p, w), dtype=np.int64)
        k_fshift_slice(k, v, sums, counts, shift, w, p)
        rep = 0
    else:
        sums3 = np.zeros((threads, p, w), dtype=np.int64)
        cnts3 = np.zeros((threads, p, w), dtype=np.int64)

        def _wb(ti):
            a, b = int(bounds[ti]), int(bounds[ti + 1])
            k_fshift_slice(k[a:b], v[a:b], sums3[ti], cnts3[ti],
                           shift, w, p)

        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(_wb, range(threads)))
        rep = int(sums3.nbytes + cnts3.nbytes)
        t1b = time.perf_counter()
        sums = sums3.sum(axis=0)
        counts = cnts3.sum(axis=0)
        agg_ms = (time.perf_counter() - t1) * 1000.0
        t2 = time.perf_counter()
        st = _compact_shift(sums, counts, w, g, p)
        merge_ms = (time.perf_counter() - t2) * 1000.0 + \
            (time.perf_counter() - t1b) * 0.0
        # split replica-sum (in agg window) honestly: re-time below
        traffic = int(k.nbytes + v.nbytes + 8 * p * w * 2 * threads
                      + 8 * st.ngroups * 2)
        return st, info(part_ms, agg_ms, merge_ms, rep, traffic,
                        {"cand": "C", "threads": threads,
                         "ws_bytes": int(8 * w * 2),
                         "partial_rows": 0, "route_ms": 0.0})
    agg_ms = (time.perf_counter() - t1) * 1000.0
    t2 = time.perf_counter()
    st = _compact_shift(sums, counts, w, g, p)
    merge_ms = (time.perf_counter() - t2) * 1000.0
    traffic = int(k.nbytes + v.nbytes + 8 * p * w * 2 + 8 * st.ngroups * 2)
    return st, info(part_ms, agg_ms, merge_ms,
                    int(sums.nbytes + counts.nbytes), traffic,
                    {"cand": "C", "threads": threads,
                     "ws_bytes": int(8 * w * 2),
                     "partial_rows": 0, "route_ms": 0.0})


def _assign_contig(p, t):
    base, rem = divmod(p, t)
    out, plo = [], 0
    for wi in range(t):
        npo = base + (1 if wi < rem else 0)
        out.append((plo, npo))
        plo += npo
    return out


def _assign_lpt(part_counts, t):
    order = np.argsort(-part_counts, kind="stable")
    loads = np.zeros(t, dtype=np.int64)
    own = [[] for _ in range(t)]
    for q in order:
        wi = int(np.argmin(loads))
        own[wi].append(int(q))
        loads[wi] += int(part_counts[q])
    # convert to (plo-list) segments; owned kernel needs contiguous rows,
    # so E stores per-worker partition LIST + per-part (1,W) slabs.
    return own, loads


def run_D(keys, vals, g, p=P_DEF, threads=1):
    # Contiguous ownership, full N scan per worker, no replication.
    k = np.asarray(keys).astype(np.int32, copy=False)
    v = np.asarray(vals).astype(np.int32, copy=False)
    n = k.size
    if n == 0:
        return empty_state(), info(0, 0, 0, 0, 0, {"cand": "D"})
    assert HAS_NUMBA
    gm = geom(g, p)
    shift, w = gm["shift"], gm["width"]
    assign = _assign_contig(p, threads)
    t0 = time.perf_counter()
    part_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    sums_list = [np.zeros((npo, w), dtype=np.int64) for _, npo in assign]
    cnts_list = [np.zeros((npo, w), dtype=np.int64) for _, npo in assign]
    plo_list = [plo for plo, _ in assign]

    def _wd(wi):
        plo, npo = assign[wi]
        if npo == 0:
            return
        k_fshift_owned(k, v, sums_list[wi], cnts_list[wi], shift, w, plo)

    if threads == 1:
        _wd(0)
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(_wd, range(threads)))
    agg_ms = (time.perf_counter() - t1) * 1000.0
    t2 = time.perf_counter()
    st = _compact_owned(sums_list, cnts_list, plo_list, w, g, p)
    merge_ms = (time.perf_counter() - t2) * 1000.0
    acc_b = int(sum(s.nbytes + c.nbytes for s, c in zip(sums_list, cnts_list)))
    traffic = int(k.nbytes * threads + v.nbytes * threads + acc_b
                  + 8 * st.ngroups * 2)
    return st, info(part_ms, agg_ms, merge_ms, acc_b, traffic,
                    {"cand": "D", "threads": threads,
                     "ws_bytes": int(8 * w * 2),
                     "partial_rows": 0, "route_ms": 0.0,
                     "reads_Nx": threads})


def run_E(keys, vals, g, p=P_DEF, threads=1):
    # LPT ownership on 100K sample (no full partition pass).
    k = np.asarray(keys).astype(np.int32, copy=False)
    v = np.asarray(vals).astype(np.int32, copy=False)
    n = k.size
    if n == 0:
        return empty_state(), info(0, 0, 0, 0, 0, {"cand": "E"})
    assert HAS_NUMBA
    gm = geom(g, p)
    shift, w = gm["shift"], gm["width"]
    t0 = time.perf_counter()
    ns = min(n, 100_000)
    step = max(1, n // ns)
    sample = np.ascontiguousarray(k[::step][:ns])
    hist = np.zeros(p, dtype=np.int64)
    k_pid_hist(sample, shift, p, hist)
    # scale sample hist to N for load reporting
    est = hist.astype(np.float64) * (n / max(1, sample.size))
    own, loads = _assign_lpt(np.ascontiguousarray(hist), threads)
    plan_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    # per-worker slabs + pos tables: single full-N scan per worker.
    slabs_list, cnts_list, pos_list, qorder = [], [], [], []
    for wi in range(threads):
        qs = sorted(own[wi])
        qorder.append(qs)
        pos = np.full(p, -1, dtype=np.int64)
        for r, q in enumerate(qs):
            pos[q] = np.int64(r)
        pos_list.append(np.ascontiguousarray(pos))
        slabs_list.append(np.zeros((len(qs), w), dtype=np.int64))
        cnts_list.append(np.zeros((len(qs), w), dtype=np.int64))

    def _we(wi):
        if len(qorder[wi]) == 0:
            return
        k_fshift_ownerpos(k, v, slabs_list[wi], cnts_list[wi],
                          shift, w, pos_list[wi])

    if threads == 1:
        _we(0)
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(_we, range(threads)))
    agg_ms = (time.perf_counter() - t1) * 1000.0
    t2 = time.perf_counter()
    uk, ss, cc = [], [], []
    for wi in range(threads):
        for r, q in enumerate(qorder[wi]):
            s, c = slabs_list[wi][r], cnts_list[wi][r]
            nz = np.flatnonzero(c > 0)
            if nz.size:
                uk.append((_key_of_shift(np.int64(q), nz.astype(np.int64),
                                         np.int64(w))).astype(np.int64))
                ss.append(s[nz])
                cc.append(c[nz])
    if not uk:
        st = empty_state()
    else:
        ukeys = np.concatenate(uk)
        o = np.argsort(ukeys, kind="stable")
        st = State(ukeys[o], np.concatenate(ss)[o], np.concatenate(cc)[o])
    merge_ms = (time.perf_counter() - t2) * 1000.0
    acc_b = int(sum(8 * w * 2 for wi in range(threads) for _ in own[wi]))
    traffic = int(k.nbytes * threads + v.nbytes * threads + acc_b
                  + 8 * st.ngroups * 2)
    imb = float((loads.max() - loads.mean()) / max(1, loads.mean()) * 100)
    return st, info(plan_ms, agg_ms, merge_ms, acc_b, traffic,
                    {"cand": "E", "threads": threads,
                     "ws_bytes": int(8 * w * 2), "partial_rows": 0,
                     "route_ms": plan_ms, "reads_Nx": threads,
                     "lpt_imb_pct": round(imb, 1),
                     "sample_rows": int(sample.size)})


def _run_F(keys, vals, g, p, threads, sub):
    k = np.asarray(keys).astype(np.int32, copy=False)
    v = np.asarray(vals).astype(np.int32, copy=False)
    n = k.size
    if n == 0:
        return empty_state(), info(0, 0, 0, 0, 0, {"cand": "F"})
    assert HAS_NUMBA
    gm = geom(g, p)
    shift, w = gm["shift"], gm["width"]
    sums = np.zeros((p, w), dtype=np.int64)
    counts = np.zeros((p, w), dtype=np.int64)
    locks = [threading.Lock() for _ in range(p)]
    bounds = _blocks(n, threads)
    t0 = time.perf_counter()
    part_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()

    def _wf(wi):
        a, b = int(bounds[wi]), int(bounds[wi + 1])
        s = a
        hist = np.empty(p, dtype=np.int64)
        perm = np.empty(sub, dtype=np.int64)
        offs = np.empty(p + 1, dtype=np.int64)
        while s < b:
            e = min(b, s + sub)
            m = e - s
            hist[:] = 0
            kk = k[s:e]
            vv = v[s:e]
            # pass 1: hist, pass 2: reorder fill, pass 3: contiguous apply
            k_sub_hist(kk, shift, p, hist)
            offs[0] = 0
            offs[1:] = np.cumsum(hist)
            cur = offs[:-1].copy()
            k_sub_fill(kk, shift, p, cur, perm[:m])
            for q in range(p):
                if hist[q]:
                    with locks[q]:
                        k_sub_apply_perm(kk, vv, perm[:m],
                                         int(offs[q]), int(offs[q + 1]),
                                         sums, counts, shift, w)
            s = e

    if threads == 1:
        _wf(0)
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(_wf, range(threads)))
    agg_ms = (time.perf_counter() - t1) * 1000.0
    t2 = time.perf_counter()
    st = _compact_shift(sums, counts, w, g, p)
    merge_ms = (time.perf_counter() - t2) * 1000.0
    traffic = int(k.nbytes + v.nbytes + 8 * p * w * 2 + 8 * st.ngroups * 2)
    return st, info(part_ms, agg_ms, merge_ms,
                    int(sums.nbytes + counts.nbytes), traffic, None)


def run_F(keys, vals, g, p=P_DEF, threads=1):
    st, inf = _run_F(keys, vals, g, p, threads, 8192)
    inf["cand"] = "F"
    inf["threads"] = threads
    inf["ws_bytes"] = int(geom(g, p)["width"] * 8 * 2)
    inf["partial_rows"] = 0
    inf["route_ms"] = 0.0
    inf["sub_block"] = 8192
    return st, inf


def run_Fbig(keys, vals, g, p=P_DEF, threads=1):
    st, inf = _run_F(keys, vals, g, p, threads, 65536)
    inf["cand"] = "Fbig"
    inf["threads"] = threads
    inf["ws_bytes"] = int(geom(g, p)["width"] * 8 * 2)
    inf["partial_rows"] = 0
    inf["route_ms"] = 0.0
    inf["sub_block"] = 65536
    return st, inf


def run_G(keys, vals, g, p=P_DEF, threads=1):
    # Radix-partitioned (DuckDB/DataFusion-style): count + scatter, then
    # partition-parallel flat agg over W ranges. Honest partition price.
    k = np.asarray(keys).astype(np.int32, copy=False)
    v = np.asarray(vals).astype(np.int32, copy=False)
    n = k.size
    if n == 0:
        return empty_state(), info(0, 0, 0, 0, 0, {"cand": "G"})
    assert HAS_NUMBA
    gm = geom(g, p)
    shift, w = gm["shift"], gm["width"]
    t0 = time.perf_counter()
    hist = np.zeros(p, dtype=np.int64)
    k_pid_hist(k, shift, p, hist)
    bounds = np.zeros(p + 1, dtype=np.int64)
    bounds[1:] = np.cumsum(hist)
    out_k = np.empty(n, dtype=np.int32)
    out_v = np.empty(n, dtype=np.int32)
    cur = bounds[:-1].copy()
    k_scatter_fused(k, v, shift, p, cur, out_k, out_v)
    part_ms = (time.perf_counter() - t0) * 1000.0
    # partition ranges in key space for shift geometry
    lo = np.array([min(q * w, g) for q in range(p)], dtype=np.int64)
    hi = np.array([min((q + 1) * w, g) for q in range(p)], dtype=np.int64)
    # assign partitions to workers contiguously
    base, rem = divmod(p, threads)
    owner, s = [], 0
    for wi in range(threads):
        npo = base + (1 if wi < rem else 0)
        owner.append(list(range(s, s + npo)))
        s += npo
    t1 = time.perf_counter()
    acc = {}
    for q in range(p):
        ww = int(hi[q] - lo[q])
        if ww > 0:
            acc[q] = (np.zeros(ww, dtype=np.int64),
                      np.zeros(ww, dtype=np.int64))

    def _wg(wi):
        for q in owner[wi]:
            if q not in acc:
                continue
            a, b = int(bounds[q]), int(bounds[q + 1])
            if b > a:
                k_flat_part(out_k[a:b], out_v[a:b], acc[q][0], acc[q][1],
                            int(lo[q]))

    if threads == 1:
        _wg(0)
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(_wg, range(threads)))
    agg_ms = (time.perf_counter() - t1) * 1000.0
    t2 = time.perf_counter()
    uk, ss, cc = [], [], []
    for q in range(p):
        if q not in acc:
            continue
        s2, c2 = acc[q]
        nz = np.flatnonzero(c2 > 0)
        if nz.size:
            uk.append((lo[q] + nz).astype(np.int64))
            ss.append(s2[nz])
            cc.append(c2[nz])
    st = State(np.concatenate(uk), np.concatenate(ss),
               np.concatenate(cc)) if uk else empty_state()
    merge_ms = (time.perf_counter() - t2) * 1000.0
    tmp_b = int(out_k.nbytes + out_v.nbytes
                + sum(s.nbytes + c.nbytes for s, c in acc.values()))
    traffic = int(k.nbytes + v.nbytes + out_k.nbytes + out_v.nbytes
                  + k.nbytes + v.nbytes + 8 * st.ngroups * 2)
    return st, info(part_ms, agg_ms, merge_ms, tmp_b, traffic,
                    {"cand": "G", "threads": threads,
                     "ws_bytes": int(8 * w * 2), "partial_rows": 0,
                     "route_ms": part_ms})


CANDS = {"A": run_A, "B": run_B, "C": run_C, "D": run_D, "E": run_E,
         "F": run_F, "Fbig": run_Fbig, "G": run_G}
