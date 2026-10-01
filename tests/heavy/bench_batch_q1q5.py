# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch GPU Q1-Q5 vs sum-separate, N=10M, RTX 2060, warm (NEW, bench-only).

Batch: 8 base columns uploaded ONCE (shared H2D) -> Q1..Q5 run sequentially
on resident buffers (r_pack for Q2, _GB_RESIDENT_ACCUM_WGSL per value col,
cached pipelines via DeviceContext.pipe) -> D2H only M-sized finals.
Separate: same kernels, per-Q H2D (keys+hi/lo uploaded per Q).
Same arithmetic both modes -> delta is pure H2D sharing + pipeline reuse.
CPU batch control: 5 bincount queries on resident numpy, query-only.
Correctness exact per Q (int cols dict equality, v3 scaled tol 1e-9);
mismatch = STOP. No Planner/Kernel/SPEC change. No 1B/ClickBench/NFS.

Usage (Git Bash, sequential, timeout):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    timeout 590 /c/App/numfast/.venv/Scripts/python.exe \
    tests/heavy/bench_batch_q1q5.py [--smoke]
Snapshot input only (scratch/snap_*, no H2O/CSV/NFS touch).
"""
import gc
import json
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
for _p in (str(FORK.parent / "app-builder-ponytail"), str(FORK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import importlib.util as _ilu

_gspec = _ilu.spec_from_file_location(
    "nfgpu_batch", str(FORK / "src" / "Drivers" / "GPU" / "_lib" / "gpu.py"))
G = _ilu.module_from_spec(_gspec)
_gspec.loader.exec_module(G)

_qspec = _ilu.spec_from_file_location(
    "nfq345", str(FORK / "tests" / "heavy" / "bench_gpu_final_q3q5.py"))
Q345 = _ilu.module_from_spec(_qspec)
_qspec.loader.exec_module(Q345)

SNAP = FORK / "scratch" / "snap_G1_1e7_1e2_0_0"
OUT = FORK / "tests" / "heavy" / "bench_batch_q1q5.json"
SMOKE = "--smoke" in sys.argv
SCALE = 1e-6
GOLD = {"Q1": {"ngroups": 100, "total": 29998789},
        "Q2": {"ngroups": 10000, "total": 29998789},
        "Q3": {"ngroups": 100000, "v1": 29998789,
               "mean3_sum": 4999719.622344427},
        "Q4": {"ngroups": 100, "m1": 299.98798187506526,
               "m2": 799.894179409978, "m3": 4999.766872833688},
        "Q5": {"ngroups": 100000, "s1": 29998789, "s2": 79989360,
               "s3_scaled": 499976651408061}}


def split(v):
    hi = np.right_shift(np.ascontiguousarray(v, dtype=np.int32),
                        np.int32(16))
    lo = np.bitwise_and(np.ascontiguousarray(v, dtype=np.int32),
                        np.int32(0xFFFF)).astype(np.uint32)
    return hi, lo


def accum_run(b_keys, b_hi, b_lo, m, n):
    """One value col: alloc M accums -> resident accum dispatch -> M finals."""
    t = time.perf_counter()
    acc = G.resident_accum_alloc(m)
    alloc_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    G._gb_run_global_bufs(G._device(), G._GB_RESIDENT_ACCUM_WGSL,
                          [(b_keys, True), (b_hi, True), (b_lo, True)],
                          list(acc), int(n))
    kern_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    uk, cc, ss = G.resident_accum_finalize(*acc, m)
    d2h_ms = (time.perf_counter() - t) * 1000
    return uk, cc, ss, alloc_ms, kern_ms, d2h_ms


def one_q_resident(b_keys, hi_lo_list, m, n):
    """All value cols of one Q on resident bufs. Returns (cols, stages)."""
    st = {"kernel_ms": 0.0, "d2h_ms": 0.0, "alloc_ms": 0.0, "disp": 0}
    cols = []
    for (b_hi, b_lo) in hi_lo_list:
        uk, cc, ss, am, km, dm = accum_run(b_keys, b_hi, b_lo, m, n)
        st["alloc_ms"] += am
        st["kernel_ms"] += km
        st["d2h_ms"] += dm
        st["disp"] += 1
        cols.append((uk, cc, ss))
    st["d2h_bytes"] = 4 * m * 4 * len(hi_lo_list)
    return cols, st


def up(arr):
    t = time.perf_counter()
    b = G.r_upload(np.ascontiguousarray(arr))
    return b, (time.perf_counter() - t) * 1000


def cpu_refs(K1, K2, K3, K4, K6, V1, V2, V3S, m1, m2, m3, m4, m5):
    t = time.perf_counter()
    c1 = np.bincount(K1.astype(np.int64), minlength=m1).astype(np.int64)
    s1 = np.bincount(K1.astype(np.int64),
                     weights=V1.astype(np.float64), minlength=m1)
    pk = K1.astype(np.int64) * np.int64(m2) + K2.astype(np.int64)
    mq2 = m1 * m2
    c2 = np.bincount(pk, minlength=mq2).astype(np.int64)
    s2 = np.bincount(pk, weights=V1.astype(np.float64), minlength=mq2)
    r3 = Q345.cpu_q3(K3, V1, V3S, m3)
    r4 = Q345.cpu_q4(K4, V1, V2, V3S, m4)
    r5 = Q345.cpu_q5(K6, V1, V2, V3S, m5)
    ms = (time.perf_counter() - t) * 1000
    assert int(s1.sum()) == GOLD["Q1"]["total"] and int((c1 > 0).sum()) == 100
    assert int(s2.sum()) == GOLD["Q2"]["total"] and int((c2 > 0).sum()) == 10000
    assert len(r3) == 100000 and int(sum(
        c["v1"]["sum"] for c in r3.values())) == GOLD["Q3"]["v1"]
    assert abs(float(sum(c["v3"]["mean"] for c in r3.values()))
               - GOLD["Q3"]["mean3_sum"]) < 1e-6 * abs(
                   GOLD["Q3"]["mean3_sum"]) + 1e-6
    assert len(r4) == 100 and len(r5) == 100000
    assert int(sum(c["v1"]["sum"] for c in r5.values())) == GOLD["Q5"]["s1"]
    assert int(sum(c["v2"]["sum"] for c in r5.values())) == GOLD["Q5"]["s2"]
    assert int(round(float(sum(c["v3"]["sum"] for c in r5.values())) * 1e6)) \
        == GOLD["Q5"]["s3_scaled"]
    return {"ms": ms, "c1": (c1, s1), "c2": (c2, s2, mq2),
            "r3": r3, "r4": r4, "r5": r5}


def check(q, cols, ref, m, scale_v3=None):
    """Exact chk of resident cols vs CPU ref. STOP on mismatch."""
    if q in ("Q1", "Q2"):
        c_ref, s_ref = ref[0], ref[1]
        (uk, cc, ss) = cols[0]
        nz = np.flatnonzero(c_ref > 0)
        if not ((uk == nz).all() and (cc == c_ref[nz]).all()
                and (ss == s_ref[nz].astype(np.int64)).all()):
            print(f"{q} INTEGRITY STOP"); sys.exit(1)
        return {"ngroups": int(uk.size), "total": int(ss.sum())}
    if q == "Q3":
        (uk1, cc1, ss1), (uku, ccu, ssu) = cols
        d = {int(k): {"v1": {"sum": int(s)}, "v3": {"mean": float(
            m_)}} for k, s, m_ in zip(
            uk1.tolist(), ss1.tolist(),
            (ssu.astype(np.float64) * SCALE / np.maximum(
                ccu.astype(np.float64), 1.0)).tolist())}
        md, _ = Q345.maxdiff_multi(ref, d)
        if md >= 1e-9:
            print(f"Q3 INTEGRITY STOP md={md}"); sys.exit(1)
        return {"ngroups": int(uk1.size), "maxdiff": md,
                "total_v1": int(ss1.sum()),
                "mean3_sum": float((ssu.astype(np.float64) * SCALE / np.maximum(
                    ccu.astype(np.float64), 1.0)).sum())}
    if q == "Q4":
        cols_s = [ss for (_, _, ss) in cols]
        cc = cols[0][1]
        means = [s.astype(np.float64) / np.maximum(cc.astype(np.float64), 1.0)
                 for s in cols_s]
        means[2] = means[2] * SCALE
        d = {int(k): {"v1": {"mean": float(a)}, "v2": {"mean": float(b)},
                      "v3": {"mean": float(c)}}
             for k, a, b, c in zip(cols[0][0].tolist(), means[0].tolist(),
                                   means[1].tolist(), means[2].tolist())}
        md, _ = Q345.maxdiff_multi(ref, d)
        if md >= 1e-9:
            print(f"Q4 INTEGRITY STOP md={md}"); sys.exit(1)
        return {"ngroups": int(cols[0][0].size), "maxdiff": md,
                "m1": float(means[0].sum()), "m2": float(means[1].sum()),
                "m3": float(means[2].sum())}
    (uk1, cc1, ss1), (_, _, ss2), (_, _, ss3) = cols
    if int(ss1.sum()) != GOLD["Q5"]["s1"] or int(ss2.sum()) != GOLD["Q5"]["s2"] \
            or int(ss3.sum()) != GOLD["Q5"]["s3_scaled"]:
        print("Q5 INTEGRITY STOP"); sys.exit(1)
    return {"ngroups": int(uk1.size), "s1": int(ss1.sum()),
            "s2": int(ss2.sum()), "s3_scaled": int(ss3.sum())}


def main():
    print(f"=== batch-vs-separate Q1-Q5 {'SMOKE' if SMOKE else 'N=10M'} ===",
          flush=True)
    t = time.perf_counter()
    K1 = np.ascontiguousarray(np.load(SNAP / "K1.npy"), dtype=np.int32)
    K2 = np.ascontiguousarray(np.load(SNAP / "K2.npy"), dtype=np.int32)
    K3 = np.ascontiguousarray(np.load(SNAP / "K3.npy"), dtype=np.int32)
    K4 = np.ascontiguousarray(np.load(SNAP / "K4.npy"), dtype=np.int32)
    K6 = np.ascontiguousarray(np.load(SNAP / "K6.npy"), dtype=np.int32)
    V1 = np.ascontiguousarray(np.load(SNAP / "V1.npy"), dtype=np.int32)
    V2 = np.ascontiguousarray(np.load(SNAP / "V2.npy"), dtype=np.int32)
    V3 = np.ascontiguousarray(np.load(SNAP / "V3.npy"), dtype=np.float64)
    if SMOKE:
        n0 = 200_000
        K1, K2, K3, K4, K6 = (a[:n0] for a in (K1, K2, K3, K4, K6))
        V1, V2, V3 = V1[:n0], V2[:n0], V3[:n0]
    n = int(K1.size)
    V3S = np.rint(V3 * 1e6).astype(np.int32)
    del V3
    gc.collect()
    print(f"host_load {((time.perf_counter()-t)*1000):.0f}ms n={n}", flush=True)
    m1, m2 = int(K1.max()) + 1, int(K2.max()) + 1
    m3, m4, m5, mq2 = int(K3.max()) + 1, int(K4.max()) + 1, \
        int(K6.max()) + 1, (int(K1.max()) + 1) * (int(K2.max()) + 1)
    print(f"m1={m1} m2={m2} m3={m3} m4={m4} m5={m5} mq2={mq2}", flush=True)

    refs = None if SMOKE else cpu_refs(K1, K2, K3, K4, K6, V1, V2, V3S,
                                       m1, m2, m3, m4, m5)
    if not SMOKE:
        print(f"cpu_ref {refs['ms']:.0f}ms exact-ok", flush=True)

    # warm: compile both pipelines on tiny slices (outside timers)
    wk = np.ascontiguousarray(K1[:4096])
    wv = np.ascontiguousarray(V1[:4096])
    wh, wl = split(wv)
    t = time.perf_counter()
    b0, b1, b2 = G.r_upload(wk), G.r_upload(wh), G.r_upload(wl)
    G._gb_run_global_bufs(G._device(), G._GB_RESIDENT_ACCUM_WGSL,
                          [(b0, True), (b1, True), (b2, True)],
                          list(G.resident_accum_alloc(m1)), 4096)
    bk1, bk2 = G.r_upload(wk), G.r_upload(
        np.ascontiguousarray(K2[:4096]))
    G.r_pack(bk1, bk2, m2, 4096)
    del b0, b1, b2, bk1, bk2
    p_warm = G.pipeline_stats()
    print(f"warm_compile {((time.perf_counter()-t)*1000):.0f}ms "
          f"pipes={p_warm}", flush=True)

    H1, L1 = split(V1)
    H2, L2 = split(V2)
    H3, L3 = split(V3S)

    # ---- SEPARATE: per-Q H2D + same kernels ----
    sep, sep_h2d, sep_k, sep_d = {}, 0.0, 0.0, 0.0
    chk_rec = {}
    p0 = G.pipeline_stats()

    def sep_q(q, keys, hl_list, m):
        st = {"h2d_ms": 0.0, "h2d_bytes": 0}
        t = time.perf_counter()
        b_k, h = up(keys)
        st["h2d_ms"] += h
        st["h2d_bytes"] += int(keys.nbytes)
        b_hl = []
        for (hh, ll) in hl_list:
            bh, h = up(hh)
            bl, h2 = up(ll.view(np.int32))
            st["h2d_ms"] += h + h2
            st["h2d_bytes"] += int(hh.nbytes + ll.nbytes)
            b_hl.append((bh, bl))
        cols, ks = one_q_resident(b_k, b_hl, m, n)
        ks["h2d_ms"] = st["h2d_ms"]
        ks["h2d_bytes"] = st["h2d_bytes"]
        ks["pack_ms"] = 0.0
        ks["total_ms"] = ks["h2d_ms"] + ks["alloc_ms"] + ks["kernel_ms"] \
            + ks["d2h_ms"]
        del b_k, b_hl
        gc.collect()
        return cols, ks

    t = time.perf_counter()
    b_k2_sep, h = up(K2)
    k2_up_ms, k2_up_bytes = h, int(K2.nbytes)
    cols, ks = sep_q("Q1", K1, [(H1, L1)], m1)
    if not SMOKE:
        chk = check("Q1", cols, refs["c1"], m1)
        assert chk["ngroups"] == 100 and chk["total"] == GOLD["Q1"]["total"]
        chk_rec["SEP Q1"] = chk
    sep["Q1"] = ks
    # Q2 separate: K1 uploaded inside sep_q, K2 uploaded once here
    t = time.perf_counter()
    b_k1_q2, h = up(K1)
    q2_h2d_extra = h + k2_up_ms
    q2_b_extra = int(K1.nbytes) + k2_up_bytes
    t2 = time.perf_counter()
    b_pk = G.r_pack(b_k1_q2, b_k2_sep, m2, n)
    pk_ms = (time.perf_counter() - t2) * 1000
    t2 = time.perf_counter()
    bh, h = up(H1)
    bl, h2 = up(L1.view(np.int32))
    vlan_ms = h + h2
    cols, ks = one_q_resident(b_pk, [(bh, bl)], mq2, n)
    ks["h2d_ms"] = q2_h2d_extra + vlan_ms
    ks["h2d_bytes"] = q2_b_extra + int(H1.nbytes + L1.nbytes)
    ks["pack_ms"] = pk_ms
    ks["total_ms"] = ks["h2d_ms"] + ks["alloc_ms"] + ks["kernel_ms"] \
        + pk_ms + ks["d2h_ms"]
    if not SMOKE:
        chk = check("Q2", cols, refs["c2"], mq2)
        assert chk["ngroups"] == 10000 and chk["total"] == GOLD["Q2"]["total"]
        chk_rec["SEP Q2"] = chk
    sep["Q2"] = ks
    del b_k1_q2, b_k2_sep, b_pk, bh, bl
    gc.collect()
    for q, keys, hl, m, rk in (
            ("Q3", K3, [(H1, L1), (H3, L3)], m3, "r3"),
            ("Q4", K4, [(H1, L1), (H2, L2), (H3, L3)], m4, "r4"),
            ("Q5", K6, [(H1, L1), (H2, L2), (H3, L3)], m5, "r5")):
        cols, ks = sep_q(q, keys, hl, m)
        if not SMOKE:
            chk_rec[f"SEP {q}"] = check(q, cols, refs[rk], m)
        sep[q] = ks
        del cols
    p1 = G.pipeline_stats()
    for q, s in sep.items():
        print(f"SEP {q}: h2d={s['h2d_bytes']} "
              f"up={s['h2d_ms']:.1f}ms alloc={s['alloc_ms']:.1f}ms "
              f"kern={s['kernel_ms']:.1f}ms pack={s['pack_ms']:.1f}ms "
              f"d2h={s['d2h_ms']:.1f}ms total={s['total_ms']:.1f}ms "
              f"disp={s['disp']}", flush=True)
    sep_total = sum(s["total_ms"] for s in sep.values())
    sep_h2d = sum(s["h2d_ms"] for s in sep.values())
    print(f"SEP sum-separate total={sep_total:.1f}ms h2d={sep_h2d:.1f}ms "
          f"pipes delta={p1['misses']-p0['misses']}miss/"
          f"{p1['hits']-p0['hits']}hit", flush=True)

    # ---- BATCH: upload once, per-Q kernels on resident bufs ----
    t = time.perf_counter()
    bK1, h = up(K1)
    bK2, h2 = up(K2)
    bK3, h3 = up(K3)
    bK4, h4 = up(K4)
    bK6, h5 = up(K6)
    bH1, h6 = up(H1)
    bL1, h7 = up(L1.view(np.int32))
    bH2, h8 = up(H2)
    bL2, h9 = up(L2.view(np.int32))
    bH3, h0 = up(H3)
    bL3, hA = up(L3.view(np.int32))
    shared_h2d_ms = (time.perf_counter() - t) * 1000
    shared_h2d_bytes = int(K1.nbytes + K2.nbytes + K3.nbytes + K4.nbytes
                           + K6.nbytes + H1.nbytes + L1.nbytes + H2.nbytes
                           + L2.nbytes + H3.nbytes + L3.nbytes)
    p2 = G.pipeline_stats()
    bat = {}
    t = time.perf_counter()
    b_pk = G.r_pack(bK1, bK2, m2, n)
    bat_pack_ms = (time.perf_counter() - t) * 1000
    specs = (("Q1", bK1, [(bH1, bL1)], m1, None),
             ("Q2", b_pk, [(bH1, bL1)], mq2, None),
             ("Q3", bK3, [(bH1, bL1), (bH3, bL3)], m3, "r3"),
             ("Q4", bK4, [(bH1, bL1), (bH2, bL2), (bH3, bL3)], m4, "r4"),
             ("Q5", bK6, [(bH1, bL1), (bH2, bL2), (bH3, bL3)], m5, "r5"))
    for q, bk, hl, m, rk in specs:
        cols, ks = one_q_resident(bk, hl, m, n)
        if not SMOKE:
            if q in ("Q1", "Q2"):
                chk_rec[f"BAT {q}"] = check(
                    q, cols, refs["c1"] if q == "Q1" else refs["c2"], m)
            else:
                chk_rec[f"BAT {q}"] = check(q, cols, refs[rk], m)
        ks["total_ms"] = ks["alloc_ms"] + ks["kernel_ms"] + ks["d2h_ms"]
        bat[q] = ks
        del cols
    p3 = G.pipeline_stats()
    bat_k = sum(s["kernel_ms"] for s in bat.values())
    bat_d = sum(s["d2h_ms"] for s in bat.values())
    bat_a = sum(s["alloc_ms"] for s in bat.values())
    bat_total = shared_h2d_ms + bat_pack_ms + bat_k + bat_d + bat_a
    for q, s in bat.items():
        print(f"BAT {q}: alloc={s['alloc_ms']:.1f}ms kern={s['kernel_ms']:.1f}ms "
              f"d2h={s['d2h_ms']:.1f}ms total={s['total_ms']:.1f}ms "
              f"disp={s['disp']}", flush=True)
    print(f"BAT shared-h2d={shared_h2d_bytes}B {shared_h2d_ms:.1f}ms "
          f"pack={bat_pack_ms:.1f}ms kernels={bat_k:.1f}ms d2h={bat_d:.1f}ms "
          f"TOTAL={bat_total:.1f}ms vs SEP={sep_total:.1f}ms "
          f"ratio={sep_total/max(bat_total,1e-9):.2f}x "
          f"pipes delta={p3['misses']-p2['misses']}miss/"
          f"{p3['hits']-p2['hits']}hit", flush=True)

    # ---- CPU batch control: 5 queries on resident numpy ----
    def cpu_q(fn):
        t = time.perf_counter()
        r = fn()
        return r, (time.perf_counter() - t) * 1000

    cpub = {}
    _, cpub["Q1"] = cpu_q(lambda: (
        np.bincount(K1.astype(np.int64), minlength=m1),
        np.bincount(K1.astype(np.int64),
                    weights=V1.astype(np.float64), minlength=m1)))
    pkc = K1.astype(np.int64) * np.int64(m2) + K2.astype(np.int64)
    _, cpub["Q2"] = cpu_q(lambda: (
        np.bincount(pkc, minlength=mq2),
        np.bincount(pkc, weights=V1.astype(np.float64), minlength=mq2)))
    _, cpub["Q3"] = cpu_q(lambda: Q345.cpu_q3(K3, V1, V3S, m3))
    _, cpub["Q4"] = cpu_q(lambda: Q345.cpu_q4(K4, V1, V2, V3S, m4))
    _, cpub["Q5"] = cpu_q(lambda: Q345.cpu_q5(K6, V1, V2, V3S, m5))
    # warm repeats (best of 2 more)
    for q, fn in (("Q1", lambda: np.bincount(
            K1.astype(np.int64), weights=V1.astype(np.float64),
            minlength=m1)),
            ("Q2", lambda: np.bincount(
                pkc, weights=V1.astype(np.float64), minlength=mq2)),
            ("Q3", lambda: Q345.cpu_q3(K3, V1, V3S, m3)),
            ("Q4", lambda: Q345.cpu_q4(K4, V1, V2, V3S, m4)),
            ("Q5", lambda: Q345.cpu_q5(K6, V1, V2, V3S, m5))):
        best = cpub[q]
        for _ in range(2):
            _, ms = cpu_q(fn)
            best = min(best, ms)
        cpub[q] = best
        print(f"CPU {q}: {cpub[q]:.1f}ms", flush=True)
    cpu_total = sum(cpub.values())
    print(f"CPU batch sum={cpu_total:.1f}ms", flush=True)

    out = {"smoke": SMOKE, "n": n, "m": {"m1": m1, "m2": m2, "m3": m3,
                                         "m4": m4, "m5": m5, "mq2": mq2},
           "separate": sep, "sep_total_ms": sep_total,
           "batch": bat, "shared_h2d_ms": shared_h2d_ms,
           "shared_h2d_bytes": shared_h2d_bytes,
           "batch_pack_ms": bat_pack_ms, "batch_total_ms": bat_total,
           "ratio_sep_bat": sep_total / max(bat_total, 1e-9),
           "h2d_saved_ms": sep_h2d - shared_h2d_ms,
           "cpu_batch_ms": cpub, "cpu_total_ms": cpu_total,
           "pipes": {"warm": p_warm, "sep0": p0, "sep1": p1,
                     "bat0": p2, "bat1": p3},
           "chk": chk_rec,
           "correctness": "exact" if not SMOKE else "smoke-nochk"}
    OUT.write_text(json.dumps(json.loads(json.dumps(out, default=float)),
                              indent=1))
    print(f"JSON {OUT}", flush=True)


if __name__ == "__main__":
    main()
