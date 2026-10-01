# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NFS 1B GPU batch DAG: loader(CPU) -> block -> H2D once -> GPU Q1..Q5 on
resident block (resident-accumulate c20254c + GPU-final, called as-is) ->
D2H M-sized state -> host merge. No per-query PCIe. No kernel/format/Core/
Planner/calibration/ClickBench/raw change. Bench-only (precedent:
bench_stream_1B_batch_overlap.py).

Usage (Git Bash, sequential, timeout 550):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    timeout 550 /c/App/numfast/.venv/Scripts/python.exe \
    tests/heavy/bench_stream_1B_batch_gpu.py [NBLOCKS] [QSET]
  NBLOCKS default 5 (50M subset, marked). QSET: all|q1q2|q3q5 default all.
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

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 5
QSET = sys.argv[2] if len(sys.argv) > 2 else "all"

import importlib.util as _ilu
_gspec = _ilu.spec_from_file_location(
    "nfgpu_batch", str(FORK / "src" / "Drivers" / "GPU" / "_lib" / "gpu.py"))
G = _ilu.module_from_spec(_gspec)
_gspec.loader.exec_module(G)

F1B = FORK / "scratch" / "nfs_stream" / "G1_1e9.stream.nfs"
OUT = FORK / "tests" / "heavy" / "bench_stream_1B_batch_gpu.json"


def split(v):
    hi = np.right_shift(np.ascontiguousarray(v, dtype=np.int32),
                        np.int32(16))
    lo = np.bitwise_and(np.ascontiguousarray(v, dtype=np.int32),
                        np.int32(0xFFFF)).astype(np.uint32)
    return hi, lo


def accum_col(b_keys, b_hi, b_lo, m, n, t):
    """One value col on resident bufs. Stages into t dict."""
    s = time.perf_counter()
    acc = G.resident_accum_alloc(m)
    t["alloc"] += (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    G._gb_run_global_bufs(G._device(), G._GB_RESIDENT_ACCUM_WGSL,
                          [(b_keys, True), (b_hi, True), (b_lo, True)],
                          list(acc), int(n))
    t["kern"] += (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    uk, cc, ss = G.resident_accum_finalize(*acc, m)
    t["d2h"] += (time.perf_counter() - s) * 1000
    t["d2h_bytes"] += 16 * m
    del acc
    return uk, cc, ss


def main():
    from builder import MAIN
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    h = a["nfs_stream_open"](str(F1B), force_lazy=True)
    idx = h["index"]
    nblocks = min(NB, len(idx))
    M1 = int(max(m["maxs"]["k1"] for m in idx)) + 1
    M2k = int(max(m["maxs"]["k2"] for m in idx)) + 1
    MQ2, M3 = M1 * M2k, int(max(m["maxs"]["k3"] for m in idx)) + 1
    M4, M5 = int(max(m["maxs"]["id4"] for m in idx)) + 1, \
        int(max(m["maxs"]["id6"] for m in idx)) + 1
    print(f"blocks={nblocks}/{len(idx)} M1={M1} MQ2={MQ2} M3={M3} "
          f"M4={M4} M5={M5} QSET={QSET}", flush=True)
    # warm tiny (compile outside timers)
    wk = np.ascontiguousarray([1, 2, 3, 1], dtype=np.int32)
    wv = np.ascontiguousarray([10, 20, 30, 40], dtype=np.int32)
    wh, wl = split(wv)
    bk, bh, bl = G.r_upload(wk), G.r_upload(wh), G.r_upload(wl.view(np.int32))
    G._gb_run_global_bufs(G._device(), G._GB_RESIDENT_ACCUM_WGSL,
                          [(bk, True), (bh, True), (bl, True)],
                          list(G.resident_accum_alloc(4)), 4)
    del bk, bh, bl
    print(f"warm ok pipes={G.pipeline_stats()}", flush=True)

    do12 = QSET in ("all", "q1q2")
    do35 = QSET in ("all", "q3q5")
    # host dense merge accums (int64 exact; v3 as int64 ticks)
    g1 = np.zeros(M1, np.int64) if do12 else None
    g2 = np.zeros(MQ2, np.int64) if do12 else None
    g3a = np.zeros(M3, np.int64) if do35 else None
    g3c = np.zeros(M3, np.int64) if do35 else None
    g3t = np.zeros(M3, np.int64) if do35 else None
    g4a = np.zeros(M4, np.int64)
    g4b = np.zeros(M4, np.int64)
    g4t = np.zeros(M4, np.int64)
    g4c = np.zeros(M4, np.int64)
    g5a = np.zeros(M5, np.int64) if do35 else None
    g5b = np.zeros(M5, np.int64) if do35 else None
    g5t = np.zeros(M5, np.int64) if do35 else None
    # cpu refs on same subset (bincount exact)
    c1 = np.zeros(M1, np.int64) if do12 else None
    c1c = np.zeros(M1, np.float64) if do12 else None
    c2 = np.zeros(MQ2, np.float64) if do12 else None
    c3a = np.zeros(M3, np.float64) if do35 else None
    c3c = np.zeros(M3, np.int64) if do35 else None
    c3t = np.zeros(M3, np.float64) if do35 else None
    c5a = np.zeros(M5, np.float64) if do35 else None
    c5b = np.zeros(M5, np.float64) if do35 else None
    c5t = np.zeros(M5, np.float64) if do35 else None

    T = dict(load=0.0, prep=0.0, h2d=0.0, alloc=0.0, kern=0.0, d2h=0.0,
             merge=0.0, cpu=0.0)
    d2h_bytes = [0]
    t_wall = time.perf_counter()
    nrows = 0
    for bi in range(nblocks):
        s = time.perf_counter()
        b = a["nfs_stream_read_block"](h, bi)
        T["load"] += (time.perf_counter() - s) * 1000
        s = time.perf_counter()
        K1 = np.ascontiguousarray(b["k1"], dtype=np.int32)
        K2 = np.ascontiguousarray(b["k2"], dtype=np.int32)
        K3 = np.ascontiguousarray(b["k3"], dtype=np.int32)
        K4 = np.ascontiguousarray(b["id4"], dtype=np.int32)
        K6 = np.ascontiguousarray(b["id6"], dtype=np.int32)
        V1 = np.ascontiguousarray(b["v1"], dtype=np.int32)
        V2 = np.ascontiguousarray(b["v2"], dtype=np.int32)
        V3 = np.ascontiguousarray(b["v3"], dtype=np.float64)
        V3S = np.rint(V3 * 1e6).astype(np.int32)
        del V3
        H1, L1 = split(V1)
        H2, L2 = split(V2)
        H3, L3 = split(V3S)
        n = int(K1.size)
        nrows += n
        T["prep"] += (time.perf_counter() - s) * 1000
        # H2D once: all resident inputs
        s = time.perf_counter()
        bK1 = G.r_upload(K1) if do12 else None
        bK2 = G.r_upload(K2) if do12 else None
        bK3 = G.r_upload(K3) if do35 else None
        bK4 = G.r_upload(K4)
        bK6 = G.r_upload(K6) if do35 else None
        bH1, bL1 = G.r_upload(H1), G.r_upload(L1.view(np.int32))
        bH2, bL2 = G.r_upload(H2), G.r_upload(L2.view(np.int32))
        bH3, bL3 = (G.r_upload(H3), G.r_upload(L3.view(np.int32))) \
            if do35 else (None, None)
        T["h2d"] += (time.perf_counter() - s) * 1000
        tt = {"alloc": 0.0, "kern": 0.0, "d2h": 0.0, "d2h_bytes": 0}
        s = time.perf_counter()
        if do12:
            uk, cc, ss = accum_col(bK1, bH1, bL1, M1, n, tt)
            g1[uk] += ss
            s2 = time.perf_counter()
            bpk = G.r_pack(bK1, bK2, M2k, n)
            tt["kern"] += (time.perf_counter() - s2) * 1000
            uk, cc, ss = accum_col(bpk, bH1, bL1, MQ2, n, tt)
            g2[uk] += ss
            del bpk
        # Q4 always (M=101 tiny, shows DAG on resident block)
        uk, cc, ss = accum_col(bK4, bH1, bL1, M4, n, tt)
        g4a[uk] += ss
        g4c[uk] += cc
        uk, cc, ss = accum_col(bK4, bH2, bL2, M4, n, tt)
        g4b[uk] += ss
        if do35:
            uk, cc, ss = accum_col(bK4, bH3, bL3, M4, n, tt)
            g4t[uk] += ss
        if do35:
            uk, cc, ss = accum_col(bK3, bH1, bL1, M3, n, tt)
            g3a[uk] += ss
            g3c[uk] += cc
            uk, cc, ss = accum_col(bK3, bH3, bL3, M3, n, tt)
            g3t[uk] += ss
            uk, cc, ss = accum_col(bK6, bH1, bL1, M5, n, tt)
            g5a[uk] += ss
            uk, cc, ss = accum_col(bK6, bH2, bL2, M5, n, tt)
            g5b[uk] += ss
            uk, cc, ss = accum_col(bK6, bH3, bL3, M5, n, tt)
            g5t[uk] += ss
        T["merge"] += (time.perf_counter() - s) * 1000 - (
            tt["alloc"] + tt["kern"] + tt["d2h"])
        T["alloc"] += tt["alloc"]
        T["kern"] += tt["kern"]
        T["d2h"] += tt["d2h"]
        d2h_bytes[0] += tt["d2h_bytes"]
        del bK1, bK2, bK3, bK4, bK6, bH1, bL1, bH2, bL2, bH3, bL3
        # CPU refs same subset (exact bincount, host-side)
        s = time.perf_counter()
        if do12:
            c1 += np.bincount(K1.astype(np.int64), minlength=M1)
            c1c += np.bincount(K1.astype(np.int64),
                              weights=V1.astype(np.float64), minlength=M1)
            pk = K1.astype(np.int64) * np.int64(M2k) + K2.astype(np.int64)
            c2 += np.bincount(pk, weights=V1.astype(np.float64),
                              minlength=MQ2)
        if do35:
            c3a += np.bincount(K3.astype(np.int64),
                              weights=V1.astype(np.float64), minlength=M3)
            c3c += np.bincount(K3.astype(np.int64), minlength=M3)
            c3t += np.bincount(K3.astype(np.int64),
                              weights=V3S.astype(np.float64), minlength=M3)
            c5a += np.bincount(K6.astype(np.int64),
                              weights=V1.astype(np.float64), minlength=M5)
            c5b += np.bincount(K6.astype(np.int64),
                              weights=V2.astype(np.float64), minlength=M5)
            c5t += np.bincount(K6.astype(np.int64),
                              weights=V3S.astype(np.float64), minlength=M5)
        T["cpu"] += (time.perf_counter() - s) * 1000
        del b, K1, K2, K3, K4, K6, V1, V2, V3S, H1, L1, H2, L2, H3, L3
        gc.collect()
        print(f"block {bi+1}/{nblocks} load={T['load']:.0f} h2d={T['h2d']:.0f} "
              f"kern={T['kern']:.0f} d2h={T['d2h']:.0f}", flush=True)
    wall = (time.perf_counter() - t_wall) * 1000
    # correctness vs CPU subset (bit-bit integer; scaled ticks exact)
    ok = {}
    if do12:
        ok["Q1"] = bool((g1 == c1c.astype(np.int64)).all())
        ok["Q2"] = bool((g2 == c2.astype(np.int64)).all())
    if do35:
        ok["Q3v1"] = bool((g3a == c3a.astype(np.int64)).all())
        ok["Q3c"] = bool((g3c == c3c.astype(np.int64)).all())
        ok["Q3t"] = bool((g3t == c3t.astype(np.int64)).all())
        ok["Q5v1"] = bool((g5a == c5a.astype(np.int64)).all())
        ok["Q5v2"] = bool((g5b == c5b.astype(np.int64)).all())
        ok["Q5t"] = bool((g5t == c5t.astype(np.int64)).all())
        # f64 scaled tolerance: mean from ticks vs float ref
        m3 = g3t.astype(np.float64) * 1e-6 / np.maximum(g3c.astype(
            np.float64), 1.0)
        m3r = c3t.astype(np.float64) * 1e-6 / np.maximum(c3c.astype(
            np.float64), 1.0)
        ok["Q3mean_tol"] = bool(float(np.abs(m3 - m3r).max()) < 1e-9)
    allok = all(ok.values())
    print(f"correctness {ok} ALL={allok}", flush=True)
    assert allok, ok
    gpu_compute = T["prep"] + T["h2d"] + T["alloc"] + T["kern"] + T["d2h"]
    out = {"nblocks": nblocks, "nrows": nrows, "qset": QSET,
           "M": {"M1": M1, "MQ2": MQ2, "M3": M3, "M4": M4, "M5": M5},
           "load_ms": T["load"], "prep_ms": T["prep"], "h2d_ms": T["h2d"],
           "gpu_kern_ms": T["kern"], "gpu_alloc_ms": T["alloc"],
           "d2h_ms": T["d2h"], "d2h_bytes": d2h_bytes[0],
           "merge_ms": T["merge"], "wall_ms": wall,
           "cpu_ref_ms": T["cpu"], "correctness": ok,
           "note": f"subset {nrows/1e6:.0f}M of 1B; x{100/nblocks:.0f} "
                   f"extrapolation marked"}
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float), flush=True)
    print(f"JSON {OUT}", flush=True)


if __name__ == "__main__":
    main()
