# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research-only control: A (row-parallel dense) vs D (hash-ownership dense).

Real H2O 10M via builder resident_prepare (pattern codes id1/id2, composite
pack Q2, scaled v3) — no synthetic keys. One Q per run, T=8/12/16.
No prod change. No push. Seed 42. CPU only, no GPU.

Usage (Git Bash, fork-first):
  SYS=/c/Users/Mikech/AppData/Local/Programs/Python/Python314/python.exe
  FORK=/c/App/numfast/numfast-ponytail
  PYTHONPATH=$FORK:$FORK/../app-builder-ponytail:$FORK/tests/research/mtgroup:$FORK/tests/research/singlepass
  timeout 600 $SYS tests/research/ad_control/bench_ad.py Q1 [8,12,16]
"""
import gc
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

FORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FORK.parent / "app-builder-ponytail"))
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / "tests" / "research" / "singlepass"))
sys.path.insert(0, str(FORK / "tests" / "research" / "mtgroup"))

import numpy as np

import kernels_mt as K
from kernels_sp import dict_fast_multi, dict_fast_single, fused_singlepass

SEED = 42
CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
NGOLD = {"Q1": 100, "Q2": 10000, "Q3": 100000, "Q4": 100, "Q5": 100000}
CHK_V1 = 29998789
POOLS = {}

try:
    import psutil
    _PROC = psutil.Process()
except Exception:
    _PROC = None


def pool(t):
    if t not in POOLS:
        POOLS[t] = ThreadPoolExecutor(max_workers=t)
    return POOLS[t]


def rss_gb():
    return _PROC.memory_info().rss / 1e9 if _PROC else 0.0


def run_agg(tasks, t):
    ex = pool(t)
    t0 = time.perf_counter()

    def wrap(fn):
        ts = time.perf_counter()
        b = fn()
        te = time.perf_counter()
        return (ts - t0) * 1000, (te - ts) * 1000, b

    futs = [ex.submit(wrap, fn) for fn in tasks]
    res = [f.result() for f in futs]
    return (time.perf_counter() - t0) * 1000, [r[0] for r in res], [r[1] for r in res]


def load_case(q):
    """Real Schema/NFS encoding only. Returns (keys, vcols, m, sig, pack_ms, meta)."""
    from builder import MAIN
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    import pandas as pd
    import pyarrow as pa
    t0 = time.perf_counter()
    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"])
    csv_ms = (time.perf_counter() - t0) * 1000
    nrows = len(df)
    assert nrows == 10_000_000, nrows
    s = {c: pa.array(df[c].to_numpy(), type=pa.string()) for c in ("id1", "id2", "id3")}
    ints = {c: df[c].to_numpy().astype(np.int32) for c in ("id4", "id6")}
    vals = {"v1": df["v1"].to_numpy().astype(np.int32),
            "v2": df["v2"].to_numpy().astype(np.int32),
            "v3": df["v3"].to_numpy().astype(np.float64)}
    del df
    gc.collect()
    t0 = time.perf_counter()
    res = {}
    for name, col in (("id1", {"values": s["id1"], "prefix": "id"}),
                      ("id2", {"values": s["id2"], "prefix": "id"}),
                      ("id3", {"values": s["id3"], "prefix": "id"})):
        res[name] = a["resident_prepare"]({name: col})[name]
    c = a["resident_prepare"]({
        "id4": {"values": ints["id4"], "dtype": "int32"},
        "id6": {"values": ints["id6"], "dtype": "int32"},
        "v1": {"values": vals["v1"], "dtype": "int32"},
        "v2": {"values": vals["v2"], "dtype": "int32"},
        "v3": {"values": vals["v3"], "dtype": "float64"}})
    res.update(c)
    prep_ms = (time.perf_counter() - t0) * 1000
    K1, K2, K3 = res["id1"]["codes"], res["id2"]["codes"], res["id3"]["codes"]
    K4, K6 = res["id4"]["codes"], res["id6"]["codes"]
    V1, V2, V3 = res["v1"]["codes"], res["v2"]["codes"], res["v3"]["codes"]
    meta = {"v3_physical": res["v3"]["physical"], "v3_kind": res["v3"]["kind"],
            "id1_kind": res["id1"]["kind"], "id1_physical": res["id1"]["physical"],
            "K1_K": int(K1.max()) + 1, "K2_K": int(K2.max()) + 1}
    t0 = time.perf_counter()
    m2 = int(K2.max()) + 1
    P2 = (K1.astype(np.int64) * np.int64(m2) + K2.astype(np.int64)).astype(np.int32)
    pack_ms = (time.perf_counter() - t0) * 1000
    cases = {"Q1": (K1, [V1]), "Q2": (P2, [V1]), "Q3": (K3, [V1, V3]),
             "Q4": (K4, [V1, V2, V3]), "Q5": (K6, [V1, V2, V3])}
    keys, vcols = cases[q]
    keys = np.ascontiguousarray(keys)
    vcols = [np.ascontiguousarray(v) for v in vcols]
    return keys, vcols, int(keys.max()) + 1, K.sig_of(vcols), pack_ms, meta, csv_ms, prep_ms


def dict_stage(q, ukeys, counts, sums):
    reps = []
    for _ in range(3):
        t = time.perf_counter()
        if len(sums) == 1 and q in ("Q1", "Q2"):
            dict_fast_single(ukeys, counts, sums[0], ("sum", "count", "mean"))
        elif q == "Q3":
            dict_fast_multi(ukeys, counts, sums, [("sum",), ("mean",)])
        elif q == "Q4":
            dict_fast_multi(ukeys, counts, sums, [("mean",), ("mean",), ("mean",)])
        else:
            dict_fast_multi(ukeys, counts, sums, [("sum",), ("sum",), ("sum",)])
        reps.append((time.perf_counter() - t) * 1000)
    return min(reps)


def cell_A(keys, vcols, m, sig, t):
    n = keys.shape[0]
    bounds = np.linspace(0, n, t + 1).astype(np.int64)
    kk = K._SLICE[sig]
    walls, mgs, got = [], [], None
    for _ in range(3):
        states = K.alloc_state(m, sig, t)
        fns = []
        for w in range(t):
            s0, s1 = keys[bounds[w]:bounds[w + 1]], [v[bounds[w]:bounds[w + 1]] for v in vcols]
            st = states[w]
            fns.append(lambda _k=s0, _v=s1, _s=st: (kk(_k, *_v, *_s[0], _s[1]), None)[1])
        wall, _, _ = run_agg(fns, t)
        tmg = time.perf_counter()
        sums_m, counts_m = K.merge_sum(states)
        mgs.append((time.perf_counter() - tmg) * 1000)
        walls.append(wall)
        got = (sums_m, counts_m)
    i = int(np.argmin(walls))
    return walls[i], mgs[i], got


def cell_D(keys, vcols, m, sig, t):
    t0 = time.perf_counter()
    groups, owner_full = K.plan_hash(m, t)
    growner = owner_full
    perm, bounds = K.build_perm(keys, growner, t)
    route_ms = (time.perf_counter() - t0) * 1000
    loads = K.load_stats(np.bincount(keys.astype(np.int64), minlength=m).astype(np.int64), groups)
    kk = K._IDX[sig]
    walls, mgs, got = [], [], None
    for _ in range(3):
        states = K.alloc_state(m, sig, t)
        fns = []
        for w in range(t):
            st = states[w]
            j0, j1 = int(bounds[w]), int(bounds[w + 1])
            fns.append(lambda _s=st, _a=j0, _b=j1: kk(keys, *vcols, perm, _a, _b, *_s[0], _s[1]))
        wall, _, _ = run_agg(fns, t)
        tmg = time.perf_counter()
        sums_m, counts_m = K.merge_gather(states, groups)
        mgs.append((time.perf_counter() - tmg) * 1000)
        walls.append(wall)
        got = (sums_m, counts_m)
    i = int(np.argmin(walls))
    return walls[i], mgs[i], route_ms, got, loads, perm.nbytes / 1e6


def correctness(q, ref, gotA, gotD):
    """Exact: ukeys/counts/int-sums identical; float max_diff reported (gate 1e-9 rel)."""
    uk, cnt, sums = ref[0], ref[1], ref[2]  # compact, aligned with uk
    out = {}
    for name, got in (("A", gotA), ("D", gotD)):
        gcnt = got[1][uk]
        ok_cnt = bool((gcnt == cnt).all())
        colrep = []
        maxd = 0.0
        ok_all = ok_cnt
        for x, y in zip(got[0], sums):
            gx, gy = x[uk], np.asarray(y)
            if gx.dtype.kind == "f":
                d = float(np.max(np.abs(gx - gy))) if gx.size else 0.0
                maxd = max(maxd, d)
                tol = 1e-9 + 1e-12 * float(np.max(np.abs(gy)))
                ok = d <= tol
            else:
                ok = bool((gx == gy).all())
            ok_all = ok_all and ok
            colrep.append(ok)
        out[name] = {"counts_exact": ok_cnt, "cols_exact": colrep,
                     "float_max_diff": maxd, "pass": bool(ok_all)}
    tot_v1 = int(np.asarray(sums[0]).sum()) if np.asarray(sums[0]).dtype.kind != "f" else None
    if q in ("Q1", "Q2"):
        out["chk_v1"] = tot_v1
        out["chk_pass"] = (tot_v1 == CHK_V1)
    else:
        out["chk_v1"] = tot_v1
        out["chk_pass"] = True
    out["ngroups"] = int(uk.size)
    return out


def main():
    q = sys.argv[1] if len(sys.argv) > 1 else "Q1"
    assert q in NGOLD, q
    ladder = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [8, 12, 16]
    assert all(t in (8, 12, 16) for t in ladder), ladder
    t_start = time.perf_counter()
    K.warmup()
    keys, vcols, m, sig, pack_ms, meta, csv_ms, prep_ms = load_case(q)
    n = int(keys.shape[0])
    t0 = time.perf_counter()
    ref = fused_singlepass(keys, vcols, m)
    ref_ms = (time.perf_counter() - t0) * 1000
    uk = ref[0]
    assert uk.size == NGOLD[q], (q, uk.size)
    dms = dict_stage(q, ref[0], ref[1], ref[2])
    print(f"{q}: N={n} G={uk.size} m={m} sig={sig} "
          f"csv={csv_ms:.0f}ms prep={prep_ms:.0f}ms pack={pack_ms:.0f}ms "
          f"ref1T={ref_ms:.0f}ms dict={dms:.2f}ms meta={meta} rss={rss_gb():.2f}GB", flush=True)
    cells = {}
    for t in ladder:
        a_agg, a_mrg, gotA = cell_A(keys, vcols, m, sig, t)
        gc.collect()
        d_agg, d_mrg, d_rte, gotD, loads, perm_mb = cell_D(keys, vcols, m, sig, t)
        gc.collect()
        corr = correctness(q, ref, (gotA[0], gotA[1]), (gotD[0], gotD[1]))
        assert corr["A"]["pass"] and corr["D"]["pass"], (q, t, corr)
        if q in ("Q1", "Q2"):
            assert corr["chk_pass"], (q, t, corr["chk_v1"])
        a_tot = a_agg + a_mrg + dms + (pack_ms if q == "Q2" else 0.0)
        d_tot = d_rte + d_agg + d_mrg + dms + (pack_ms if q == "Q2" else 0.0)
        cells[str(t)] = {
            "N": n, "G": int(uk.size), "T": t,
            "A": {"agg_ms": round(a_agg, 2), "merge_ms": round(a_mrg, 3),
                  "total_ms": round(a_tot, 2)},
            "D": {"route_ms": round(d_rte, 2), "agg_ms": round(d_agg, 2),
                  "merge_ms": round(d_mrg, 3), "total_ms": round(d_tot, 2),
                  "perm_mb": round(perm_mb, 2), "imb_pct": round(loads["imb_pct"], 2)},
            "ratio_agg_D_over_A": round(d_agg / a_agg, 3),
            "ratio_total_D_over_A": round(d_tot / a_tot, 3),
            "correct": corr,
        }
        print(f"  {q} T={t}: A agg={a_agg:.1f} mrg={a_mrg:.2f} tot={a_tot:.1f} | "
              f"D rte={d_rte:.0f} agg={d_agg:.1f} mrg={d_mrg:.2f} tot={d_tot:.1f} "
              f"r_agg={d_agg / a_agg:.2f} r_tot={d_tot / a_tot:.2f} "
              f"maxdiffA={corr['A']['float_max_diff']:.1e} "
              f"maxdiffD={corr['D']['float_max_diff']:.1e} chk={corr.get('chk_v1')}",
              flush=True)
    out = {"tag": f"ad_{q}", "Q": q, "N": n, "G": int(uk.size), "m": m, "sig": sig,
           "seed": SEED, "pack_ms": round(pack_ms, 1), "dict_ms": round(dms, 3),
           "meta": meta, "cells": cells,
           "total_s": round(time.perf_counter() - t_start, 1)}
    fn = FORK / "tests" / "research" / "ad_control" / f"results_ad_{q}.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=1)
    print(f"WROTE {fn.name} total={out['total_s']}s rss={rss_gb():.2f}GB", flush=True)
    for p in POOLS.values():
        p.shutdown()


if __name__ == "__main__":
    main()
