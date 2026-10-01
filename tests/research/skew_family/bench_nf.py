# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research-only: NumFast production resident path (query-only, run1/run2,
chk separate) + MT diagnostic A/B/C/E at 1/4/8/16T on skew family 10M.

Production path = builder Schema->IR->Planner->Runtime->CPU, resident codes,
NO weighted ownership (current planner only). Diagnostic variants reuse
mtgroup/kernels_mt (measurement only, no integration, no prod change).

Boundary: t_start->t_end per query (query-only; load+prep reported
separately); chk vs own stats_chk.json reference AFTER timing (separate).
Thread proof: prod single-thread (OMP/NUMBA=1, pool=1); diagnostic pool size
= T (ThreadPoolExecutor max_workers=T), cpu_cores measured, ncpu logged.
No GPU. No SPEC change. No push.
Usage (Git Bash, fork-first):
  C:/App/competitions/H2O/python310/python.exe tests/research/skew_family/bench_nf.py [level]
  levels: uniform | mild-skew | medium-skew | heavy-skew | all (default)
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
import psutil

import kernels_mt as K
from kernels_sp import fused_singlepass

SESS = Path("C:/App/competitions/H2O/session/skew_family")
LEVELS_ALL = ["uniform", "mild-skew", "medium-skew", "heavy-skew"]
LADDER = [1, 4, 8, 16]
VARIANTS = ("A", "B", "C", "E")
PROC = psutil.Process()
POOLS = {}


def pool(t):
    if t not in POOLS:
        POOLS[t] = ThreadPoolExecutor(max_workers=t)
    return POOLS[t]


def rss():
    return PROC.memory_info().rss / 1e9


def run_graph(a, jobs, n):
    st = {}
    s = time.perf_counter()
    g = a["compile"](jobs)
    st["compile"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    g = a["optimize"](g)
    st["optimize"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    r = a["evaluate"](g, "cpu", n)
    st["execute"] = (time.perf_counter() - s) * 1000
    st["total"] = st["compile"] + st["optimize"] + st["execute"]
    return r["result"], st


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
    wall = (time.perf_counter() - t0) * 1000
    return wall, [r[0] for r in res], [r[1] for r in res]


def check_mt(got, ref, label):
    uka, ca, sa = got
    ukb, cb, sb = ref
    assert uka.shape == ukb.shape and bool((uka == ukb).all()), label
    assert bool((ca == cb).all()), label + " counts"
    for x, y in zip(sa, sb):
        if x.dtype.kind == "f":
            d = float(np.max(np.abs(x - y))) if x.size else 0.0
            assert d <= 1e-9 + 1e-12 * float(np.max(np.abs(y))), f"{label} fdiff {d}"
        else:
            assert x.dtype == y.dtype and bool((x == y).all()), label + " sums"


def cell_A(keys, vcols, m, sig, t):
    n = keys.shape[0]
    bounds = np.linspace(0, n, t + 1).astype(np.int64)
    segs = [(keys[bounds[w]:bounds[w + 1]],
             [v[bounds[w]:bounds[w + 1]] for v in vcols]) for w in range(t)]
    k = K._SLICE[sig]
    walls, mgs = [], []
    got = None
    for _ in range(3):
        states = K.alloc_state(m, sig, t)

        def mk(w, segs=segs, states=states, k=k):
            kk, vv = segs[w]
            st = states[w]
            return lambda: (k(kk, *vv, *st[0], st[1]), None)[1]

        wall, _, _ = run_agg([mk(w) for w in range(t)], t)
        tmg = time.perf_counter()
        sums_m, counts_m = K.merge_sum(states)
        mgs.append((time.perf_counter() - tmg) * 1000)
        walls.append(wall)
        got = (sums_m, counts_m)
    i = int(np.argmin(walls))
    return {"best_ms": walls[i], "reps_ms": walls, "merge_ms": mgs[i],
            "route_ms": 0.0}, got


def cell_B(keys, vcols, m, sig, t):
    k = K._STR[sig]
    walls, mgs = [], []
    got = None
    for _ in range(3):
        states = K.alloc_state(m, sig, t)

        def mk(w, states=states, k=k):
            st = states[w]
            return lambda: k(keys, *vcols, *st[0], st[1], w, t)

        wall, _, _ = run_agg([mk(w) for w in range(t)], t)
        tmg = time.perf_counter()
        sums_m, counts_m = K.merge_sum(states)
        mgs.append((time.perf_counter() - tmg) * 1000)
        walls.append(wall)
        got = (sums_m, counts_m)
    i = int(np.argmin(walls))
    return {"best_ms": walls[i], "reps_ms": walls, "merge_ms": mgs[i],
            "route_ms": 0.0}, got


def cell_CDE(keys, vcols, m, sig, t, mode, ukeys):
    t0 = time.perf_counter()
    cnt_m = np.bincount(keys, minlength=m).astype(np.int64)
    gi_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    if mode == "C":
        groups = K.plan_equal_groups(ukeys, t)
        loads = K.load_stats(cnt_m, groups)
        growner = K.owner_array(m, groups, t)
    else:
        groups, lpt = K.plan_lpt(cnt_m, ukeys, t)
        loads = {"loads": [int(x) for x in lpt], "max": int(lpt.max()),
                 "min": int(lpt.min()), "mean": float(lpt.mean()),
                 "imb_pct": float((lpt.max() - lpt.mean()) / lpt.mean() * 100)}
        growner = K.owner_array(m, groups, t)
    perm, bounds = K.build_perm(keys, growner, t)
    route_ms = (time.perf_counter() - t1) * 1000 + gi_ms
    k = K._IDX[sig]
    walls, mgs = [], []
    got = None
    for _ in range(3):
        states = K.alloc_state(m, sig, t)

        def mk(w, states=states, k=k, bounds=bounds, perm=perm):
            st = states[w]
            j0, j1 = int(bounds[w]), int(bounds[w + 1])
            return lambda: k(keys, *vcols, perm, j0, j1, *st[0], st[1])

        wall, _, _ = run_agg([mk(w) for w in range(t)], t)
        tmg = time.perf_counter()
        sums_m, counts_m2 = K.merge_gather(states, groups)
        mgs.append((time.perf_counter() - tmg) * 1000)
        walls.append(wall)
        got = (sums_m, counts_m2)
    i = int(np.argmin(walls))
    return {"best_ms": walls[i], "reps_ms": walls, "merge_ms": mgs[i],
            "route_ms": route_ms, "loads": loads}, got


def bench_level(level):
    import pyarrow.parquet as pq
    from builder import MAIN
    ref = json.load(open(SESS / f"skew_{level}_10M_stats_chk.json"))
    print(f"=== {level} ===", flush=True)
    t = time.perf_counter()
    tbl = pq.read_table(SESS / f"skew_{level}_10M.parquet")
    load_ms = (time.perf_counter() - t) * 1000
    n = tbl.num_rows
    print(f"load {load_ms:.0f}ms n={n} RSS {rss():.2f}GB", flush=True)
    import pyarrow as pa
    s_id1 = tbl["id1"].combine_chunks()
    s_id2 = tbl["id2"].combine_chunks()
    s_id3 = tbl["id3"].combine_chunks()
    id4 = tbl["id4"].to_numpy().astype(np.int32)
    id6 = tbl["id6"].to_numpy().astype(np.int32)
    v1 = tbl["v1"].to_numpy().astype(np.int32)
    v2 = tbl["v2"].to_numpy().astype(np.int32)
    v3 = tbl["v3"].to_numpy().astype(np.float64)
    del tbl
    gc.collect()
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    t = time.perf_counter()
    r1 = a["resident_prepare"]({"id1": {"values": s_id1, "prefix": "id"}})
    r2 = a["resident_prepare"]({"id2": {"values": s_id2, "prefix": "id"}})
    r3 = a["resident_prepare"]({"id3": {"values": s_id3, "prefix": "id"}})
    rn = a["resident_prepare"]({
        "id4": {"values": id4, "dtype": "int32"}, "id6": {"values": id6, "dtype": "int32"},
        "v1": {"values": v1, "dtype": "int32"}, "v2": {"values": v2, "dtype": "int32"},
        "v3": {"values": v3, "dtype": "float64"}})
    prep_ms = (time.perf_counter() - t) * 1000
    res = {"id1": r1["id1"], "id2": r2["id2"], "id3": r3["id3"]}
    res.update(rn)
    print(f"prep {prep_ms:.0f}ms RSS {rss():.2f}GB", flush=True)
    k1, k2, k3 = res["id1"]["codes"], res["id2"]["codes"], res["id3"]["codes"]
    K4c, K6c = res["id4"]["codes"], res["id6"]["codes"]
    V1, V2, V3 = res["v1"]["codes"], res["v2"]["codes"], res["v3"]["codes"]
    m2 = int(k2.max()) + 1
    P2 = (k1.astype(np.int64) * np.int64(m2) + k2.astype(np.int64)).astype(np.int32)
    out = {"level": level, "N": n, "load_ms": load_ms, "prep_ms": prep_ms,
           "ncpu": os.cpu_count(), "prod": {}, "diag": {}}
    # ---- PROD path query-only run1/run2, chk separate ----
    prods = {
        "Q1": [a["ir_series"]("k", k1), a["ir_series"]("v", V1),
               a["ir_groupby"]("g", "v", "k", "sum")],
        "Q2": [a["ir_series"]("c1", k1), a["ir_series"]("c2", k2),
               a["ir_series"]("v", V1), a["ir_pack_keys"]("k", "c1", "c2"),
               a["ir_groupby"]("g", "v", "k", "sum")],
        "Q3": [a["ir_series"]("k", k3), a["ir_series"]("v1", V1),
               a["ir_series"]("v3", V3, "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                     {"v1": ("sum",), "v3": ("mean",)})],
        "Q4": [a["ir_series"]("k", K4c), a["ir_series"]("v1", V1),
               a["ir_series"]("v2", V2), a["ir_series"]("v3", V3, "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                     {"v1": ("mean",), "v2": ("mean",), "v3": ("mean",)})],
        "Q5": [a["ir_series"]("k", K6c), a["ir_series"]("v1", V1),
               a["ir_series"]("v2", V2), a["ir_series"]("v3", V3, "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                     {"v1": ("sum",), "v2": ("sum",), "v3": ("sum",)})],
    }
    for q, jobs in prods.items():
        reps = []
        res_r = None
        for _ in range(2):
            rr, st = run_graph(a, jobs, n)
            reps.append(st["total"])
            res_r = rr
        # chk AFTER timing vs own reference
        if q in ("Q1", "Q2"):
            tot = int(sum(int(v) for v in res_r.values()))
            assert len(res_r) == ref["chk"][q]["ngroups"], (level, q, len(res_r))
            assert tot == ref["chk"][q]["total"], (level, q, tot)
        elif q == "Q3":
            m3 = float(sum(c["v3"]["mean"] for c in res_r.values()))
            assert len(res_r) == ref["chk"][q]["ngroups"], (level, q, len(res_r))
            assert abs(m3 - ref["chk"][q]["mean3_sum"]) < 1e-6 * abs(ref["chk"][q]["mean3_sum"]) + 1e-6
        elif q == "Q4":
            sums = {c: float(sum(cell[c]["mean"] for cell in res_r.values())) for c in ("v1", "v2", "v3")}
            md = max(abs(sums["v1"] - ref["chk"][q]["m1"]), abs(sums["v2"] - ref["chk"][q]["m2"]),
                     abs(sums["v3"] - ref["chk"][q]["m3"]))
            assert md < 1e-9, (level, q, md)
        else:
            assert len(res_r) == ref["chk"][q]["ngroups"], (level, q, len(res_r))
            s3 = float(sum(c["v3"]["sum"] for c in res_r.values()))
            assert abs(s3 - ref["chk"][q]["s3"]) <= 1e-6 * abs(ref["chk"][q]["s3"]) + 1e-3, \
                (level, q, s3, ref["chk"][q]["s3"])
        thr = [n / r / 1e3 for r in reps]
        out["prod"][q] = {"run1_ms": reps[0], "run2_ms": reps[1],
                          "best_ms": min(reps), "thr_k_per_s": thr,
                          "rss_gb": rss()}
        print(f"prod.{q}: run1={reps[0]:.0f} run2={reps[1]:.0f} RSS {rss():.2f}GB", flush=True)
        gc.collect()
    # ---- DIAG MT A/B/C/E ----
    cases = {"Q1": (k1, [V1]), "Q2": (P2, [V1]), "Q3": (k3, [V1, V3]),
             "Q4": (K4c, [V1, V2, V3]), "Q5": (K6c, [V1, V2, V3])}
    K.warmup()
    for q, (keys, vcols) in cases.items():
        keys = np.ascontiguousarray(keys)
        vcols = [np.ascontiguousarray(v) for v in vcols]
        m = int(keys.max()) + 1
        sig = K.sig_of(vcols)
        ref_mt = fused_singlepass(keys, vcols, m)
        ukeys = ref_mt[0]
        out["diag"][q] = {"m": m, "sig": sig, "ngroups": int(ukeys.size), "cells": {}}
        for var in VARIANTS:
            for tct in LADDER:
                c0 = PROC.cpu_times()
                if var == "A":
                    info, got = cell_A(keys, vcols, m, sig, tct)
                elif var == "B":
                    info, got = cell_B(keys, vcols, m, sig, tct)
                else:
                    info, got = cell_CDE(keys, vcols, m, sig, tct, var, ukeys)
                c1t = PROC.cpu_times()
                got_c = (ukeys, got[1][ukeys], [s[ukeys] for s in got[0]])
                check_mt(got_c, ref_mt, f"{level}{q}{var}T{tct}")
                agg_wall = info["best_ms"]
                cpu_cores = ((c1t.user - c0.user) + (c1t.system - c0.system)) / (sum(info["reps_ms"]) / 1000)
                cell = {"best_ms": round(agg_wall, 2), "merge_ms": round(info["merge_ms"], 2),
                        "route_ms": round(info["route_ms"], 2),
                        "cpu_cores": round(float(cpu_cores), 2), "rss_gb": round(rss(), 2),
                        "pool_workers": tct}
                if "loads" in info:
                    cell["loads"] = info["loads"]
                out["diag"][q]["cells"][f"{var}@{tct}"] = cell
                print(f"diag.{q}-{var}@{tct}: agg={agg_wall:.1f} mrg={info['merge_ms']:.1f} "
                      f"rte={info['route_ms']:.0f} cores={cpu_cores:.1f}", flush=True)
        gc.collect()
    with open(SESS / f"results_nf_{level}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"results_nf_{level}.json written", flush=True)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    levels = LEVELS_ALL if arg == "all" else [arg]
    for lv in levels:
        bench_level(lv)
    for p in POOLS.values():
        p.shutdown()


if __name__ == "__main__":
    main()
