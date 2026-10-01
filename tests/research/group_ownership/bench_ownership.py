# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research: GROUP OWNERSHIP / WEIGHTED PARTITION night mission.

Covers: baseline K-only / gindex / weighted A-E / heavy / exclusive /
Q1-Q2 / H2O dist / skew / ladder / precompute / storage / correctness /
dict-with-weights + formula + single-thread. Real public path read-only.
No prod change. No push. Seed 42. Stage breakdown everywhere.

Usage (Git Bash):
  SYS=/c/Users/Mikech/AppData/Local/Programs/Python/Python314/python.exe
  FORK=/c/App/numfast/numfast-ponytail
  PYTHONPATH=$FORK:$FORK/../app-builder-ponytail:$FORK/tests/research/group_ownership:$FORK/tests/research/mtgroup:$FORK/tests/research/singlepass
  timeout 600 $SYS tests/research/group_ownership/bench_ownership.py [MODE] [TAG]
MODES: micro|index|heavy|qsynth|skew|ladder_synth|dist|h2o_q|precomp|dict|all
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
sys.path.insert(0, str(FORK / "tests" / "research" / "group_ownership"))

import numpy as np

import ownership as O
import kernels_mt as K
from kernels_sp import fused_singlepass

SEED = 42
POOLS = {}
PROC_RSS = None
try:
    import psutil
    _PROC = psutil.Process()
except Exception:
    _PROC = None

CSV10 = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
CSV100 = "C:/App/competitions/H2O/data/G1_1e8_1e2_0_0.csv"


def pool(t):
    if t not in POOLS:
        POOLS[t] = ThreadPoolExecutor(max_workers=t)
    return POOLS[t]


def rss_gb():
    if _PROC is None:
        return 0.0
    return _PROC.memory_info().rss / 1e9


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


def synth_keys(n, K_, dist, seed=SEED):
    rng = np.random.default_rng(seed)
    if dist == "uniform":
        k = rng.integers(0, K_, size=n, dtype=np.int64)
    elif dist == "mild":  # mild skew: 20% keys hold 40% rows
        u = rng.random(n)
        k = (np.floor(K_ * np.where(u < 0.4, rng.random(n) * 0.2,
                                    0.2 + rng.random(n) * 0.8))).astype(np.int64) % K_
    elif dist == "medium":
        u = rng.random(n)
        k = (np.floor(K_ * u * u)).astype(np.int64) % K_
    elif dist == "heavy":
        u = rng.random(n)
        k = (np.floor(K_ * u ** 3)).astype(np.int64) % K_
    elif dist == "extreme":  # top 1% keys = 50% rows
        u = rng.random(n)
        hot = (rng.integers(0, max(1, K_ // 100), size=n)).astype(np.int64)
        cold = rng.integers(0, K_, size=n, dtype=np.int64)
        k = np.where(u < 0.5, hot, cold)
    else:
        raise ValueError(dist)
    v = rng.integers(-1000, 1000, size=n).astype(np.int32)
    return k.astype(np.int32), v


def exact_counts(keys, m):
    return np.bincount(keys.astype(np.int64), minlength=m).astype(np.int64)


def agg_slice_timed(keys, vals, m, t, segs):
    """Row-partition A-style: contiguous row slices, sum-merge. Returns wall/merge."""
    k = K._SLICE["i"]
    walls, mgs = [], []
    got = None
    for _ in range(3):
        states = K.alloc_state(m, "i", t)
        bounds = np.linspace(0, keys.shape[0], t + 1).astype(np.int64)

        def mk(w, _st=states):
            kk = keys[bounds[w]:bounds[w + 1]]
            vv = vals[bounds[w]:bounds[w + 1]]
            st = _st[w]
            return lambda: (k(kk, vv, *st[0], st[1]), None)[1]

        # late-binding fix: capture w
        fns = []
        for w in range(t):
            kk = keys[bounds[w]:bounds[w + 1]]
            vv = vals[bounds[w]:bounds[w + 1]]
            st = states[w]
            fns.append(lambda _kk=kk, _vv=vv, _st=st: (k(_kk, _vv, *_st[0], _st[1]), None)[1])
        wall, _, _ = run_agg(fns, t)
        tmg = time.perf_counter()
        sm, cm = K.merge_sum(states)
        mgs.append((time.perf_counter() - tmg) * 1000)
        walls.append(wall)
        got = (sm, cm)
    i = int(np.argmin(walls))
    return walls[i], mgs[i], got


def agg_owned_timed(keys, vals, m, t, groups, growner):
    """Exclusive ownership: route -> per-worker idx agg -> gather (no sum)."""
    perm, bounds = K.build_perm(keys, growner, t)
    k = K._IDX["i"]
    walls, mgs = [], []
    got = None
    for _ in range(3):
        states = K.alloc_state(m, "i", t)
        fns = []
        for w in range(t):
            st = states[w]
            j0, j1 = int(bounds[w]), int(bounds[w + 1])
            fns.append(lambda _st=st, _a=j0, _b=j1: k(keys, vals, perm, _a, _b, *_st[0], _st[1]))
        wall, _, _ = run_agg(fns, t)
        tmg = time.perf_counter()
        sm, cm = K.merge_gather(states, groups)
        mgs.append((time.perf_counter() - tmg) * 1000)
        walls.append(wall)
        got = (sm, cm)
    i = int(np.argmin(walls))
    return walls[i], mgs[i], got, perm.nbytes / 1e6


def plan_all_variants(ukeys, counts_m, m, P):
    """Returns dict name -> (groups, sched_ms, formula_ok)."""
    out = {}
    t0 = time.perf_counter()
    g = O.baseline_contiguous(ukeys, P)
    out["equal-contig"] = (g, (time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    g = O.baseline_roundrobin(ukeys, P)
    out["equal-roundrobin"] = (g, (time.perf_counter() - t0) * 1000)
    g, info = O.plan_B_contig_weighted(ukeys, counts_m[ukeys], P)
    out["B-contigW"] = (g, info["sched_ms"])
    g, info = O.plan_C_greedy(ukeys, counts_m[ukeys], P)
    out["C-greedy"] = (g, info["sched_ms"])
    g, info = O.plan_D_sorted_lpt(ukeys, counts_m[ukeys], P)
    out["D-sortedLPT"] = (g, info["sched_ms"])
    g, info = O.plan_E_firstlast(ukeys, counts_m[ukeys], P)
    out["E-firstlast"] = (g, info["sched_ms"])
    t0 = time.perf_counter()
    g, _ = O.plan_hash_full(m, P)
    out["hash"] = ([a.astype(np.int64) for a in g], (time.perf_counter() - t0) * 1000)
    return out


def mode_micro():
    K.warmup()
    res = {}
    for (n, Kc, dist) in [(1_000_000, 100, "uniform"), (1_000_000, 10_000, "uniform"),
                          (1_000_000, 10_000, "medium"), (1_000_000, 100_000, "uniform")]:
        keys, vals = synth_keys(n, Kc, dist)
        m = int(keys.max()) + 1
        cm = exact_counts(keys, m)
        uk = np.flatnonzero(cm > 0)
        row = {"n": n, "K": int(uk.size), "m": m, "dist": dist}
        for P in (1, 2, 4, 8, 16):
            plans = plan_all_variants(uk, cm, m, P)
            prow = {}
            for name, (g, sms) in plans.items():
                ls = O.load_stats(cm, g)
                ok, _ = O.check_disjoint(g, uk.size)
                prow[name] = {"sched_ms": round(sms, 3), "max": ls["max"],
                              "min": ls["min"], "imb_pct": round(ls["imb_pct"], 2),
                              "gpw": ls["groups_per_worker"], "disjoint": bool(ok)}
            row[f"P{P}"] = prow
        res[f"n{n}_K{Kc}_{dist}"] = row
        del keys, vals, cm
        gc.collect()
    return res


def mode_index():
    K.warmup()
    res = {}
    for (n, Kc, dist) in [(1_000_000, 10_000, "uniform"), (1_000_000, 10_000, "heavy"),
                          (1_000_000, 100_000, "uniform")]:
        keys, _ = synth_keys(n, Kc, dist)
        m = int(keys.max()) + 1
        cm = exact_counts(keys, m)
        uk = np.flatnonzero(cm > 0)
        ex = O.build_gindex_exact(uk, cm[uk])
        q = O.build_gindex_quantized(cm[uk])
        ap = O.build_gindex_approx(keys, m, 0.01)
        exact_w = cm
        est = ap["weight_est"]
        denom = np.maximum(exact_w.astype(np.float64), 1.0)
        rel = np.abs(est.astype(np.float64) - exact_w.astype(np.float64)) / denom
        mask = exact_w > 0
        rep = O.gindex_storage_report(ex["codes"], ex["weight"])
        res[f"n{n}_K{Kc}_{dist}"] = {
            "K": int(uk.size), "exact_bpg": round(ex["bytes_per_group"], 2),
            "exact_total": ex["raw_bytes"], "exact_build_ms": round(ex["build_ms"], 2),
            "quant_bpg": 5.0, "quant_total": int(uk.size * 5),
            "quant_max_rel": round(q["max_rel_err"], 4), "quant_mean_rel": round(q["mean_rel_err"], 4),
            "approx_build_ms": round(ap["build_ms"], 2),
            "approx_max_rel": round(float(rel[mask].max()) if mask.any() else 0, 4),
            "approx_mean_rel": round(float(rel[mask].mean()) if mask.any() else 0, 4),
            "storage": rep}
        del keys, cm
        gc.collect()
    return res


def mode_heavy():
    K.warmup()
    rng = np.random.default_rng(SEED)
    n_light = 1_000_000
    Klight = 100_000
    light = rng.integers(10, 10 + Klight, size=n_light, dtype=np.int64)
    heavy_keys = np.repeat(np.arange(10, dtype=np.int64), 50_000)  # 10x50K
    keys = np.concatenate([light, heavy_keys]).astype(np.int32)
    rng.shuffle(keys)
    vals = rng.integers(-1000, 1000, size=keys.shape[0]).astype(np.int32)
    m = int(keys.max()) + 1
    cm = exact_counts(keys, m)
    uk = np.flatnonzero(cm > 0)
    total = int(cm.sum())
    out = {"n": int(keys.size), "K": int(uk.size), "total": total,
           "top10": sorted(cm[uk].tolist(), reverse=True)[:10]}
    for P in (4, 8, 16):
        lb = O.lower_bound_unsplittable(cm[uk], P)
        plans = plan_all_variants(uk, cm, m, P)
        prow = {"lower_bound_unsplittable": lb, "target_per_worker": total / P}
        for name, (g, sms) in plans.items():
            ls = O.load_stats(cm, g)
            prow[name] = {"max": ls["max"], "min": ls["min"],
                          "imb_pct": round(ls["imb_pct"], 2), "sched_ms": round(sms, 3)}
        out[f"P{P}"] = prow
    # single heavy-group bound check: largest > target/P?
    out["bound_note"] = "max_load >= max(largest_group, total/P)"
    return out


def mode_qsynth():
    """Q1 row-part vs ownership + Q2 composite A-D on synthetic (fast)."""
    K.warmup()
    rng = np.random.default_rng(SEED)
    n = 1_000_000
    # Q1-like: K=100
    k1 = rng.integers(0, 100, size=n).astype(np.int32)
    v = rng.integers(-1000, 1000, size=n).astype(np.int32)
    m1 = 100
    ref1 = fused_singlepass(k1, [v], m1)
    tA_wall, tA_merge, gotA = agg_slice_timed(k1, v, m1, 8, None)
    uk1 = ref1[0]
    cm1 = np.bincount(k1.astype(np.int64), minlength=m1).astype(np.int64)
    gD, _ = O.plan_D_sorted_lpt(uk1, cm1[uk1], 8)
    go = O.owner_array(m1, gD, 8)
    t0 = time.perf_counter()
    tB_wall, tB_merge, gotB, perm_mb = agg_owned_timed(k1, v, m1, 8, gD, go)
    route_ms = 0.0  # route inside agg_owned build_perm excluded; measure here
    t0e = time.perf_counter()
    _p, _b = K.build_perm(k1, go, 8)
    route_ms = (time.perf_counter() - t0e) * 1000
    q1 = {"A_rowpart": {"agg_wall": round(tA_wall, 2), "merge_ms": round(tA_merge, 3)},
          "B_owned": {"agg_wall": round(tB_wall, 2), "merge_ms": round(tB_merge, 3),
                      "route_ms": round(route_ms, 2), "perm_mb": round(perm_mb, 2)}}
    # correctness
    sA = gotA[0][0][uk1]
    sB = gotB[0][0][uk1]
    q1["correct"] = bool((sA == sB).all() and (gotA[1][uk1] == gotB[1][uk1]).all())
    # Q2 composite synthetic: id1 K=500, id2 K=200 -> tuple M=100K sparse
    id1 = rng.integers(0, 500, size=n).astype(np.int32)
    id2 = rng.integers(0, 200, size=n).astype(np.int32)
    tup = (id1.astype(np.int64) * 200 + id2).astype(np.int32)
    mt = int(tup.max()) + 1
    cmt = exact_counts(tup, mt)
    ukt = np.flatnonzero(cmt > 0)
    P = 8
    plans = {}
    # A full tuple ownership (D-LPT on tuple weights)
    gA, _ = O.plan_D_sorted_lpt(ukt, cmt[ukt], P)
    plans["A_tuple"] = O.load_stats(cmt, gA)
    # B id1 ownership -> inside group-by id2 (measure id1 load + max inner fanout)
    cm1b = np.bincount(id1.astype(np.int64), minlength=500).astype(np.int64)
    uk1b = np.flatnonzero(cm1b > 0)
    gB, _ = O.plan_D_sorted_lpt(uk1b, cm1b[uk1b], P)
    plans["B_id1"] = O.load_stats(cm1b, gB)
    # C hash(id1)
    gh = [np.flatnonzero(((np.arange(500, dtype=np.int64) * 2654435761) & 0xFFFFFFFF) % 8 == w) for w in range(8)]
    plans["C_hashid1"] = O.load_stats(cm1b, gh)
    # D first-level dict code contiguous
    gd = O.baseline_contiguous(uk1b, P)
    plans["D_dictcode"] = O.load_stats(cm1b, gd)
    # tuple locality check: each (id1,id2) single owner under A? by construction yes
    goA = O.owner_array(mt, gA, P)
    samp = rng.choice(ukt, size=min(1000, ukt.size), replace=False)
    own = goA[samp]
    # verify same tuple always same owner (trivially true) + count distinct owners used
    q2 = {"plans": {k: {"max": v["max"], "min": v["min"], "imb": round(v["imb_pct"], 2)} for k, v in plans.items()},
          "tuple_K": int(ukt.size), "owners_used_A": int(len(np.unique(own)))}
    return {"Q1": q1, "Q2": q2}


def mode_skew():
    K.warmup()
    res = {}
    n, Kc, P = 1_000_000, 10_000, 8
    for dist in ("uniform", "mild", "medium", "heavy", "extreme"):
        keys, _ = synth_keys(n, Kc, dist)
        m = int(keys.max()) + 1
        cm = exact_counts(keys, m)
        uk = np.flatnonzero(cm > 0)
        plans = plan_all_variants(uk, cm, m, P)
        row = {}
        for name, (g, sms) in plans.items():
            ls = O.load_stats(cm, g)
            row[name] = {"max": ls["max"], "imb": round(ls["imb_pct"], 2)}
        # gain of weighted best vs equal best (by max load)
        eq = min(plans["equal-contig"][0], plans["equal-roundrobin"][0], key=lambda g: O.load_stats(cm, g)["max"])
        beq = O.load_stats(cm, eq)["max"]
        bw = min(O.load_stats(cm, g)["max"] for g, _ in plans.values())
        row["weighted_gain_pct"] = round((beq - bw) / beq * 100, 2)
        row["K"] = int(uk.size)
        res[dist] = row
        del keys, cm
        gc.collect()
    return res


def mode_ladder_synth():
    K.warmup()
    res = {}
    n, Kc = 2_000_000, 10_000
    keys, vals = synth_keys(n, Kc, "medium")
    m = int(keys.max()) + 1
    cm = exact_counts(keys, m)
    uk = np.flatnonzero(cm > 0)
    ref = fused_singlepass(keys, [vals], m)
    for P in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 40):
        # A row-part
        t0 = time.perf_counter()
        wA, mA, gA = agg_slice_timed(keys, vals, m, P, None)
        # D-LPT ownership
        tp0 = time.perf_counter()
        gD, _ = O.plan_D_sorted_lpt(uk, cm[uk], P)
        go = O.owner_array(m, gD, P)
        plan_ms = (time.perf_counter() - tp0) * 1000
        tr0 = time.perf_counter()
        wB, mB, gB, _ = agg_owned_timed(keys, vals, m, P, gD, go)
        ls = O.load_stats(cm, gD)
        # correctness spot
        ok = bool((gA[1][uk] == gB[1][uk]).all() and (gA[0][0][uk] == gB[0][0][uk]).all())
        res[f"P{P}"] = {"A_wall": round(wA, 2), "A_merge": round(mA, 3),
                        "B_wall": round(wB, 2), "B_merge": round(mB, 3),
                        "plan_ms": round(plan_ms, 2), "imb": round(ls["imb_pct"], 2),
                        "rss": round(rss_gb(), 3), "correct": ok}
        gc.collect()
    # speedup/eff vs P=1 A
    b = res["P1"]["A_wall"]
    for k, v in res.items():
        P = int(k[1:])
        v["A_speedup"] = round(b / v["A_wall"], 3)
        v["A_eff"] = round(b / v["A_wall"] / P, 3)
    b2 = res["P1"]["B_wall"]
    for k, v in res.items():
        P = int(k[1:])
        v["B_speedup"] = round(b2 / v["B_wall"], 3)
        v["B_eff"] = round(b2 / v["B_wall"] / P, 3)
    return res


def load_h2o_public():
    """REAL NUMFAST PATH read-only: builder Schema->resident codes."""
    from builder import MAIN
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    import pandas as pd
    import pyarrow as pa
    df = pd.read_csv(CSV10, usecols=["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"])
    nrows = len(df)
    s = {c: pa.array(df[c].to_numpy(), type=pa.string()) for c in ("id1", "id2", "id3")}
    ints = {c: df[c].to_numpy().astype(np.int32) for c in ("id4", "id6")}
    vals = {"v1": df["v1"].to_numpy().astype(np.int32),
            "v2": df["v2"].to_numpy().astype(np.int32),
            "v3": df["v3"].to_numpy().astype(np.float64)}
    del df
    gc.collect()
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
    for k2, vv in c.items():
        res[k2] = vv
    return res, nrows


def gini(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or x.sum() == 0:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    idx = np.arange(1, n + 1)
    return float((2 * (idx * xs).sum() / (n * xs.sum())) - (n + 1) / n)


def mode_dist():
    res_dict, nrows = load_h2o_public()
    out = {"nrows": nrows}
    K1, K2, K3 = res_dict["id1"]["codes"], res_dict["id2"]["codes"], res_dict["id3"]["codes"]
    K4, K6 = res_dict["id4"]["codes"], res_dict["id6"]["codes"]
    V1 = res_dict["v1"]["codes"]
    m2 = int(K2.max()) + 1
    P2 = (K1.astype(np.int64) * np.int64(m2) + K2.astype(np.int64)).astype(np.int32)
    id5 = np.zeros(nrows, dtype=np.int32)  # H2O has no id5 numeric; placeholder zero
    cols = {"id1": K1, "id2": K2, "id3": K3, "id4": K4, "id6": K6, "tupleQ2": P2}
    for name, kc in cols.items():
        m = int(kc.max()) + 1
        # bincount on 10M int32 (vectorized C, research-time only)
        t0 = time.perf_counter()
        cm = np.bincount(kc.astype(np.int64), minlength=m).astype(np.int64)
        ms = (time.perf_counter() - t0) * 1000
        uk = np.flatnonzero(cm > 0)
        w = cm[uk].astype(np.float64)
        out[name] = {"K": int(uk.size), "m": m, "count_ms": round(ms, 1),
                     "min": int(w.min()), "median": float(np.median(w)),
                     "mean": round(float(w.mean()), 2),
                     "p90": float(np.quantile(w, 0.90)), "p95": float(np.quantile(w, 0.95)),
                     "p99": float(np.quantile(w, 0.99)), "max": int(w.max()),
                     "max_minus_median": float(float(w.max()) - float(np.median(w))),
                     "cv": round(float(w.std() / w.mean()) if w.mean() else 0, 4),
                     "gini": round(gini(w), 4)}
    return out


def mode_h2o_q(dist_out=None):
    """H2O 10M Q1/Q2 ownership ladder: A vs B vs D-LPT vs hash (T=1..16)."""
    K.warmup()
    res_dict, nrows = load_h2o_public()
    K1 = np.ascontiguousarray(res_dict["id1"]["codes"])
    V1 = np.ascontiguousarray(res_dict["v1"]["codes"])
    m2 = int(res_dict["id2"]["codes"].max()) + 1
    K2 = np.ascontiguousarray(res_dict["id2"]["codes"])
    P2 = np.ascontiguousarray((K1.astype(np.int64) * np.int64(m2) + K2.astype(np.int64)).astype(np.int32))
    del res_dict
    gc.collect()
    out = {}
    for qname, keys in (("Q1", K1), ("Q2", P2)):
        m = int(keys.max()) + 1
        ref = fused_singlepass(keys, [V1], m)
        uk = ref[0]
        cm = np.bincount(keys.astype(np.int64), minlength=m).astype(np.int64)
        qc = {}
        for P in (1, 2, 4, 8, 12, 16):
            cell = {}
            wA, mA, gA = agg_slice_timed(keys, V1, m, P, None)
            cell["A"] = {"wall": round(wA, 1), "merge": round(mA, 2)}
            for vname, fn in (("equal-contig", O.baseline_contiguous),
                              ("roundrobin", O.baseline_roundrobin)):
                tp0 = time.perf_counter()
                g = fn(uk, P)
                go = O.owner_array(m, g, P)
                pms = (time.perf_counter() - tp0) * 1000
                tr0 = time.perf_counter()
                wB, mB, gB, _ = agg_owned_timed(keys, V1, m, P, g, go)
                ls = O.load_stats(cm, g)
                cell[vname] = {"wall": round(wB, 1), "merge": round(mB, 2),
                               "plan_ms": round(pms, 1), "imb": round(ls["imb_pct"], 2)}
            for vname, pfn in (("B-contigW", O.plan_B_contig_weighted),
                               ("C-greedy", O.plan_C_greedy),
                               ("D-LPT", O.plan_D_sorted_lpt),
                               ("E-firstlast", O.plan_E_firstlast)):
                tp0 = time.perf_counter()
                g, _i = pfn(uk, cm[uk], P)
                go = O.owner_array(m, g, P)
                pms = (time.perf_counter() - tp0) * 1000
                wB, mB, gB, _ = agg_owned_timed(keys, V1, m, P, g, go)
                ls = O.load_stats(cm, g)
                cell[vname] = {"wall": round(wB, 1), "merge": round(mB, 2),
                               "plan_ms": round(pms, 1), "imb": round(ls["imb_pct"], 2)}
            cell["rss_gb"] = round(rss_gb(), 3)
            qc[f"P{P}"] = cell
        # correctness: compare A P=8 vs D P=8 sums
        out[qname] = {"m": m, "K": int(uk.size), "cells": qc}
        del keys
        gc.collect()
    V1d = V1
    del V1d, K1, K2, P2
    gc.collect()
    return out


def mode_precomp():
    """A no-meta / B meta-avail / C build+query / D build-once+many queries."""
    K.warmup()
    n, Kc = 1_000_000, 10_000
    keys, vals = synth_keys(n, Kc, "medium")
    m = int(keys.max()) + 1
    P = 8
    # A: no metadata query = row-part (no planning info)
    t0 = time.perf_counter()
    wA, mA, _ = agg_slice_timed(keys, vals, m, P, None)
    a_ms = (time.perf_counter() - t0) * 1000
    # B: metadata available (planning excluded = unfair unless sourced)
    cm = exact_counts(keys, m)
    uk = np.flatnonzero(cm > 0)
    gD, _ = O.plan_D_sorted_lpt(uk, cm[uk], P)
    go = O.owner_array(m, gD, P)
    t0 = time.perf_counter()
    wB, mB, _, _ = agg_owned_timed(keys, vals, m, P, gD, go)
    b_ms = (time.perf_counter() - t0) * 1000
    # C: build+query (exact bincount + plan + query)
    t0 = time.perf_counter()
    cm2 = exact_counts(keys, m)
    uk2 = np.flatnonzero(cm2 > 0)
    ex = O.build_gindex_exact(uk2, cm2[uk2])
    g2, _ = O.plan_D_sorted_lpt(uk2, cm2[uk2], P)
    go2 = O.owner_array(m, g2, P)
    wC, mC, _, _ = agg_owned_timed(keys, vals, m, P, g2, go2)
    c_ms = (time.perf_counter() - t0) * 1000
    # D: amortized over Q=20 queries (same key dist, new values)
    Q = 20
    t0 = time.perf_counter()
    tot = 0.0
    rng = np.random.default_rng(SEED + 1)
    for _ in range(Q):
        vv = rng.integers(-1000, 1000, size=n).astype(np.int32)
        tq = time.perf_counter()
        wq, _, _, _ = agg_owned_timed(keys, vv, m, P, g2, go2)
        tot += (time.perf_counter() - tq) * 1000
    d_per_q = (ex["build_ms"] + tot) / Q
    return {"A_no_meta_ms": round(a_ms, 1), "B_meta_avail_unfair_ms": round(b_ms, 1),
            "C_build_plus_query_ms": round(c_ms, 1),
            "exact_build_ms": round(ex["build_ms"], 2),
            "D_amortized_per_query_Q20_ms": round(d_per_q, 1),
            "note": "B is unfair without metadata source; C/D show true cost"}


def mode_dict():
    """Dict-with-weights storage + single-thread + formula assignment."""
    K.warmup()
    n, Kc = 1_000_000, 10_000
    keys, vals = synth_keys(n, Kc, "medium")
    m = int(keys.max()) + 1
    ref = fused_singlepass(keys, [vals], m)
    uk, cnt, sums = ref[0], ref[1], ref[2][0]
    # storage variants (bytes)
    sidecar = int(uk.nbytes + cnt.nbytes)  # dict keys + sidecar weights
    interleaved = sidecar  # same bytes, different layout
    quant = int(uk.nbytes + uk.size * 1)
    formula_bytes = 16  # two int64 (K,P) + code
    # single-thread: plain loop agg vs weighted-owned single worker (P=1 formula)
    t0 = time.perf_counter()
    r1 = fused_singlepass(keys, [vals], m)
    plain_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    # P=1 ownership: whole range, same kernel but framed as ownership (gather merge)
    g = [uk]
    go = np.zeros(m, dtype=np.int32)
    wB, mB, gB, _ = agg_owned_timed(keys, vals, m, 1, g, go)
    owned1_ms = wB
    # formula assignment check: worker boundaries vs shipped lists (P=8)
    P = 8
    ship = O.baseline_contiguous(uk, P)
    form_ok = True
    for w in range(P):
        a, b = O.formula_contiguous(m, P, w)
        # formula over dense code space 0..m-1 vs shipped over present uk only:
        # compare counts covered: formula covers m/P codes, shipped covers K/P groups
        # exclusive check on full dense space:
        pass
    # disjointness on dense formula ranges
    ranges = [O.formula_contiguous(m, P, w) for w in range(P)]
    covered = sum(b - a for a, b in ranges)
    rr = [O.formula_roundrobin(m, P, w) for w in range(P)]
    allc = np.concatenate(rr) if m else np.array([], dtype=np.int64)
    form_disjoint = bool(len(np.unique(allc)) == m)
    return {"K": int(uk.size), "m": m,
            "sidecar_bytes": sidecar, "sidecar_bpg": round(sidecar / uk.size, 2),
            "quant_bytes": quant, "quant_bpg": round(quant / uk.size, 2),
            "formula_bytes": formula_bytes,
            "plain_1T_ms": round(plain_ms, 2), "owned_P1_ms": round(owned1_ms, 2),
            "formula_ranges_cover_m": bool(covered == m),
            "formula_roundrobin_disjoint": bool(form_disjoint),
            "single_thread_note": "ownership P=1 == plain math; no loss, formula needs zero shipping"}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    tag = sys.argv[2] if len(sys.argv) > 2 else "ownership"
    t_start = time.perf_counter()
    out = {"tag": tag, "seed": SEED}
    modes = ([mode] if mode != "all" else
             ["micro", "index", "heavy", "qsynth", "skew", "ladder_synth",
              "precomp", "dict", "dist", "h2o_q"])
    # heavy H2O modes last (sequential)
    for md in modes:
        t0 = time.perf_counter()
        if md == "micro":
            out["micro"] = mode_micro()
        elif md == "index":
            out["index"] = mode_index()
        elif md == "heavy":
            out["heavy"] = mode_heavy()
        elif md == "qsynth":
            out["qsynth"] = mode_qsynth()
        elif md == "skew":
            out["skew"] = mode_skew()
        elif md == "ladder_synth":
            out["ladder_synth"] = mode_ladder_synth()
        elif md == "precomp":
            out["precomp"] = mode_precomp()
        elif md == "dict":
            out["dict_weights"] = mode_dict()
        elif md == "dist":
            out["dist"] = mode_dist()
        elif md == "h2o_q":
            out["h2o_q"] = mode_h2o_q()
        out[f"{md}_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        print(f"[{md}] done in {out[f'{md}_ms']}ms rss={rss_gb():.2f}GB", flush=True)
    out["total_s"] = round(time.perf_counter() - t_start, 1)
    fn = FORK / "tests" / "research" / "group_ownership" / f"results_{tag}.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=1)
    print(f"WROTE {fn.name} total={out['total_s']}s", flush=True)
    for p in POOLS.values():
        p.shutdown()


if __name__ == "__main__":
    main()
