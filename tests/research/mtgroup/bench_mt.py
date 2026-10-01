# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research: multithread groupby scaling matrix (A-E x Q1-Q5 x T ladder).

A contiguous rows + local state + sum-merge.
B round-robin rows + local state + sum-merge.
C group-owned workers, equal-#groups assignment (preliminary global GI).
D hash-partition ownership (multiplicative hash % T).
E group-owned + weighted LPT scheduling on exact counts.
C/D/E: worker-exclusive groups, pick-by-owner merge, no shared mutable agg.

Metrics per cell: wall (best of 3), speedup/efficiency vs same-variant T=1,
CPU cores used, RSS, analytical alloc bytes, merge ms, route ms, max queue
wait, worker busy/idle. End-to-end = pack + route + agg + merge + dict.
Research-only. No prod change. No push.
Usage: python tests/research/mtgroup/bench_mt.py [TAG]
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
from kernels_sp import (dict_fast_multi, dict_fast_single,
                        fused_singlepass)

CSV = os.environ.get("MT_CSV",
                      "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv")
N_TAG = os.environ.get("MT_NTAG", "10M")
NGOLD = {"Q1": 100, "Q2": 10000, "Q3": 100000, "Q4": 100, "Q5": 100000}
if "1e8" in CSV:
    NGOLD = {"Q1": 100, "Q2": 10000, "Q3": 1000000, "Q4": 100,
             "Q5": 1000000}
N = 10_000_000
VARIANTS = tuple(os.environ.get("MT_VARS", "A,B,C,D,E").split(","))
TADDER = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else
                           [1, 2, 3, 4, 5, 6, 7, 8, 12, 16])]
TAG = sys.argv[1] if len(sys.argv) > 1 else "mt10M"
OUT = FORK / "tests" / "research" / "mtgroup" / f"results_{TAG}.json"
PROC = psutil.Process()
NCPU = os.cpu_count() or 1
POOLS = {}


def pool(t):
    if t not in POOLS:
        POOLS[t] = ThreadPoolExecutor(max_workers=t)
    return POOLS[t]


def load_cases():
    from builder import MAIN
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    import pandas as pd
    import pyarrow as pa
    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id4", "id6",
                                   "v1", "v2", "v3"])
    nrows = len(df)
    print(f"rows={nrows}", flush=True)
    s = {c: pa.array(df[c].to_numpy(), type=pa.string())
         for c in ("id1", "id2", "id3")}
    ints = {c: df[c].to_numpy().astype(np.int32)
            for c in ("id4", "id6")}
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
    for k, vv in c.items():
        res[k] = vv
    K1, K2, K3 = res["id1"]["codes"], res["id2"]["codes"], res["id3"]["codes"]
    K4, K6 = res["id4"]["codes"], res["id6"]["codes"]
    V1, V2, V3 = res["v1"]["codes"], res["v2"]["codes"], res["v3"]["codes"]
    t = time.perf_counter()
    m1 = int(K1.max()) + 1
    m2 = int(K2.max()) + 1
    # radix pack K1*M2+K2 is dense in [0, M1*M2) by construction (K1, K2 are
    # exact int32 codes) — bijective regroup, no densify/unique needed.
    P2 = (K1.astype(np.int64) * np.int64(m2) + K2.astype(np.int64)).astype(np.int32)
    pack_ms = (time.perf_counter() - t) * 1000
    cases = {
        "Q1": (K1, [V1]),
        "Q2": (P2, [V1]),
        "Q3": (K3, [V1, V3]),
        "Q4": (K4, [V1, V2, V3]),
        "Q5": (K6, [V1, V2, V3]),
    }
    return cases, pack_ms


FLOAT_DIFFS = {}


def check(got, ref, label):
    uka, ca, sa = got
    ukb, cb, sb = ref
    assert uka.shape == ukb.shape and bool((uka == ukb).all()), label
    assert bool((ca == cb).all()), label + " counts"
    for x, y in zip(sa, sb):
        if x.dtype.kind == "f":
            # float summation order changes under threading: tolerance gate.
            assert x.dtype == y.dtype, label + " dtype"
            d = float(np.max(np.abs(x - y))) if x.size else 0.0
            FLOAT_DIFFS[label] = d
            assert d <= 1e-9 + 1e-12 * float(np.max(np.abs(y))), \
                f"{label}: float diff {d}"
        else:
            assert x.dtype == y.dtype and bool((x == y).all()), label + " sums"


def run_agg(tasks, t):
    """Submit worker closures; returns (wall_ms, qwaits, busys)."""
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


def dict_stage(q, ukeys, counts, sums):
    reps = []
    for _ in range(3):
        t = time.perf_counter()
        if len(sums) == 1 and q in ("Q1", "Q2"):
            dict_fast_single(ukeys, counts, sums[0], ("sum", "count", "mean"))
        elif q == "Q3":
            dict_fast_multi(ukeys, counts, sums,
                            [("sum",), ("mean",)])
        elif q == "Q4":
            dict_fast_multi(ukeys, counts, sums,
                            [("mean",), ("mean",), ("mean",)])
        else:
            dict_fast_multi(ukeys, counts, sums,
                            [("sum",), ("sum",), ("sum",)])
        reps.append((time.perf_counter() - t) * 1000)
    return min(reps)


def cell_A(keys, vcols, m, sig, t):
    n = keys.shape[0]
    bounds = np.linspace(0, n, t + 1).astype(np.int64)
    segs = [(keys[bounds[w]:bounds[w + 1]],
             [v[bounds[w]:bounds[w + 1]] for v in vcols]) for w in range(t)]
    k = K._SLICE[sig]
    walls, qws, busys, mgs = [], [], [], []
    got = None
    for _ in range(3):
        states = K.alloc_state(m, sig, t)

        def mk(w):
            kk, vv = segs[w]
            st = states[w]
            return lambda: (k(kk, *vv, *st[0], st[1]), None)[1]

        wall, qw, busy = run_agg([mk(w) for w in range(t)], t)
        tmg = time.perf_counter()
        sums_m, counts_m = K.merge_sum(states)
        mgs.append((time.perf_counter() - tmg) * 1000)
        walls.append(wall)
        qws, busys = qw, busy
        got = (sums_m, counts_m)
    i = int(np.argmin(walls))
    return {"walls": walls, "best": walls[i], "qwait_max": max(qws),
            "busy": busys, "merge_ms": mgs[i], "route_ms": 0.0,
            "alloc_mb": K.state_bytes(m, sig, t) / 1e6}, got


def cell_B(keys, vcols, m, sig, t):
    k = K._STR[sig]
    walls, qws, busys, mgs = [], [], [], []
    got = None
    for _ in range(3):
        states = K.alloc_state(m, sig, t)

        def mk(w):
            st = states[w]
            return lambda: k(keys, *vcols, *st[0], st[1], w, t)

        wall, qw, busy = run_agg([mk(w) for w in range(t)], t)
        tmg = time.perf_counter()
        sums_m, counts_m = K.merge_sum(states)
        mgs.append((time.perf_counter() - tmg) * 1000)
        walls.append(wall)
        qws, busys = qw, busy
        got = (sums_m, counts_m)
    i = int(np.argmin(walls))
    return {"walls": walls, "best": walls[i], "qwait_max": max(qws),
            "busy": busys, "merge_ms": mgs[i], "route_ms": 0.0,
            "alloc_mb": K.state_bytes(m, sig, t) / 1e6}, got


def route_common(keys, m, t, mode, counts_m, ukeys, sig):
    t0 = time.perf_counter()
    if mode == "C":
        groups = K.plan_equal_groups(ukeys, t)
        loads = K.load_stats(counts_m, groups)
        growner = K.owner_array(m, groups, t)
    elif mode == "E":
        groups, lpt_loads = K.plan_lpt(counts_m, ukeys, t)
        loads = {"loads": [int(x) for x in lpt_loads],
                 "max": int(lpt_loads.max()), "min": int(lpt_loads.min()),
                 "mean": float(lpt_loads.mean()),
                 "imb_pct": float((lpt_loads.max() - lpt_loads.mean()) /
                                  lpt_loads.mean() * 100)}
        growner = K.owner_array(m, groups, t)
    else:  # D
        groups, owner_full = K.plan_hash(m, t)
        loads = K.load_stats(counts_m, groups)
        growner = owner_full
    perm, bounds = K.build_perm(keys, growner, t)
    route_ms = (time.perf_counter() - t0) * 1000
    alloc_mb = K.state_bytes(m, sig, t) / 1e6 + \
        keys.shape[0] * 8 / 1e6
    return groups, growner, perm, bounds, route_ms, loads, alloc_mb


def cell_CDE(keys, vcols, m, sig, t, mode, ukeys):
    t0 = time.perf_counter()
    cnt_m = np.bincount(keys, minlength=m).astype(np.int64)
    gi_ms = (time.perf_counter() - t0) * 1000
    groups, growner, perm, bounds, route_ms, loads, alloc_mb = \
        route_common(keys, m, t, mode, cnt_m, ukeys, sig)
    route_ms += gi_ms
    k = K._IDX[sig]
    walls, qws, busys, mgs = [], [], [], []
    got = None
    for _ in range(3):
        states = K.alloc_state(m, sig, t)

        def mk(w):
            st = states[w]
            j0, j1 = int(bounds[w]), int(bounds[w + 1])
            return lambda: k(keys, *vcols, perm, j0, j1, *st[0], st[1])

        wall, qw, busy = run_agg([mk(w) for w in range(t)], t)
        tmg = time.perf_counter()
        sums_m, counts_m2 = K.merge_gather(states, groups)
        mgs.append((time.perf_counter() - tmg) * 1000)
        walls.append(wall)
        qws, busys = qw, busy
        got = (sums_m, counts_m2)
    i = int(np.argmin(walls))
    return {"walls": walls, "best": walls[i], "qwait_max": max(qws),
            "busy": busys, "merge_ms": mgs[i], "route_ms": route_ms,
            "loads": loads, "alloc_mb": alloc_mb}, got


def main():
    K.warmup()
    # thread-pool warmup (parallel region init outside timing)
    kk = np.arange(20000, dtype=np.int32) % 17
    vv = (np.arange(20000, dtype=np.int32) % 100).astype(np.int32)
    st = K.alloc_state(17, "i", 2)
    ex = pool(2)
    list(ex.map(lambda w: K._SLICE["i"](kk[w::2], vv[w::2], *st[w][0], st[w][1]),
                range(2)))
    cases, pack_ms = load_cases()
    print(f"pack(Q2 radix+densify, 1x, shared)={pack_ms:.0f}ms", flush=True)
    out = {"tag": TAG, "N": N, "ladder": TADDER, "pack_Q2_ms": pack_ms,
           "ncpu": NCPU, "variants": {}}
    for q, (keys, vcols) in cases.items():
        keys = np.ascontiguousarray(keys)
        vcols = [np.ascontiguousarray(v) for v in vcols]
        m = int(keys.max()) + 1
        sig = K.sig_of(vcols)
        ref = fused_singlepass(keys, vcols, m)
        ukeys = ref[0]
        assert ukeys.size == NGOLD[q], (q, ukeys.size)
        dms = dict_stage(q, ref[0], ref[1], ref[2])
        print(f"{q}: m={m} ng={ukeys.size} sig={sig} dict={dms:.0f}ms",
              flush=True)
        gc.collect()
        out["variants"][q] = {"m": m, "sig": sig, "dict_ms": dms,
                              "ngroups": int(ukeys.size), "cells": {}}
        for variant in VARIANTS:
            cells = {}
            base = None
            for t in TADDER:
                c0 = PROC.cpu_times()
                if variant == "A":
                    info, got = cell_A(keys, vcols, m, sig, t)
                elif variant == "B":
                    info, got = cell_B(keys, vcols, m, sig, t)
                else:
                    info, got = cell_CDE(keys, vcols, m, sig, t, variant,
                                         ukeys)
                c1 = PROC.cpu_times()
                cpu_cores = ((c1.user - c0.user) + (c1.system - c0.system)) / \
                    (sum(info["walls"]) / 1000)
                got_c = (ukeys, got[1][ukeys],
                         [s[ukeys] for s in got[0]])
                check(got_c, ref, f"{q}{variant}T{t}")
                if base is None:
                    base = info["best"]
                busy = info["busy"]
                wall = info["best"]
                sp = base / wall
                info.update({
                    "speedup": sp, "eff": sp / t,
                    "cpu_cores": cpu_cores,
                    "rss_gb": PROC.memory_info().rss / 1e9,
                    "idle_mean_ms": float(np.mean([wall - b for b in busy])),
                    "idle_max_ms": float(max(wall - b for b in busy)),
                    "e2e_ms": (pack_ms if q == "Q2" else 0.0) +
                    info["route_ms"] + wall + info["merge_ms"] + dms,
                })
                del info["walls"], info["busy"]
                cells[str(t)] = {k: (round(v, 2) if isinstance(v, float) else v)
                                 for k, v in info.items() if k != "loads"}
                if "loads" in info:
                    cells[str(t)]["loads"] = info["loads"]
                print(f"  {q}-{variant} T={t}: agg={wall:.0f} sp={sp:.2f} "
                      f"eff={sp / t:.2f} mrg={info['merge_ms']:.1f} "
                      f"rte={info['route_ms']:.0f} cores={cpu_cores:.1f} "
                      f"e2e={info['e2e_ms']:.0f}", flush=True)
            out["variants"][q][variant] = cells
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{OUT.name} written", flush=True)
    if FLOAT_DIFFS:
        mx = max(FLOAT_DIFFS.values())
        print(f"float maxdiff over {len(FLOAT_DIFFS)} cells: {mx:.3e}",
              flush=True)
        out["float_maxdiff"] = mx
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
    for p in POOLS.values():
        p.shutdown()


if __name__ == "__main__":
    main()
