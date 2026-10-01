# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research: group weights as OPTIONAL schema statistics (NOT into NFS).

Reuses prior mission numbers (exact 12B/group, u8-quant 5B/group) + re-measures
on real H2O 10M id1/id3. Measures: index size, index load cost, planner benefit
(dense/hash/sorted choice + row-vs-group-owned distribution), amortization over
1/10/100 queries, statistics modes none/basic/group_weights/auto, block scalar
metadata vs global weights. No prod change. No NFS change. No push.
Usage: python tests/research/group_ownership/bench_weights_opt.py
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

FORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FORK.parent / "app-builder-ponytail"))
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / "tests" / "research" / "group_ownership"))

import numpy as np

import ownership as O

CSV = os.environ.get("MT_CSV", "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv")
N = 10_000_000
SEED = 42
OUT = FORK / "tests" / "research" / "group_ownership" / "results_weights_opt.json"
TMP = FORK / "tests" / "research" / "group_ownership" / ".tmp_gindex.npy"


def best_of(fn, reps=5):
    ts = []
    out = None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t) * 1000)
    return min(ts), float(np.median(ts)), out


def load_codes():
    from builder import MAIN
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    import pandas as pd
    import pyarrow as pa
    t = time.perf_counter()
    df = pd.read_csv(CSV, usecols=["id1", "id3"])
    csv_ms = (time.perf_counter() - t) * 1000
    assert len(df) == N
    s1 = pa.array(df["id1"].to_numpy(), type=pa.string())
    s3 = pa.array(df["id3"].to_numpy(), type=pa.string())
    del df
    gc.collect()
    t = time.perf_counter()
    r1 = a["resident_prepare"]({"id1": {"values": s1, "prefix": "id"}})["id1"]
    enc1_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    r3 = a["resident_prepare"]({"id3": {"values": s3, "prefix": "id"}})["id3"]
    enc3_ms = (time.perf_counter() - t) * 1000
    return (np.ascontiguousarray(r1["codes"]), np.ascontiguousarray(r3["codes"]),
            csv_ms, enc1_ms, enc3_ms)


def main():
    K1, K3, csv_ms, enc1_ms, enc3_ms = load_codes()
    print(f"load: csv={csv_ms:.0f}ms enc_id1={enc1_ms:.0f}ms enc_id3={enc3_ms:.0f}ms",
          flush=True)
    out = {"tag": "weights_opt", "N": N, "seed": SEED,
           "load": {"csv_ms": csv_ms, "enc_id1_ms": enc1_ms, "enc_id3_ms": enc3_ms}}
    idx = {}
    for name, K in (("id1", K1), ("id3", K3)):
        m = int(K.max()) + 1
        t = time.perf_counter()
        cnt = np.bincount(K, minlength=m).astype(np.int64)
        build_ms = (time.perf_counter() - t) * 1000
        assert int(cnt.sum()) == N
        ukeys = np.flatnonzero(cnt > 0).astype(np.int64)
        Kg = int(ukeys.size)
        gi = O.build_gindex_exact(ukeys.astype(np.int32), cnt[ukeys])
        q = O.build_gindex_quantized(cnt[ukeys])
        # distribution shape (planner signal)
        c = cnt[ukeys].astype(np.float64)
        cv = float(c.std() / c.mean())
        # file roundtrip = index load cost proxy (real file, removed after)
        arr = np.empty(Kg, dtype=[("code", "<i4"), ("w", "<i8")])
        arr["code"] = ukeys.astype(np.int32)
        arr["w"] = cnt[ukeys]
        t = time.perf_counter()
        np.save(str(TMP), arr)
        save_ms = (time.perf_counter() - t) * 1000
        fbytes = int(TMP.stat().st_size)
        t = time.perf_counter()
        back = np.load(str(TMP))
        load_ms = (time.perf_counter() - t) * 1000
        assert bool((back["w"] == arr["w"]).all())
        TMP.unlink()
        idx[name] = {"K": Kg, "m": m, "build_bincount_ms": round(build_ms, 2),
                     "exact_bytes": gi["raw_bytes"],
                     "exact_bpg": round(gi["bytes_per_group"], 3),
                     "quant_bytes": q["raw_bytes"],
                     "quant_bpg": round(q["bytes_per_group"], 3),
                     "quant_max_rel": round(q["max_rel_err"], 4),
                     "file_bytes": fbytes, "save_ms": round(save_ms, 2),
                     "load_ms": round(load_ms, 2),
                     "cv": round(cv, 4), "min": int(c.min()), "max": int(c.max()),
                     "median": float(np.median(c))}
        print(f"{name}: K={Kg} exact={gi['raw_bytes']}B "
              f"({gi['bytes_per_group']:.2f}B/g) quant={q['raw_bytes']}B "
              f"build={build_ms:.1f}ms load={load_ms:.1f}ms cv={cv:.4f}", flush=True)
        del cnt, ukeys, arr, back
        gc.collect()
    out["index"] = idx

    # ---- planner benefit: distribution quality with/without weights ----
    # uniform real id3 + synthetic skewed (zipf) at K=10K/1M rows (cheap)
    rng = np.random.default_rng(SEED)
    skew_w = rng.zipf(1.5, size=10000).astype(np.int64)
    skew_w = np.minimum(skew_w, 5000)
    skew_u = np.arange(10000, dtype=np.int64)
    plan = {}
    for label, uu, ww, full, P in (("id3_uniform_16T", None, None, None, 16),
                                ("skew_zipf16T", skew_u, skew_w, skew_w, 16)):
        if uu is None:  # real id3 weights, recompute cheap bincount once
            full = np.bincount(K3, minlength=int(K3.max()) + 1).astype(np.int64)
            uu = np.flatnonzero(full > 0).astype(np.int64)
            ww = full[uu]
        gA = O.plan_A_equal_number(uu, P)
        sA = O.load_stats(full, gA)
        (gB, iB), (gD, iD) = O.plan_B_contig_weighted(uu, ww, P), \
            O.plan_D_sorted_lpt(uu, ww, P)
        okA, _ = O.check_disjoint(gA, uu.size)
        okB, _ = O.check_disjoint(gB, uu.size)
        okD, _ = O.check_disjoint(gD, uu.size)
        assert okA and okB and okD
        sB = O.load_stats(full, gB)
        sD = O.load_stats(full, gD)
        lb = O.lower_bound_unsplittable(ww, P)
        plan[label] = {
            "equal_nogroups_imb": round(sA["imb_pct"], 2),
            "contig_weighted_imb": round(sB["imb_pct"], 2),
            "lpt_imb": round(sD["imb_pct"], 2),
            "sched_B_ms": round(iB["sched_ms"], 2),
            "sched_D_ms": round(iD["sched_ms"], 2),
            "unsplittable_lb": lb, "mean_load": round(sA["mean"], 1)}
        print(f"{label}: imb_noW={sA['imb_pct']:.2f}% imb_B={sB['imb_pct']:.2f}% "
              f"imb_LPT={sD['imb_pct']:.2f}% sched={iB['sched_ms']:.1f}/"
              f"{iD['sched_ms']:.1f}ms", flush=True)
    out["planner"] = plan

    # ---- amortization: build once -> use on Nq queries ----
    # per-query use cost = weighted sched (LPT on id3 K=100K, measured)
    cnt = np.bincount(K3, minlength=int(K3.max()) + 1).astype(np.int64)
    uu = np.flatnonzero(cnt > 0).astype(np.int64)
    ww = cnt[uu]
    del cnt
    _, sched_med, _ = best_of(lambda: O.plan_D_sorted_lpt(uu, ww, 16), reps=3)
    build = idx["id3"]["build_bincount_ms"]
    amort = {}
    for nq in (1, 10, 100):
        amort[str(nq)] = round(build / nq + sched_med, 2)
    # reference scale: MT-measured 16T agg walls (results_mt10M.json, variant A)
    amort["ref_agg16T_ms"] = {"Q2": 3.89, "Q3": 14.1, "Q5": 25.02}
    amort["build_ms"] = build
    amort["sched_LPT_med_ms"] = round(sched_med, 2)
    out["amortization"] = amort
    print(f"amort: build={build:.1f}ms sched={sched_med:.1f}ms "
          f"per-q(1/10/100)={amort['1']}/{amort['10']}/{amort['100']}ms", flush=True)

    # ---- block scalar metadata (10 x 1M) vs global weights ----
    NB = 10
    bmeta, bbytes, t = [], 0, time.perf_counter()
    for b in range(NB):
        seg = K3[b * (N // NB):(b + 1) * (N // NB)]
        u = np.unique(seg)
        sm = {"rows": int(seg.size), "unique": int(u.size),
              "min": int(seg.min()), "max": int(seg.max()),
              "sorted": bool((np.diff(seg.astype(np.int64)) >= 0).all()),
              "null": 0}
        bmeta.append(sm)
        bbytes += 6 * 8  # rows/unique/min/max/sorted/null as i64
    block_ms = (time.perf_counter() - t) * 1000
    nunique_spread = max(x["unique"] for x in bmeta) - min(x["unique"] for x in bmeta)
    out["blockmeta"] = {
        "nblocks": NB, "bytes": bbytes, "build_ms": round(block_ms, 1),
        "global_exact_bytes": idx["id3"]["exact_bytes"],
        "ratio_global_over_block": round(idx["id3"]["exact_bytes"] / bbytes, 1),
        "unique_spread": int(nunique_spread),
        "all_sorted": all(x["sorted"] for x in bmeta)}
    print(f"blockmeta: {bbytes}B in {block_ms:.0f}ms vs global "
          f"{idx['id3']['exact_bytes']}B "
          f"({idx['id3']['exact_bytes'] / bbytes:.0f}x); "
          f"unique_spread={nunique_spread}", flush=True)

    # ---- statistics modes cost table ----
    out["modes"] = {
        "none": {"bytes": 0, "cost": "0ms; planner blind (adaptive/row-partition "
                 "fallback); best when uniform + row-partition (A/B)"},
        "basic": {"bytes_per_col": 48, "cost": "~0ms piggyback on encode "
                  "(K/min/max/sorted/null); picks dense/hash/sorted + m; "
                  "covers uniform H2O fully"},
        "group_weights": {
            "exact_bytes": {"id1": idx["id1"]["exact_bytes"],
                            "id3": idx["id3"]["exact_bytes"]},
            "quant_bytes": {"id1": idx["id1"]["quant_bytes"],
                            "id3": idx["id3"]["quant_bytes"]},
            "build_ms": {"id1": idx["id1"]["build_bincount_ms"],
                         "id3": idx["id3"]["build_bincount_ms"]},
            "use": "exact LPT / weighted split for group-owned (C/D/E) on "
                   "skewed data; pays only if skew*agg_gain > build/N+sched"},
        "auto": {"rule": "basic always; +weights iff (K<=4096 and cv>0.05) or "
                 "group-owned dispatch requested; quant-u8 default, exact on "
                 "demand for LPT final cut"}}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{OUT.name} written", flush=True)


if __name__ == "__main__":
    main()
