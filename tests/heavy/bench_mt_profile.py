# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MT stage profile 10M (NEW file; existing benches untouched).

Snapshot scratch/snap_G1_1e7_1e2_0_0 loaded ONCE, then sequential
T-ladder x Q1-Q5 in-process (carry e2e + gi/agg/carry/merge stages from
g#groupindex + dict e2e). One process, strictly sequential.
Usage (Git Bash): timeout ... python tests/heavy/bench_mt_profile.py [T...]
Default ladder: 1 8 12 16 (trimmed; see report for justification).
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

T0 = time.perf_counter()
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

FORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, "C:/App/numfast/app-builder-ponytail")
sys.path.insert(0, "C:/App/numfast/numfast-ponytail")

import numpy as np
import psutil

PROC = psutil.Process()
CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
N = 10_000_000
SNAP = FORK / "scratch" / "snap_G1_1e7_1e2_0_0"
COLS = ("K1", "K2", "K3", "K4", "K6", "V1", "V2", "V3")
TS = tuple(int(a) for a in sys.argv[1:]) or (1, 8, 12, 16)
QS = ("Q1", "Q2", "Q3", "Q4", "Q5")
OUT = FORK / "tests" / "heavy" / "bench_mt_profile.json"
GOLD_TOTAL = 29998789


def rss():
    return PROC.memory_info().rss / 1e9


def best_of(fn, reps=3):
    best, out = 1e18, None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, (time.perf_counter() - t) * 1000)
    return out, best


def main():
    from builder import MAIN
    print(f"=== mt-profile 10M T={list(TS)} pid={os.getpid()} ===", flush=True)
    a = MAIN["build"]("C:/App/numfast/numfast-ponytail").alias
    arr = {k: np.load(str(SNAP / f"{k}.npy")) for k in COLS}
    assert all(arr[k].shape == (N,) for k in COLS)
    fp = {k: int(arr[k].astype(np.int64).sum()) for k in COLS}
    assert fp["V1"] == GOLD_TOTAL, fp
    print(f"snap ok RSS {rss():.2f}GB", flush=True)
    K1, K2, K3, K4, K6, V1, V2, V3 = (arr[k] for k in COLS)

    JD = {
        "Q1": [a["ir_series"]("k", K1), a["ir_series"]("v", V1),
               a["ir_groupby"]("g", "v", "k", "sum")],
        "Q2": [a["ir_series"]("c1", K1), a["ir_series"]("c2", K2),
               a["ir_series"]("v", V1),
               a["ir_pack_keys"]("k", "c1", "c2", mode="radix"),
               a["ir_groupby"]("g", "v", "k", "sum")],
        "Q3": [a["ir_series"]("k", K3), a["ir_series"]("v1", V1),
               a["ir_series"]("v3", V3, "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                     {"v1": ("sum",), "v3": ("mean",)})],
        "Q4": [a["ir_series"]("k", K4), a["ir_series"]("v1", V1),
               a["ir_series"]("v2", V2), a["ir_series"]("v3", V3, "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                     {"v1": ("mean",), "v2": ("mean",),
                                      "v3": ("mean",)})],
        "Q5": [a["ir_series"]("k", K6), a["ir_series"]("v1", V1),
               a["ir_series"]("v2", V2), a["ir_series"]("v3", V3, "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                     {"v1": ("sum",), "v2": ("sum",),
                                      "v3": ("sum",)})],
    }
    compc, compd = {}, {}
    for q, jj in JD.items():
        compd[q] = a["optimize"](a["compile"](jj))
        last = jj[-1]
        p = dict(last["params"])
        p["result"] = "carry"
        nn = dict(last)
        nn["params"] = p
        compc[q] = a["optimize"](a["compile"](jj[:-1] + [nn]))

    os.environ["NUMFAST_THREADS"] = "1"
    bufs = a["cpu_execute"](compc["Q1"]["nodes"])
    assert int(bufs["g#carry"].sums["v"].sum()) == GOLD_TOTAL
    del bufs
    gc.collect()

    matrix = {}
    for T in TS:
        os.environ["NUMFAST_THREADS"] = str(T)
        row = {}
        for q in QS:
            bufs, cms = best_of(lambda: a["cpu_execute"](compc[q]["nodes"]), 3)
            gi = dict(bufs.get("g#groupindex", {}))
            assert gi.get("threads") == T, (q, gi)
            chk = int(bufs["g#carry"].sums[
                "v" if q in ("Q1", "Q2") else ("v1")].sum())
            assert chk == GOLD_TOTAL, (q, chk)
            stages = {k: gi.get(k) for k in
                      ("gi_ms", "agg_ms", "carry_ms", "merge_ms", "backend")}
            del bufs
            gc.collect()
            res, dms = best_of(
                lambda: a["evaluate"](compd[q], "cpu", N)["result"], 3)
            del res
            gc.collect()
            row[q] = {"carry_ms": round(cms, 1), "dict_ms": round(dms, 1),
                      **{k: (round(v, 1) if isinstance(v, float) else v)
                         for k, v in stages.items()}}
            print(f"T={T} {q}: carry={cms:.0f}ms(dict {dms:.0f}) "
                  f"agg={stages.get('agg_ms'):.0f} "
                  f"merge={stages.get('merge_ms'):.0f} "
                  f"carry_st={stages.get('carry_ms'):.0f} "
                  f"gi={stages.get('gi_ms'):.0f} RSS {rss():.2f}GB",
                  flush=True)
        matrix[str(T)] = row
    with open(str(OUT), "w") as f:
        json.dump({"T": list(TS), "matrix": matrix,
                   "rss_gb": rss(), "gold": "EXACT/PASS"}, f, indent=1)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
