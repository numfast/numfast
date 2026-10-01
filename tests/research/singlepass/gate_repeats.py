# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate repeats: B-resident Q1-Q5 x6 reps (rep0 cold, best/median of all 6).

Public path only. Goldens asserted every rep. Reports ms + GB/s + RSS.
Usage: python tests/research/singlepass/gate_repeats.py [tag]
"""
import gc
import json
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FORK.parent / "app-builder-ponytail"))
sys.path.insert(0, str(FORK))

import numpy as np
import psutil

CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
N = 10_000_000
GOLD = {
    "Q1": {"ngroups": 100, "total": 29998789},
    "Q2": {"ngroups": 10000, "total": 29998789},
    "Q3": {"ngroups": 100000, "v1": 29998789, "mean3_sum": 4999719.622344427},
    "Q4": {"ngroups": 100, "m1": 299.98798187506526,
           "m2": 799.894179409978, "m3": 4999.766872833688},
    "Q5": {"ngroups": 100000, "s1": 29998789, "s2": 79989360,
           "s3_scaled": 499976651408061},
}
PROC = psutil.Process()
TAG = sys.argv[1] if len(sys.argv) > 1 else "NEW"


def rss():
    return PROC.memory_info().rss / 1e9


def chk_q(res, q):
    g = GOLD[q]
    assert len(res) == g["ngroups"], f"{q}: groups {len(res)}"
    if q in ("Q1", "Q2"):
        tot = int(sum(int(v) for v in res.values()))
        assert tot == g["total"], f"{q}: total {tot}"
        return {"chk": tot}
    if q == "Q3":
        v1 = int(sum(c["v1"]["sum"] for c in res.values()))
        m3 = float(sum(c["v3"]["mean"] for c in res.values()))
        assert v1 == g["v1"], f"Q3 v1 {v1}"
        assert abs(m3 - g["mean3_sum"]) < 1e-6 * abs(g["mean3_sum"]) + 1e-6
        return {"v1": v1, "mean3": m3}
    if q == "Q4":
        s = {c: float(sum(cell[c]["mean"] for cell in res.values()))
             for c in ("v1", "v2", "v3")}
        assert max(abs(s["v1"] - g["m1"]), abs(s["v2"] - g["m2"]),
                   abs(s["v3"] - g["m3"])) < 1e-9
        return s
    s1 = int(sum(c["v1"]["sum"] for c in res.values()))
    s2 = int(sum(c["v2"]["sum"] for c in res.values()))
    s3 = float(sum(c["v3"]["sum"] for c in res.values()))
    assert s1 == g["s1"] and s2 == g["s2"], f"Q5 {s1} {s2}"
    assert int(round(s3 * 1e6)) == g["s3_scaled"]
    return {"s1": s1, "s2": s2, "s3_scaled": int(round(s3 * 1e6))}


def main():
    from builder import MAIN
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    import pandas as pd
    import pyarrow as pa
    print(f"=== gate repeats {TAG} ===", flush=True)
    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"])
    assert len(df) == N
    s_id1 = pa.array(df["id1"].to_numpy(), type=pa.string())
    s_id2 = pa.array(df["id2"].to_numpy(), type=pa.string())
    s_id3 = pa.array(df["id3"].to_numpy(), type=pa.string())
    id4 = df["id4"].to_numpy().astype(np.int32)
    id6 = df["id6"].to_numpy().astype(np.int32)
    v1 = df["v1"].to_numpy().astype(np.int32)
    v2 = df["v2"].to_numpy().astype(np.int32)
    v3 = df["v3"].to_numpy().astype(np.float64)
    del df
    gc.collect()

    res = {}
    for name, col in (("id1", {"values": s_id1, "prefix": "id"}),
                      ("id2", {"values": s_id2, "prefix": "id"}),
                      ("id3", {"values": s_id3, "prefix": "id"})):
        res[name] = a["resident_prepare"]({name: col})[name]
    c = a["resident_prepare"]({
        "id4": {"values": id4, "dtype": "int32"}, "id6": {"values": id6, "dtype": "int32"},
        "v1": {"values": v1, "dtype": "int32"}, "v2": {"values": v2, "dtype": "int32"},
        "v3": {"values": v3, "dtype": "float64"}})
    for k, vv in c.items():
        res[k] = vv
    K1, K2, K3 = res["id1"]["codes"], res["id2"]["codes"], res["id3"]["codes"]
    K4, K6 = res["id4"]["codes"], res["id6"]["codes"]
    V1, V2, V3 = res["v1"]["codes"], res["v2"]["codes"], res["v3"]["codes"]
    print(f"prep done RSS {rss():.2f}GB", flush=True)

    queries = {
        "Q1": [a["ir_series"]("k", K1), a["ir_series"]("v", V1),
               a["ir_groupby"]("g", "v", "k", "sum")],
        "Q2": [a["ir_series"]("c1", K1), a["ir_series"]("c2", K2),
               a["ir_series"]("v", V1), a["ir_pack_keys"]("k", "c1", "c2"),
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
    out = {"tag": TAG, "N": N}
    for q, jobs in queries.items():
        g = a["optimize"](a["compile"](jobs))
        ts = []
        for rep in range(6):
            t = time.perf_counter()
            r = a["evaluate"](g, "cpu", N)
            dt = (time.perf_counter() - t) * 1000
            ts.append(dt)
            chk_q(r["result"], q)
        ts_sorted = sorted(ts)
        best, med = ts_sorted[0], ts_sorted[len(ts_sorted) // 2]
        thr = N / best / 1e3
        print(f"{q}: reps={[f'{t:.0f}' for t in ts]} best={best:.0f} "
              f"median={med:.0f}ms thr={thr:.0f}k rows/s RSS {rss():.2f}GB",
              flush=True)
        out[q] = {"reps_ms": ts, "best_ms": best, "median_ms": med,
                  "rss_gb": rss()}
    with open(FORK / "tests" / "research" / "singlepass" /
               f"gate_{TAG}_10M.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"gate_{TAG}_10M.json written", flush=True)


if __name__ == "__main__":
    main()
