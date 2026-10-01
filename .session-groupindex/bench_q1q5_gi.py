#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SESSION ARTIFACT (not src, NOT committed): OLD vs NEW Q1-Q5 10M end-to-end.

OLD run (current np.unique path): measures per-Q
  gi_ms  = sorted-precheck(np.all) + np.unique(return_inverse)   [faithful OLD _group_index]
  agg_ms = bincount sums+counts per value col                    [faithful OLD _fused_sums_counts]
  total  = compile+optimize+cpu_execute via PUBLIC production path (cold+warm)
  strategy sidecar (#strategy) read from production bufs.
NEW run (after integration): same totals + #groupindex sidecar
  {strategy, reason, gi_ms, agg_ms} timed INSIDE production cpu path.
Decision fn is pure in keys -> direct group_index calls on same arrays give
the same strategy as inside the graph (reported for transparency).

Usage (Git Bash):
  /c/Users/Mikech/AppData/Local/Programs/Python/Python314/python.exe \
    .session-groupindex/bench_q1q5_gi.py [--tag OLD|NEW]
Writes .session-groupindex/results_q1q5_<TAG>.json
"""
import gc
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORK = HERE.parent
sys.path.insert(0, str(FORK.parent / "app-builder-ponytail"))
sys.path.insert(0, str(FORK))

CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
N = 10_000_000
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "OLD"

import numpy as np
import psutil

PROC = psutil.Process()


def rss():
    return PROC.memory_info().rss / 1e9


def build_kernel():
    from builder import MAIN
    return MAIN["build"](str(FORK))


GOLD = {
    "Q1": {"ngroups": 100, "total": 29998789},
    "Q2": {"ngroups": 10000, "total": 29998789},
    "Q3": {"ngroups": 100000, "v1": 29998789, "mean3_sum": 4999719.622344427},
    "Q4": {"ngroups": 100, "m1": 299.98798187506526,
           "m2": 799.894179409978, "m3": 4999.766872833688},
    "Q5": {"ngroups": 100000, "s1": 29998789, "s2": 79989360,
           "s3_scaled": 499976651408061},
}


def old_gi_agg(keys, val_cols):
    """Faithful OLD arithmetic: precheck+unique (gi) + bincounts (agg)."""
    t0 = time.perf_counter()
    _ = bool(np.all(keys[1:] >= keys[:-1])) if keys.size > 1 else True
    ukeys, inverse = np.unique(keys, return_inverse=True)
    gi_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    ng = ukeys.size
    for v in val_cols:
        if np.issubdtype(v.dtype, np.integer):
            w = v.astype(np.int64, copy=False)
            s = np.bincount(inverse, weights=w, minlength=ng).astype(np.int64)
        else:
            w = v.astype(np.float64, copy=False)
            s = np.bincount(inverse, weights=w, minlength=ng)
        c = np.bincount(inverse, minlength=ng).astype(np.int64)
        assert s.size == ng and c.size == ng
    agg_ms = (time.perf_counter() - t1) * 1000
    return gi_ms, agg_ms, ukeys.size


def run_graph_full(a, jobs, n):
    st = {}
    s = time.perf_counter()
    g = a["compile"](jobs)
    st["compile"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    g = a["optimize"](g)
    st["optimize"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    bufs = a["cpu_execute"](g["nodes"])
    st["execute"] = (time.perf_counter() - s) * 1000
    st["total"] = st["compile"] + st["optimize"] + st["execute"]
    last = g["outputs"][-1] if g["outputs"] else g["nodes"][-1]["out"]
    side = {k: v for k, v in bufs.items() if "#strategy" in k or "#groupindex" in k}
    for k, v in side.items():
        if isinstance(v, dict):
            side[k] = {kk: (float(vv) if isinstance(vv, float) else vv)
                       for kk, vv in v.items()}
        else:
            side[k] = v
    return bufs[last], st, side


def main():
    import pandas as pd
    import pyarrow as pa

    print(f"=== Q1-Q5 gi gate [{TAG}] N={N} ===", flush=True)
    print(f"RSS start {rss():.2f}GB", flush=True)
    kernel = build_kernel()
    a = kernel.alias

    t0 = time.perf_counter()
    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"])
    print(f"csv_load {(time.perf_counter()-t0)*1000:.0f}ms RSS {rss():.2f}GB", flush=True)
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

    t0 = time.perf_counter()
    res = {}
    res.update(a["resident_prepare"]({"id1": {"values": s_id1, "prefix": "id"}}))
    res.update(a["resident_prepare"]({"id2": {"values": s_id2, "prefix": "id"}}))
    res.update(a["resident_prepare"]({"id3": {"values": s_id3, "prefix": "id"}}))
    res.update(a["resident_prepare"]({
        "id4": {"values": id4, "dtype": "int32"}, "id6": {"values": id6, "dtype": "int32"},
        "v1": {"values": v1, "dtype": "int32"}, "v2": {"values": v2, "dtype": "int32"},
        "v3": {"values": v3, "dtype": "float64"}}))
    print(f"resident_prepare {(time.perf_counter()-t0)*1000:.0f}ms RSS {rss():.2f}GB", flush=True)
    del s_id1, s_id2, s_id3, id4, id6, v1, v2, v3
    gc.collect()
    k1, k2, k3 = res["id1"]["codes"], res["id2"]["codes"], res["id3"]["codes"]
    id4c, id6c = res["id4"]["codes"], res["id6"]["codes"]
    v1c, v2c, v3c = res["v1"]["codes"], res["v2"]["codes"], res["v3"]["codes"]
    # Q2 packed keys (same pack_keys semantics as production "pack" mode)
    hi = k1.astype(np.int64, copy=False)
    lo = k2.astype(np.int64, copy=False) & np.int64(0xFFFFFFFF)
    k12 = (hi << np.int64(32)) | lo

    queries = {
        "Q1": {"keys": k1, "vals": [v1c],
               "jobs": lambda: [a["ir_series"]("k", k1), a["ir_series"]("v", v1c),
                                a["ir_groupby"]("g", "v", "k", "sum")]},
        "Q2": {"keys": k12, "vals": [v1c],
               "jobs": lambda: [a["ir_series"]("c1", k1), a["ir_series"]("c2", k2),
                                a["ir_series"]("v", v1c),
                                a["ir_pack_keys"]("k", "c1", "c2"),
                                a["ir_groupby"]("g", "v", "k", "sum")]},
        "Q3": {"keys": k3, "vals": [v1c, v3c],
               "jobs": lambda: [a["ir_series"]("k", k3),
                                a["ir_series"]("v1", v1c),
                                a["ir_series"]("v3", v3c, "float64"),
                                a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                                      {"v1": ("sum",), "v3": ("mean",)})]},
        "Q4": {"keys": id4c, "vals": [v1c, v2c, v3c],
               "jobs": lambda: [a["ir_series"]("k", id4c),
                                a["ir_series"]("v1", v1c), a["ir_series"]("v2", v2c),
                                a["ir_series"]("v3", v3c, "float64"),
                                a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                                      {"v1": ("mean",), "v2": ("mean",),
                                                       "v3": ("mean",)})]},
        "Q5": {"keys": id6c, "vals": [v1c, v2c, v3c],
               "jobs": lambda: [a["ir_series"]("k", id6c),
                                a["ir_series"]("v1", v1c), a["ir_series"]("v2", v2c),
                                a["ir_series"]("v3", v3c, "float64"),
                                a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                                      {"v1": ("sum",), "v2": ("sum",),
                                                       "v3": ("sum",)})]},
    }

    out = {"tag": TAG, "N": N, "queries": {}}
    for q, spec in queries.items():
        keys, vals = spec["keys"], spec["vals"]
        gi_ms, agg_ms, ng = old_gi_agg(np.asarray(keys), [np.asarray(v) for v in vals])
        r0 = rss()
        r, st, side = run_graph_full(a, spec["jobs"](), N)
        r1 = rss()
        # goldens (same contracts as bench_resident_10M B mode)
        if q in ("Q1", "Q2"):
            tot = int(sum(int(v) for v in r.values()))
            assert len(r) == GOLD[q]["ngroups"] and tot == GOLD[q]["total"], (q, len(r), tot)
            chk = tot
        elif q == "Q3":
            m3 = float(sum(cell["v3"]["mean"] for cell in r.values()))
            assert len(r) == 100000 and abs(m3 - GOLD[q]["mean3_sum"]) < 1e-6 * abs(GOLD[q]["mean3_sum"]) + 1e-6
            chk = m3
        elif q == "Q4":
            sums = {c: float(sum(cell[c]["mean"] for cell in r.values())) for c in ("v1", "v2", "v3")}
            md = max(abs(sums["v1"] - GOLD[q]["m1"]), abs(sums["v2"] - GOLD[q]["m2"]), abs(sums["v3"] - GOLD[q]["m3"]))
            assert len(r) == 100 and md < 1e-9
            chk = sums
        else:
            s1 = int(sum(cell["v1"]["sum"] for cell in r.values()))
            s2 = int(sum(cell["v2"]["sum"] for cell in r.values()))
            s3s = int(round(float(sum(cell["v3"]["sum"] for cell in r.values())) * 1e6))
            assert len(r) == 100000 and s1 == 29998789 and s2 == 79989360 and s3s == 499976651408061
            chk = {"s1": s1, "s2": s2, "s3_scaled": s3s}
        r, st2, side2 = run_graph_full(a, spec["jobs"](), N)  # warm
        del r
        gc.collect()
        row = {"ngroups": ng, "old_gi_ms": gi_ms, "old_agg_ms": agg_ms,
               "cold_ms": st["total"], "cold_stages": st, "sidecar_cold": side,
               "warm_ms": st2["total"], "sidecar_warm": side2,
               "rss_delta_gb": r1 - r0, "chk": chk}
        out["queries"][q] = row
        print(f"{q}: groups={ng} OLDgi={gi_ms:.0f}ms OLDagg={agg_ms:.0f}ms "
              f"cold={st['total']:.0f}ms (exec {st['execute']:.0f}) warm={st2['total']:.0f}ms "
              f"side={side} rss+{r1-r0:.2f}GB", flush=True)
    out["rss_end_gb"] = rss()
    p = HERE / f"results_q1q5_{TAG}.json"
    p.write_text(json.dumps(json.loads(json.dumps(out, default=float)), indent=1))
    print(f"DONE -> {p} RSS {rss():.2f}GB", flush=True)


if __name__ == "__main__":
    main()
