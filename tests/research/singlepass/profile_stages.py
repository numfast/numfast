# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage profile of prod B-resident path on H2O 10M (research, read-only).

Measures per Q: series/pack ms, gi_ms, agg_ms, dict-materialize ms, total.
Plus micro-checks: bincount int32-keys/weights exactness, dict-build cost.
No prod change. Single-thread (OMP/NUMBA single).
"""
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

FORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FORK.parent / "app-builder-ponytail"))
sys.path.insert(0, str(FORK))

import numpy as np

CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
N = 10_000_000


def build_kernel():
    from builder import MAIN
    return MAIN["build"](str(FORK))


def main():
    import pandas as pd
    import pyarrow as pa
    kernel = build_kernel()
    a = kernel.alias

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

    def run(jobs):
        g = a["optimize"](a["compile"](jobs))
        t = time.perf_counter()
        r = a["evaluate"](g, "cpu", N)
        tot = (time.perf_counter() - t) * 1000
        return r["result"], r, tot

    # warm up numba sortedness probe + show strategies
    k1 = res["id1"]["codes"]
    vv1 = res["v1"]["codes"]
    jobs_q1 = [a["ir_series"]("k", k1), a["ir_series"]("v", vv1),
               a["ir_groupby"]("g", "v", "k", "sum")]
    r0, raw0, _ = run(jobs_q1)
    print("warm strategies:", {k: v for k, v in raw0.get("result", {}).items()} if False else "")
    # sidecars live in bufs, not result; re-run via cpu_execute to grab them
    from builder import MAIN as _M  # noqa
    print("Q1 chk:", sum(r0.values()), "ngroups:", len(r0), flush=True)

    queries = {
        "Q1": [a["ir_series"]("k", k1), a["ir_series"]("v", vv1),
               a["ir_groupby"]("g", "v", "k", "sum")],
        "Q2": [a["ir_series"]("c1", k1), a["ir_series"]("c2", res["id2"]["codes"]),
               a["ir_series"]("v", vv1),
               a["ir_pack_keys"]("k", "c1", "c2"),
               a["ir_groupby"]("g", "v", "k", "sum")],
        "Q3": [a["ir_series"]("k", res["id3"]["codes"]),
               a["ir_series"]("v1", vv1),
               a["ir_series"]("v3", res["v3"]["codes"], "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                     {"v1": ("sum",), "v3": ("mean",)})],
        "Q4": [a["ir_series"]("k", res["id4"]["codes"]),
               a["ir_series"]("v1", vv1),
               a["ir_series"]("v2", res["v2"]["codes"]),
               a["ir_series"]("v3", res["v3"]["codes"], "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                     {"v1": ("mean",), "v2": ("mean",),
                                      "v3": ("mean",)})],
        "Q5": [a["ir_series"]("k", res["id6"]["codes"]),
               a["ir_series"]("v1", vv1),
               a["ir_series"]("v2", res["v2"]["codes"]),
               a["ir_series"]("v3", res["v3"]["codes"], "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                     {"v1": ("sum",), "v2": ("sum",),
                                      "v3": ("sum",)})],
    }
    for q, jobs in queries.items():
        # 2 timed reps
        for rep in range(2):
            t = time.perf_counter()
            g = a["compile"](jobs)
            t_compile = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            g = a["optimize"](g)
            t_opt = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            rr = a["evaluate"](g, "cpu", N)
            t_exec = (time.perf_counter() - t) * 1000
            bufs = rr.get("buffers", rr)
            # evaluate returns {"result":...}; sidecars not exposed -> call cpu path directly
            print(f"{q} rep{rep}: compile={t_compile:.2f} opt={t_opt:.2f} "
                  f"exec={t_exec:.0f}ms total={t_compile+t_opt+t_exec:.0f}ms "
                  f"ngroups={len(rr['result'])}", flush=True)

    # ---- micro: bincount int32 keys/weights exactness ----
    rng = np.random.default_rng(42)
    kk = rng.integers(0, 100, size=1_000_000).astype(np.int32)
    w32 = rng.integers(0, 100, size=1_000_000).astype(np.int32)
    ref = np.bincount(kk.astype(np.int64), weights=w32.astype(np.int64), minlength=100)
    got = np.bincount(kk, weights=w32, minlength=100)
    print("bincount int32/int32 exact:", bool((ref == got).all()), got.dtype, flush=True)
    wf = (w32.astype(np.float64) + 0.5)
    ref2 = np.bincount(kk.astype(np.int64), weights=wf, minlength=100)
    got2 = np.bincount(kk, weights=wf, minlength=100)
    print("bincount int32-keys/float-weights exact:",
          bool((ref2 == got2).all()), got2.dtype, flush=True)
    # int32 keys with values > int32 range? keys are codes, small. weights int32 sum fits int64.
    print("bincount out dtype int-weights:", got.dtype, flush=True)

    # ---- micro: dict materialization cost at 100K groups ----
    ng = 100000
    uk = np.arange(ng, dtype=np.int64)
    sums = np.arange(ng, dtype=np.int64)
    counts = np.ones(ng, dtype=np.int64)
    t = time.perf_counter()
    d = {}
    for i in range(uk.size):
        d[int(uk[i])] = sums[i].item()
    print(f"dict-build 100K single-col: {(time.perf_counter()-t)*1000:.0f}ms", flush=True)
    t = time.perf_counter()
    d2 = {}
    for i in range(uk.size):
        key = int(uk[i])
        cell = {}
        cell["sum"] = sums[i].item()
        cell["count"] = int(counts[i])
        cell["mean"] = float(sums[i] / counts[i])
        d2[key] = cell
    print(f"dict-build 100K multi-cell: {(time.perf_counter()-t)*1000:.0f}ms", flush=True)
    t = time.perf_counter()
    d3 = dict(zip(map(int, uk), map(int, sums)))
    print(f"dict-zip 100K: {(time.perf_counter()-t)*1000:.0f}ms same={d3==d}", flush=True)


if __name__ == "__main__":
    main()
