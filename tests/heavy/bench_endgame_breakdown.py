# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ENDGAME breakdown (NEW file, no existing bench touched): kernel vs e2e vs
prep, memory traffic, RSS, allocations. Native-only hot path: timed section
calls ONLY native wrappers (fused/pack/pattern/carry); numpy appears solely
as buffer alloc + out-of-measure reference/fallback.

Usage (Git Bash, sequential): NUMFAST_THREADS=<T> .../python tests/heavy/bench_endgame_breakdown.py <T>
Env: NF_SOURCE=csv10M (default).
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

T = int(sys.argv[1]) if len(sys.argv) > 1 else 1
os.environ["NUMFAST_THREADS"] = str(T)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

FORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORK.parent / "app-builder-ponytail"))
sys.path.insert(0, str(FORK))
SOURCE = os.environ.get("NF_SOURCE", "csv10M")

import numpy as np
import psutil

PROC = psutil.Process()


def rss():
    return PROC.memory_info().rss / 1e9


def main():
    from builder import MAIN
    DATA = "C:/App/competitions/H2O/data"
    path = DATA + "/G1_1e7_1e2_0_0.csv"
    N = 10_000_000
    print(f"=== endgame-breakdown {SOURCE} N={N} T={T} ===", flush=True)
    t0 = time.perf_counter()
    import pandas as pd
    df = pd.read_csv(path, usecols=["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"])
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"load {load_ms:.0f}ms RSS {rss():.2f}GB", flush=True)
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias

    def enc(col, prefix="id"):
        gg = a["compile"]([a["ir_encode_pattern"]("c", col, prefix)])
        rr = a["evaluate"](a["optimize"](gg), "cpu", len(col))["result"]
        return np.ascontiguousarray(rr)

    t0 = time.perf_counter()
    K1 = enc(df["id1"].to_numpy(), "id")
    K2 = enc(df["id2"].to_numpy(), "id")
    K3 = enc(df["id3"].to_numpy(), "id")
    enc_ms = (time.perf_counter() - t0) * 1000
    K4 = np.ascontiguousarray((df["id4"].to_numpy().astype(np.int32)))
    K4 = np.ascontiguousarray((K4 - int(K4.min())).astype(np.int32))
    K6 = np.ascontiguousarray((df["id6"].to_numpy().astype(np.int32)))
    K6 = np.ascontiguousarray((K6 - int(K6.min())).astype(np.int32))
    V1 = np.ascontiguousarray(df["v1"].to_numpy().astype(np.int32))
    V2 = np.ascontiguousarray(df["v2"].to_numpy().astype(np.int32))
    V3 = np.ascontiguousarray(df["v3"].to_numpy().astype(np.float64))
    del df
    gc.collect()
    print(f"prep(encode) {enc_ms:.0f}ms RSS {rss():.2f}GB", flush=True)

    jobs = {
        "Q1": ([a["ir_series"]("k", K1), a["ir_series"]("v", V1),
                a["ir_groupby"]("g", "v", "k", "sum")], [("v1", 4, 1)]),
        "Q2": ([a["ir_series"]("c1", K1), a["ir_series"]("c2", K2),
                a["ir_series"]("v", V1),
                a["ir_pack_keys"]("k", "c1", "c2", mode="radix"),
                a["ir_groupby"]("g", "v", "k", "sum")], [("pack", 4, 0), ("v1", 4, 1)]),
        "Q3": ([a["ir_series"]("k", K3), a["ir_series"]("v1", V1),
                a["ir_series"]("v3", V3, "float64"),
                a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                      {"v1": ("sum",), "v3": ("mean",)})],
               [("v1", 4, 1), ("v3", 8, 1)]),
        "Q4": ([a["ir_series"]("k", K4), a["ir_series"]("v1", V1),
                a["ir_series"]("v2", V2), a["ir_series"]("v3", V3, "float64"),
                a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                      {"v1": ("mean",), "v2": ("mean",), "v3": ("mean",)})],
               [("v1", 4, 1), ("v2", 4, 1), ("v3", 8, 1)]),
        "Q5": ([a["ir_series"]("k", K6), a["ir_series"]("v1", V1),
                a["ir_series"]("v2", V2), a["ir_series"]("v3", V3, "float64"),
                a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                      {"v1": ("sum",), "v2": ("sum",), "v3": ("sum",)})],
               [("v1", 4, 1), ("v2", 4, 1), ("v3", 8, 1)]),
    }
    # kernel: direct native calls on resident buffers (no dict/carry-to-dict);
    # e2e: engine execute (incl carry + explicit dict). numpy only allocs here.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "native_cpu", str(FORK / "src" / "Drivers" / "CPU" / "_lib" / "native_cpu.py"))
    nc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nc)

    def kbest(fn, reps=5):
        for _ in range(2):
            fn()
        best = 1e18
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            best = min(best, (time.perf_counter() - t) * 1000)
        return best

    def _as_carry(agg):
        s, c = agg
        return c, s

    def _as_carry_multi(agg):
        ss, c = agg
        return c, ss[0]

    kbufs = {"Q1": (K1, [V1]), "Q2": None, "Q3": (K3, [V1, V3]),
             "Q4": (K4, [V1, V2, V3]), "Q5": (K6, [V1, V2, V3])}
    # traffic: rows * (key4B + sum(col bytes)) read + dense-state write est
    for q, (jj, cols) in jobs.items():
        g = a["optimize"](a["compile"](jj))
        for _ in range(2):
            out = a["evaluate"](g, "cpu", N)
        t = time.perf_counter()
        out = a["evaluate"](g, "cpu", N)
        e2e = (time.perf_counter() - t) * 1000
        kb = kbufs[q]
        if kb is None:  # Q2: pack kernel + 1-col fused kernel
            mpack = int(max(int(K1.max()), int(K2.max())) + 1)
            P0 = nc.pack_i32_direct(K1, K2, mpack)
            m2 = int(P0.max()) + 1
            A0 = nc.fused_sum_count(P0, V1, m2)
            pk = kbest(lambda: nc.pack_i32_direct(K1, K2, mpack))
            agg = kbest(lambda: nc.fused_sum_count(P0, V1, m2))
            kern = pk + agg
            carry = kbest(lambda: nc.carry_build(A0[1], A0[0]))
        else:
            kk, vv = kb
            m = int(kk.max()) + 1
            if len(vv) == 1:
                A0 = nc.fused_sum_count(kk, vv[0], m)
                kern = kbest(lambda: nc.fused_sum_count(kk, vv[0], m))
                carry = kbest(lambda: nc.carry_build(A0[1], A0[0]))
            else:
                A0 = nc.multi_sum_count(kk, vv, m)
                kern = kbest(lambda: nc.multi_sum_count(kk, vv, m))
                carry = kbest(lambda: nc.carry_build(A0[1], A0[0][0]))
        read_mb = N * (4 + sum(cb for _, cb, _ in cols)) / 1e6
        # dense state write: T partials x m groups x 8B (f64) + 8B counts
        m_est = {"Q1": 100, "Q2": 10000, "Q3": 100000, "Q4": 100, "Q5": 100000}[q]
        ncols = len(cols) if q != "Q2" else 1
        alloc_partials = T * (m_est * 8 * ncols + m_est * 8) / 1e6
        print(f"{q}: e2e={e2e:.1f}ms kernel(agg)={kern:.1f} carry={carry:.1f} "
              f"read~{read_mb:.0f}MB partials~{alloc_partials:.2f}MB RSS={rss():.2f}GB",
              flush=True)
        del out
        gc.collect()


if __name__ == "__main__":
    main()
