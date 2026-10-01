# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H2O 10M G1_1e7_1e2_0_0 Q1-Q5: same graph, same Planner, same Runtime —
CPU driver vs GPU driver (NEW file; existing benches untouched).

Graphs are verbatim copies of bench_cpu_modes_10M.py Q1-Q5 (same GOLD/chk).
Q1/Q2 run on both backends (Q2 label spaces differ by construction:
CPU pack=bitpack int64, GPU pack=int32-direct DELTA-2 — equivalence is
checked via (c1,c2) mapping in THIS harness, never in the engine).
Q3/Q4/Q5 carry v3 float64 -> GPU raises the honest MapF64 guard (DELTA-6.4);
recorded as guard_error, never faked.

One process, sequential, cold then warm per (Q, backend). No 1B.

Usage (Git Bash):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    timeout 550 /c/Users/Mikech/AppData/Local/Programs/Python/Python314/python.exe \
    tests/heavy/bench_h2o_gpu_10M.py
"""

import gc
import json
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
for _p in (str(FORK.parent / "app-builder-ponytail"), str(FORK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import psutil

PROC = psutil.Process()

CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
N = 10_000_000
OUT = FORK / "tests" / "heavy" / "bench_h2o_gpu_10M.json"

GOLD = {
    "Q1": {"ngroups": 100, "total": 29998789},
    "Q2": {"ngroups": 10000, "total": 29998789},
    "Q3": {"ngroups": 100000, "v1": 29998789, "mean3_sum": 4999719.622344427},
    "Q4": {"ngroups": 100, "m1": 299.98798187506526,
           "m2": 799.894179409978, "m3": 4999.766872833688},
    "Q5": {"ngroups": 100000, "s1": 29998789, "s2": 79989360,
           "s3_scaled": 499976651408061},
}


def rss():
    return PROC.memory_info().rss / 1e9


def run_q(a, jobs, backend):
    st = {}
    s = time.perf_counter()
    g = a["compile"](jobs)
    st["compile"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    g = a["optimize"](g)
    st["optimize"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    r = a["evaluate"](g, backend, N)["result"]
    st["execute"] = (time.perf_counter() - s) * 1000
    st["total"] = st["compile"] + st["optimize"] + st["execute"]
    return r, st


def main():
    import pandas as pd

    from builder import MAIN

    print("=== h2o-gpu-10M: same graph CPU vs GPU ===", flush=True)
    print(f"RSS start {rss():.2f}GB", flush=True)
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    import wgpu

    dev = wgpu.gpu.request_adapter_sync(
        power_preference="high-performance").info
    gpu_name = dev.device
    print(f"GPU: {gpu_name}", flush=True)
    out = {"N": N, "gpu": gpu_name, "Q": {}}

    t0 = time.perf_counter()
    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id4", "id6",
                                   "v1", "v2", "v3"])
    out["csv_load_ms"] = (time.perf_counter() - t0) * 1000
    assert len(df) == N
    print(f"csv_load {out['csv_load_ms']:.0f}ms RSS {rss():.2f}GB", flush=True)

    enc = {}
    for col in ("id1", "id2", "id3"):
        raw = df[col].to_numpy()
        t0 = time.perf_counter()
        g = a["optimize"](a["compile"]([a["ir_encode_pattern"]("c", raw,
                                                               "id")]))
        bufs = a["cpu_execute"](g["nodes"])
        ms = (time.perf_counter() - t0) * 1000
        enc[col] = bufs["c"]
        print(f"encode {col}: {ms:.0f}ms sum="
              f"{int(enc[col].astype(np.int64).sum())} RSS {rss():.2f}GB",
              flush=True)
    c1, c2, c3 = enc["id1"], enc["id2"], enc["id3"]
    id4 = df["id4"].to_numpy().astype(np.int32)
    id6 = df["id6"].to_numpy().astype(np.int32)
    v1 = df["v1"].to_numpy().astype(np.int32)
    v2 = df["v2"].to_numpy().astype(np.int32)
    v3 = df["v3"].to_numpy().astype(np.float64)
    m2_q2 = int(c2.max()) + 1
    del df
    gc.collect()

    jobs_q1 = [a["ir_series"]("k", c1), a["ir_series"]("v", v1),
               a["ir_groupby"]("g", "v", "k", "sum")]
    jobs_q2 = [a["ir_series"]("c1", c1), a["ir_series"]("c2", c2),
               a["ir_series"]("v", v1),
               a["ir_pack_keys"]("k", "c1", "c2"),
               a["ir_groupby"]("g", "v", "k", "sum")]
    jobs_q3 = [a["ir_series"]("k", c3),
               a["ir_series"]("v1", v1), a["ir_series"]("v3", v3, "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                     {"v1": ("sum",), "v3": ("mean",)})]
    jobs_q4 = [a["ir_series"]("k", id4),
               a["ir_series"]("v1", v1), a["ir_series"]("v2", v2),
               a["ir_series"]("v3", v3, "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                     {"v1": ("mean",), "v2": ("mean",),
                                      "v3": ("mean",)})]
    jobs_q5 = [a["ir_series"]("k", id6),
               a["ir_series"]("v1", v1), a["ir_series"]("v2", v2),
               a["ir_series"]("v3", v3, "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                     {"v1": ("sum",), "v2": ("sum",),
                                      "v3": ("sum",)})]

    def check_q1(r):
        assert isinstance(r, dict) and len(r) == 100
        tot = int(sum(int(v) for v in r.values()))
        assert tot == 29998789, tot
        return tot

    def check_q2_cpu(r):
        # CPU labels: bitpack (hi<<32)|lo -> total chk
        assert isinstance(r, dict) and len(r) == 10000, len(r)
        tot = int(sum(int(v) for v in r.values()))
        assert tot == 29998789, tot
        return tot

    def check_q2_gpu(r):
        # GPU labels: int32-direct c1*m2+c2 -> total chk
        assert isinstance(r, dict) and len(r) == 10000, len(r)
        tot = int(sum(int(v) for v in r.values()))
        assert tot == 29998789, tot
        return tot

    def pairs_cpu(r):
        return {((k >> 32), (k & 0xFFFFFFFF)): int(v)
                for k, v in r.items()}

    def pairs_gpu(r):
        return {divmod(k, m2_q2): int(v) for k, v in r.items()}

    def check_q3(r):
        assert len(r) == 100000
        assert int(sum(c["v1"]["sum"] for c in r.values())) == 29998789
        m3 = float(sum(c["v3"]["mean"] for c in r.values()))
        assert abs(m3 - GOLD["Q3"]["mean3_sum"]) < 1e-6 * abs(
            GOLD["Q3"]["mean3_sum"]) + 1e-6
        return m3

    def check_q4(r):
        assert len(r) == 100
        s = {c: float(sum(cell[c]["mean"] for cell in r.values()))
             for c in ("v1", "v2", "v3")}
        md = max(abs(s["v1"] - GOLD["Q4"]["m1"]),
                 abs(s["v2"] - GOLD["Q4"]["m2"]),
                 abs(s["v3"] - GOLD["Q4"]["m3"]))
        assert md < 1e-9
        return md

    def check_q5(r):
        assert len(r) == 100000
        s1 = int(sum(c["v1"]["sum"] for c in r.values()))
        s2 = int(sum(c["v2"]["sum"] for c in r.values()))
        s3 = float(sum(c["v3"]["sum"] for c in r.values()))
        assert s1 == 29998789 and s2 == 79989360
        sc = int(round(s3 * 1e6))
        assert sc == 499976651408061, sc
        return sc

    plan = (("Q1", jobs_q1, check_q1, check_q1),
            ("Q2", jobs_q2, check_q2_cpu, check_q2_gpu),
            ("Q3", jobs_q3, check_q3, None),
            ("Q4", jobs_q4, check_q4, None),
            ("Q5", jobs_q5, check_q5, None))
    for name, jobs, chk_cpu, chk_gpu in plan:
        row = {}
        for backend, chk in (("cpu", chk_cpu), ("gpu", chk_gpu)):
            if chk is None:
                try:
                    run_q(a, jobs, backend)
                    row[backend] = {"status": "UNEXPECTED-PASS"}
                    print(f"{name} {backend}: UNEXPECTED PASS", flush=True)
                except ValueError as e:
                    row[backend] = {"status": "guard_error",
                                    "error": str(e)[:160]}
                    print(f"{name} {backend}: guard_error "
                          f"{str(e)[:100]} RSS {rss():.2f}GB", flush=True)
                continue
            r, st = run_q(a, jobs, backend)
            cv = chk(r)
            r2, st2 = run_q(a, jobs, backend)
            chk(r2)
            if name == "Q2" and backend == "gpu":
                # cross-backend label-mapped equivalence (harness-level)
                ref = pairs_cpu(run_q(a, jobs, "cpu")[0])
                gpairs = pairs_gpu(r)
                assert gpairs.keys() == ref.keys(), "Q2 (c1,c2) groups differ"
                for kk in ref:
                    assert int(gpairs[kk]) == int(ref[kk]), kk
            row[backend] = {"cold_ms": st["total"], "cold_stages": st,
                            "warm_ms": st2["total"], "warm_stages": st2,
                            "chk": cv, "status": "ok"}
            print(f"{name} {backend} cold={st['total']:.0f}ms "
                  f"(exec {st['execute']:.0f}) warm={st2['total']:.0f}ms "
                  f"(exec {st2['execute']:.0f}) chk={cv} RSS {rss():.2f}GB",
                  flush=True)
        out["Q"][name] = row
    out["rss_gb"] = rss()
    with open(OUT, "w") as f:
        json.dump(json.loads(json.dumps(out, default=float)), f, indent=1)
    print(f"JSON written RSS {rss():.2f}GB", flush=True)


if __name__ == "__main__":
    main()
