# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H2O 10M Q1-Q5 CPU vs GPU with SPEC-DELTA-7 scaled v3 (NEW file).

Same graphs both backends, same Planner, same Runtime. v3 float64 carries
the Schema contract (column_schema -> ir_series scale/offset); GPU runs
exact integer WGSL on int32 ticks, f64 unscale on host. Q1/Q2 unchanged
(int path). chk official GOLD from bench_cpu_modes_10M.py.

One process, sequential, cold then warm per (Q, backend). No 1B. No RTX
profiling (wall stages only, separate GO).

Usage (Git Bash):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    timeout 550 /c/Users/Mikech/AppData/Local/Programs/Python/Python314/python.exe \
    tests/heavy/bench_h2o_scaled_gpu_10M.py
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
OUT = FORK / "tests" / "heavy" / "bench_h2o_scaled_gpu_10M.json"

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


def maxdiff_multi(rcpu, rgpu):
    """Max |cpu-gpu| per (key, col, op); integer ops must be 0."""
    md, worst = 0.0, None
    assert set(rcpu) == set(rgpu), "group key sets differ"
    for kk in rcpu:
        for col in rcpu[kk]:
            for op in rcpu[kk][col]:
                d = abs(float(rcpu[kk][col][op]) - float(rgpu[kk][col][op]))
                if d > md:
                    md, worst = d, (kk, col, op)
    return md, worst


def main():
    import pandas as pd

    from builder import MAIN

    print("=== h2o-scaled-gpu-10M: same graph CPU vs GPU ===", flush=True)
    print(f"RSS start {rss():.2f}GB", flush=True)
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    import wgpu

    dev = wgpu.gpu.request_adapter_sync(
        power_preference="high-performance").info
    gpu_name = dev.device
    print(f"GPU: {gpu_name}", flush=True)
    out = {"N": N, "gpu": gpu_name, "Q": {}}

    # Scale contract comes from Schema, never hardcoded in the engine.
    v3col = a["column_schema"]("v3", "float64", 1e-6, 0)
    assert v3col["physical"] == "int32" and v3col["scale"] == 1e-6
    SC, OFF = float(v3col["scale"]), float(v3col["offset"])
    print(f"Schema v3: logical={v3col['logical']} physical={v3col['physical']} "
          f"scale={SC} offset={OFF}", flush=True)
    out["v3_schema"] = v3col

    t0 = time.perf_counter()
    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id4", "id6",
                                   "v1", "v2", "v3"])
    out["csv_load_ms"] = (time.perf_counter() - t0) * 1000
    assert len(df) == N
    print(f"csv_load {out['csv_load_ms']:.0f}ms RSS {rss():.2f}GB", flush=True)

    # v3 range proof before any bench (SPEC-DELTA-7.4): int32 per value,
    # int64 total, scaled sum == GOLD.
    v3raw = df["v3"].to_numpy(dtype=np.float64)
    phys = np.rint((v3raw - OFF) / SC)
    out["v3_range"] = {"min": float(v3raw.min()), "max": float(v3raw.max()),
                       "phys_min": int(phys.min()), "phys_max": int(phys.max()),
                       "scaled_sum": int(phys.astype(np.int64).sum())}
    assert int(phys.min()) >= -(2 ** 31) and int(phys.max()) <= 2 ** 31 - 1
    assert int(v3raw.size) * int(max(abs(float(phys.min())),
                                     abs(float(phys.max())))) < 2 ** 63
    assert out["v3_range"]["scaled_sum"] == GOLD["Q5"]["s3_scaled"]
    print(f"v3 range [{out['v3_range']['min']}, {out['v3_range']['max']}] "
          f"phys [{out['v3_range']['phys_min']}, {out['v3_range']['phys_max']}] "
          f"scaled_sum={out['v3_range']['scaled_sum']} RSS {rss():.2f}GB",
          flush=True)

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
               a["ir_series"]("v1", v1),
               a["ir_series"]("v3", v3, "float64", scale=SC, offset=OFF),
               a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                     {"v1": ("sum",), "v3": ("mean",)})]
    jobs_q4 = [a["ir_series"]("k", id4),
               a["ir_series"]("v1", v1), a["ir_series"]("v2", v2),
               a["ir_series"]("v3", v3, "float64", scale=SC, offset=OFF),
               a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                     {"v1": ("mean",), "v2": ("mean",),
                                      "v3": ("mean",)})]
    jobs_q5 = [a["ir_series"]("k", id6),
               a["ir_series"]("v1", v1), a["ir_series"]("v2", v2),
               a["ir_series"]("v3", v3, "float64", scale=SC, offset=OFF),
               a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                     {"v1": ("sum",), "v2": ("sum",),
                                      "v3": ("sum",)})]

    def check_q1(r):
        assert isinstance(r, dict) and len(r) == 100
        tot = int(sum(int(v) for v in r.values()))
        assert tot == 29998789, tot
        return tot

    def check_q2_cpu(r):
        assert isinstance(r, dict) and len(r) == 10000, len(r)
        tot = int(sum(int(v) for v in r.values()))
        assert tot == 29998789, tot
        return tot

    check_q2_gpu = check_q2_cpu

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

    plan = (("Q1", jobs_q1, check_q1, check_q1, None),
            ("Q2", jobs_q2, check_q2_cpu, check_q2_gpu, "pairs"),
            ("Q3", jobs_q3, check_q3, check_q3, "multi"),
            ("Q4", jobs_q4, check_q4, check_q4, "multi"),
            ("Q5", jobs_q5, check_q5, check_q5, "multi"))
    for name, jobs, chk_cpu, chk_gpu, cmp in plan:
        row = {}
        results = {}
        for backend, chk in (("cpu", chk_cpu), ("gpu", chk_gpu)):
            r, st = run_q(a, jobs, backend)
            cv = chk(r)
            r2, st2 = run_q(a, jobs, backend)
            chk(r2)
            results[backend] = r
            row[backend] = {"cold_ms": st["total"], "cold_stages": st,
                            "warm_ms": st2["total"], "warm_stages": st2,
                            "chk": cv, "status": "ok"}
            print(f"{name} {backend} cold={st['total']:.0f}ms "
                  f"(exec {st['execute']:.0f}) warm={st2['total']:.0f}ms "
                  f"(exec {st2['execute']:.0f}) chk={cv} RSS {rss():.2f}GB",
                  flush=True)
        if cmp == "pairs":
            ref = pairs_cpu(results["cpu"])
            gpairs = pairs_gpu(results["gpu"])
            assert gpairs.keys() == ref.keys(), "Q2 (c1,c2) groups differ"
            for kk in ref:
                assert int(gpairs[kk]) == int(ref[kk]), kk
            row["cpu_gpu_maxdiff"] = 0
        elif cmp == "multi":
            md, worst = maxdiff_multi(results["cpu"], results["gpu"])
            row["cpu_gpu_maxdiff"] = md
            row["cpu_gpu_worst"] = None if worst is None else list(worst)
            print(f"{name} CPUvsGPU maxdiff={md} worst={worst} "
                  f"RSS {rss():.2f}GB", flush=True)
        else:
            assert results["cpu"] == results["gpu"], f"{name} CPU!=GPU"
            row["cpu_gpu_maxdiff"] = 0
        out["Q"][name] = row
    out["rss_gb"] = rss()
    with open(OUT, "w") as f:
        json.dump(json.loads(json.dumps(out, default=float)), f, indent=1)
    print(f"JSON written RSS {rss():.2f}GB", flush=True)


if __name__ == "__main__":
    main()
