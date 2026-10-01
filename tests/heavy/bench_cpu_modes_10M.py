# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU modes A/B on H2O 10M groupby-only (heavy): pure-core vs optional.

Groupby-only on resident int32 codes (encode done once per process, verified
bit-identical across modes): isolates group-index + aggregate, the native
candidate surface (STEP 3 compares native Q1 here).

Modes (fresh process each, Git Bash, fork-first PYTHONPATH):
  optional : as-is (Arrow encode, Numba groupindex kernels)
  purecore : pyarrow+numba import-blocked (NumPy fallback encode, np.all probe,
             unique/bincount groupindex paths)

Goldens (bench_resident_10M): Q1/Q2 total 29998789; Q3 v1 29998789 +
mean3_sum 4999719.622344427; Q4 m1/m2/m3; Q5 s1/s2/s3_scaled 499976651408061.
Each Q runs cold then warm; stages compile/optimize/execute split; chk exact.
GPU untouched. No 1B.

Usage:
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    /c/Users/Mikech/AppData/Local/Programs/Python/Python314/python.exe \
    tests/heavy/bench_cpu_modes_10M.py --mode optional
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

MODE = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "optional"
assert MODE in ("optional", "purecore"), MODE

if MODE == "purecore":
    import importlib.abc

    class _Blocker(importlib.abc.MetaPathFinder):
        _BAN = ("pyarrow", "numba")

        def find_spec(self, name, path=None, target=None):
            if name in self._BAN or name.startswith(("pyarrow.", "numba.")):
                raise ImportError(f"blocked for pure-core mode: {name}")
            return None

    sys.meta_path.insert(0, _Blocker())
    for _m in [m for m in sys.modules if m == "pyarrow" or m.startswith(("pyarrow.", "numba", "llvmlite"))]:
        del sys.modules[_m]

import numpy as np
import psutil

PROC = psutil.Process()

CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
N = 10_000_000
OUT = FORK / "tests" / "heavy" / "bench_cpu_modes_10M.json"

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


def build_kernel():
    from builder import MAIN

    return MAIN["build"](str(FORK))


def run_q(a, jobs):
    st = {}
    s = time.perf_counter()
    g = a["compile"](jobs)
    st["compile"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    g = a["optimize"](g)
    st["optimize"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    r = a["evaluate"](g, "cpu", N)["result"]
    st["execute"] = (time.perf_counter() - s) * 1000
    st["total"] = st["compile"] + st["optimize"] + st["execute"]
    return r, st


def int_check(res, gold_total, gold_groups, label):
    assert isinstance(res, dict), f"{label}: groupby must return dict"
    assert len(res) == gold_groups, f"{label}: groups {len(res)} != {gold_groups}"
    tot = int(sum(int(v) for v in res.values()))
    assert tot == gold_total, f"{label}: total {tot} != {gold_total}"
    return tot


def main():
    import pandas as pd

    print(f"=== cpu-modes-10M [{MODE}] ===", flush=True)
    print(f"RSS start {rss():.2f}GB", flush=True)
    kernel = build_kernel()
    a = kernel.alias
    out = {"mode": MODE, "N": N}

    t0 = time.perf_counter()
    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"])
    out["csv_load_ms"] = (time.perf_counter() - t0) * 1000
    assert len(df) == N
    print(f"csv_load {out['csv_load_ms']:.0f}ms RSS {rss():.2f}GB", flush=True)

    # ---- encode once (Arrow vs NumPy-fallback; bit-identical per STEP 0) ----
    enc = {}
    for col in ("id1", "id2", "id3"):
        raw = df[col].to_numpy()
        t0 = time.perf_counter()
        g = a["optimize"](a["compile"]([a["ir_encode_pattern"]("c", raw, "id")]))
        bufs = a["cpu_execute"](g["nodes"])
        ms = (time.perf_counter() - t0) * 1000
        enc[col] = {"ms": ms, "codes": bufs["c"],
                    "sum": int(bufs["c"].astype(np.int64).sum()),
                    "width": bufs["c#pattern"]["width"]}
        print(f"encode {col}: {ms:.0f}ms sum={enc[col]['sum']} "
              f"width={enc[col]['width']} RSS {rss():.2f}GB", flush=True)
    out["encode_ms"] = {c: enc[c]["ms"] for c in enc}
    out["encode_sums"] = {c: enc[c]["sum"] for c in enc}
    c1, c2, c3 = enc["id1"]["codes"], enc["id2"]["codes"], enc["id3"]["codes"]
    id4 = df["id4"].to_numpy().astype(np.int32)
    id6 = df["id6"].to_numpy().astype(np.int32)
    v1 = df["v1"].to_numpy().astype(np.int32)
    v2 = df["v2"].to_numpy().astype(np.int32)
    v3 = df["v3"].to_numpy().astype(np.float64)
    del df
    gc.collect()

    Q = {}
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
        return int_check(r, 29998789, 100, "Q1")

    def check_q2(r):
        return int_check(r, 29998789, 10000, "Q2")

    def check_q3(r):
        assert len(r) == 100000
        assert int(sum(c["v1"]["sum"] for c in r.values())) == 29998789
        m3 = float(sum(c["v3"]["mean"] for c in r.values()))
        assert abs(m3 - GOLD["Q3"]["mean3_sum"]) < 1e-6 * abs(GOLD["Q3"]["mean3_sum"]) + 1e-6
        return m3

    def check_q4(r):
        assert len(r) == 100
        s = {c: float(sum(cell[c]["mean"] for cell in r.values())) for c in ("v1", "v2", "v3")}
        md = max(abs(s["v1"] - GOLD["Q4"]["m1"]), abs(s["v2"] - GOLD["Q4"]["m2"]),
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

    for name, jobs, chk in (("Q1", jobs_q1, check_q1), ("Q2", jobs_q2, check_q2),
                            ("Q3", jobs_q3, check_q3), ("Q4", jobs_q4, check_q4),
                            ("Q5", jobs_q5, check_q5)):
        r, st = run_q(a, jobs)
        cv = chk(r)
        r2, st2 = run_q(a, jobs)
        chk(r2)
        Q[name] = {"cold_ms": st["total"], "cold_stages": st,
                   "warm_ms": st2["total"], "warm_stages": st2, "chk": cv}
        print(f"{name} cold={st['total']:.0f}ms (exec {st['execute']:.0f}) "
              f"warm={st2['total']:.0f}ms (exec {st2['execute']:.0f}) "
              f"chk={cv} RSS {rss():.2f}GB", flush=True)
    out["Q"] = Q
    out["rss_gb"] = rss()
    with open(OUT if MODE == "optional" else str(OUT).replace(".json", f"_{MODE}.json"), "w") as f:
        json.dump(json.loads(json.dumps(out, default=float)), f, indent=1)
    print(f"JSON written RSS {rss():.2f}GB", flush=True)


if __name__ == "__main__":
    main()
