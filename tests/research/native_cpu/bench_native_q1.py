# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native Q1 prototype vs NumPy vs production (research, 10M H2O id1).

Compares on the SAME resident int32 codes + int32 values (Arrow encode once,
sum-verified):
  native : q1_dense.dll q1_dense_fused via ctypes (one fused sum+count pass)
  numpy  : bincount sums + bincount counts (pure-core equivalent state)
  prod   : production Q1 groupby-only (Planner-gated dense + dict), reference
Correctness: per-group exact vs production dict (max_diff 0), total 29998789.
ABI overhead measured separately (n=0 call + marshal).
Python API / IR / Planner untouched. No Polars/DuckDB/pandas-in-hot-path
(pandas only for CSV load, outside timing), no Arrow in prototype.

Usage (Git Bash, fork-first PYTHONPATH):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    /c/Users/Mikech/AppData/Local/Programs/Python/Python314/python.exe \
    tests/research/native_cpu/bench_native_q1.py
"""

import ctypes
import gc
import json
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[3]
for _p in (str(FORK.parent / "app-builder-ponytail"), str(FORK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

HERE = Path(__file__).resolve().parent
CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
N = 10_000_000

import numpy as np
import psutil

PROC = psutil.Process()


def rss():
    return PROC.memory_info().rss / 1e9


def best_of(fn, reps=5):
    best = 1e18
    for _ in range(reps):
        best = min(best, fn())
    return best


def main():
    import pandas as pd

    print("=== native-q1 vs numpy vs prod (10M) ===", flush=True)
    print(f"RSS start {rss():.2f}GB", flush=True)

    t0 = time.perf_counter()
    df = pd.read_csv(CSV, usecols=["id1", "v1"])
    load_ms = (time.perf_counter() - t0) * 1000
    assert len(df) == N
    raw = df["id1"].to_numpy()
    vals = df["v1"].to_numpy().astype(np.int32)
    del df
    gc.collect()

    from builder import MAIN

    a = MAIN["build"](str(FORK)).alias
    t0 = time.perf_counter()
    g = a["optimize"](a["compile"]([a["ir_encode_pattern"]("c", raw, "id")]))
    bufs = a["cpu_execute"](g["nodes"])
    enc_ms = (time.perf_counter() - t0) * 1000
    codes = np.ascontiguousarray(bufs["c"], dtype=np.int32)
    vals = np.ascontiguousarray(vals, dtype=np.int32)
    assert codes.dtype == np.int32 and codes.flags["C_CONTIGUOUS"]
    assert int(codes.astype(np.int64).sum()) == 505040489  # STEP 0 bit-identical sum
    kmin, kmax = int(codes.min()), int(codes.max())
    assert kmin >= 0
    m = kmax + 1
    print(f"load {load_ms:.0f}ms encode {enc_ms:.0f}ms "
          f"codes sum=505040489 kmin={kmin} kmax={kmax} m={m} RSS {rss():.2f}GB", flush=True)

    dll = ctypes.CDLL(str(HERE / "q1_dense.dll"))
    fn = dll.q1_dense_fused
    fn.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                   ctypes.c_int64, ctypes.c_int32,
                   ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64)]
    fn.restype = None
    cp = codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    vp = vals.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    # ---- correctness: native state vs production dict, exact ----
    sums = np.zeros(m, dtype=np.int64)
    counts = np.zeros(m, dtype=np.int64)
    fn(cp, vp, N, m,
       sums.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
       counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
    mask = counts > 0
    ukeys = np.flatnonzero(mask)
    g_ref, st_ref = None, None
    s = time.perf_counter()
    gg = a["optimize"](a["compile"]([a["ir_series"]("k", codes),
                                     a["ir_series"]("v", vals),
                                     a["ir_groupby"]("gr", "v", "k", "sum")]))
    bbs = a["cpu_execute"](gg["nodes"])
    prod = bbs["gr"]
    prod_ms = (time.perf_counter() - s) * 1000
    gi_side = bbs.get("gr#groupindex", {})
    assert len(prod) == ukeys.size == 100, (len(prod), ukeys.size)
    max_diff = 0
    for k in ukeys:
        max_diff = max(max_diff, abs(int(sums[k]) - int(prod[int(k)])))
    tot = int(sums.sum())
    assert max_diff == 0 and tot == 29998789, (max_diff, tot)
    print(f"correctness: max_diff={max_diff} total={tot} groups={ukeys.size} "
          f"(prod Q1 {prod_ms:.0f}ms incl dispatch+dict) RSS {rss():.2f}GB", flush=True)

    # ---- timing: native fused (fresh zeroed state per rep, like prod) ----
    def run_native():
        sums.fill(0)
        counts.fill(0)
        sp = sums.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        kp = counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        s = time.perf_counter()
        fn(cp, vp, N, m, sp, kp)
        return (time.perf_counter() - s) * 1000

    def run_zero_only():
        s = time.perf_counter()
        sums.fill(0)
        counts.fill(0)
        return (time.perf_counter() - s) * 1000

    def run_numpy():
        s = time.perf_counter()
        k64 = codes.astype(np.int64, copy=False)
        w = vals.astype(np.int64, copy=False)
        ss = np.bincount(k64, weights=w, minlength=m)
        cc = np.bincount(k64, minlength=m)
        return (time.perf_counter() - s) * 1000

    k64 = codes.astype(np.int64)
    w64 = vals.astype(np.int64)

    def run_numpy_preconv():
        s = time.perf_counter()
        ss = np.bincount(k64, weights=w64, minlength=m)
        cc = np.bincount(k64, minlength=m)
        return (time.perf_counter() - s) * 1000

    def run_abi():
        s = time.perf_counter()
        fn(cp, vp, 0, m,
           sums.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
           counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
        return (time.perf_counter() - s) * 1000

    native_ms = best_of(run_native)
    zero_ms = best_of(run_zero_only)
    numpy_ms = best_of(run_numpy)
    numpy_preconv_ms = best_of(run_numpy_preconv)
    abi_ms = best_of(run_abi, reps=100)
    print(f"native fused best={native_ms:.1f}ms (zeroing {zero_ms:.2f}ms inside)", flush=True)
    print(f"numpy bincount sums+counts best={numpy_ms:.1f}ms "
          f"(preconverted int64: {numpy_preconv_ms:.1f}ms)", flush=True)
    print(f"ABI overhead (n=0 call, best-of-100)={abi_ms*1000:.1f}us", flush=True)
    print(f"RATIO numpy/native={numpy_ms/native_ms:.2f}x "
          f"prod-Q1/native={prod_ms/native_ms:.2f}x (prod incl dispatch+dict)", flush=True)
    state_bytes = sums.nbytes + counts.nbytes
    print(f"state={state_bytes}B m={m} RSS {rss():.2f}GB", flush=True)

    out = {"N": N, "m": m, "kmin": kmin, "kmax": kmax,
           "correctness": {"max_diff": max_diff, "total": tot, "ngroups": int(ukeys.size)},
           "native_fused_ms": native_ms, "zero_ms": zero_ms,
           "numpy_bincount_ms": numpy_ms, "numpy_preconv_ms": numpy_preconv_ms,
           "abi_us": abi_ms * 1000,
           "prod_q1_ms": prod_ms, "prod_groupindex": gi_side,
           "state_bytes": state_bytes,
           "c_lines": sum(1 for _ in open(HERE / "q1_dense.c", encoding="utf-8")),
           "rss_gb": rss()}
    with open(HERE / "bench_native_q1.json", "w") as f:
        json.dump(json.loads(json.dumps(out, default=float)), f, indent=1)
    print("JSON written", flush=True)


if __name__ == "__main__":
    main()
