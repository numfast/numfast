# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research: CPU priority microexperiment (NORMAL vs ABOVE_NORMAL vs HIGH).

1T agg path on resident 10M codes (Q1 small + Q2 heavy). REALTIME never.
Sets process priority via psutil (Windows classes). Reports best/median.
Usage: python tests/research/mtgroup/bench_priority.py
"""
import gc
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
sys.path.insert(0, str(FORK / "tests" / "research" / "singlepass"))
sys.path.insert(0, str(FORK / "tests" / "research" / "mtgroup"))

import numpy as np
import psutil

import kernels_mt as K
from kernels_sp import fused_singlepass

CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
N = 10_000_000
GOLD_TOT = 29998789
PROC = psutil.Process()

PRIOS = []
try:
    PRIOS = [("NORMAL", psutil.NORMAL_PRIORITY_CLASS),
             ("ABOVE_NORMAL", psutil.ABOVE_NORMAL_PRIORITY_CLASS),
             ("HIGH", psutil.HIGH_PRIORITY_CLASS)]
except AttributeError:
    pass


def load():
    from builder import MAIN
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    import pandas as pd
    import pyarrow as pa
    df = pd.read_csv(CSV, usecols=["id1", "id2", "v1"])
    assert len(df) == N
    s_id1 = pa.array(df["id1"].to_numpy(), type=pa.string())
    s_id2 = pa.array(df["id2"].to_numpy(), type=pa.string())
    v1 = df["v1"].to_numpy().astype(np.int32)
    del df
    gc.collect()
    r1 = a["resident_prepare"]({"id1": {"values": s_id1, "prefix": "id"}})["id1"]
    r2 = a["resident_prepare"]({"id2": {"values": s_id2, "prefix": "id"}})["id2"]
    rv = a["resident_prepare"]({"v1": {"values": v1, "dtype": "int32"}})["v1"]
    K1, K2, V1 = r1["codes"], r2["codes"], rv["codes"]
    m2 = int(K2.max()) + 1
    P2 = (K1.astype(np.int64) * np.int64(m2) + K2.astype(np.int64))
    # densify bit-free radix pack -> int32 dense codes (bijective regroup)
    _, pack32 = np.unique(P2, return_inverse=True)
    pack32 = pack32.astype(np.int32)
    return K1, pack32, V1


def bench_case(keys, vcols, m, reps=5):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        uk, c, s = fused_singlepass(keys, vcols, m)
        ts.append((time.perf_counter() - t) * 1000)
    tot = int(sum(int(x.sum()) for x in s if x.dtype.kind == "i"))
    assert tot == GOLD_TOT, tot
    ts_sorted = sorted(ts)
    return ts_sorted[0], ts_sorted[len(ts_sorted) // 2], ts


def main():
    assert PRIOS, "psutil priority classes unavailable"
    K.warmup()
    K1, P2, V1 = load()
    m1, m2 = int(K1.max()) + 1, int(P2.max()) + 1
    print(f"Q1 m={m1} Q2 m={m2}", flush=True)
    # cold warm full pass outside timing
    fused_singlepass(K1, [V1], m1)
    for name, cls in PRIOS:
        PROC.nice(cls)
        got = PROC.nice()
        assert int(got) == int(cls), (name, got, cls)
        time.sleep(0.5)
        b1, med1, reps1 = bench_case(K1, [V1], m1)
        b2, med2, reps2 = bench_case(P2, [V1], m2)
        print(f"{name}: Q1 best={b1:.1f} med={med1:.1f} "
              f"{[f'{t:.0f}' for t in reps1]} | Q2 best={b2:.1f} "
              f"med={med2:.1f} {[f'{t:.0f}' for t in reps2]}", flush=True)
    PROC.nice(psutil.NORMAL_PRIORITY_CLASS)
    print("priority restored to NORMAL", flush=True)


if __name__ == "__main__":
    main()
