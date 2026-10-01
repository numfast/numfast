# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research: F2 single-pass on real H2O 10M resident arrays + Q2 stage split.

Part 1: per-Q M (=kmax+1), F2 vs prod-agg timing on resident codes, golden chk.
Part 2: Q2 pack_keys ms vs hash gi_ms vs agg_ms vs dict ms (direct cpu_execute).
No prod change. Single-thread.
"""
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

FORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FORK.parent / "app-builder-ponytail"))
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / "tests" / "research" / "singlepass"))

import numpy as np
from kernels_sp import fused_singlepass, ref_unique

CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
N = 10_000_000
GOLD = {"Q1": 29998789, "Q2": 29998789}


def timed(fn, reps=3):
    best = None
    out = None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        dt = (time.perf_counter() - t) * 1000
        best = dt if best is None else min(best, dt)
    return out, best


def main():
    import kernels_sp as K
    assert K._HAS_NUMBA
    kk = np.array([0, 1, 0], dtype=np.int32)
    vv = np.array([1, 2, 3], dtype=np.int32)
    vf = np.array([1.0, 2.0, 3.0])
    for cols in ([vv], [vf], [vv, vv], [vv, vf], [vv, vv, vv],
                 [vv, vv, vf], [vv, vf, vf], [vf, vf, vf]):
        fused_singlepass(kk, cols, 2)

    from builder import MAIN
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    import pandas as pd
    import pyarrow as pa
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
    for k, vv2 in c.items():
        res[k] = vv2
    K1, K2, K3 = res["id1"]["codes"], res["id2"]["codes"], res["id3"]["codes"]
    K4, K6 = res["id4"]["codes"], res["id6"]["codes"]
    V1, V2, V3 = res["v1"]["codes"], res["v2"]["codes"], res["v3"]["codes"]

    cases = {
        "Q1": (K1, [V1]),
        "Q3": (K3, [V1, V3]),
        "Q4": (K4, [V1, V2, V3]),
        "Q5": (K6, [V1, V2, V3]),
    }
    for q, (keys, vcols) in cases.items():
        m = int(keys.max()) + 1
        ng = len(np.unique(keys))
        print(f"{q}: M={m} ngroups={ng}", flush=True)
        (uk2, c2, s2), t_f2 = timed(lambda: fused_singlepass(keys, vcols, m))
        (ukr, cr, sr), t_ref = timed(lambda: ref_unique(keys, vcols, m))
        assert bool((uk2 == ukr).all()) and bool((c2 == cr).all())
        for x, y in zip(s2, sr):
            assert bool((x == y).all()), f"{q}: sums differ"
        tot = int(sum(int(x.sum()) for x in s2
                      if x.dtype.kind == "i") or [0])
        print(f"{q}: REF-agg={t_ref:.0f}ms F2-agg={t_f2:.0f}ms ({t_ref/t_f2:.2f}x) "
              f"chk_int={tot}", flush=True)

    # ---- Part 2: see stage_split.py for Q2 pack/hash/agg/dict split ----
    print("part2: see stage_split.py", flush=True)


if __name__ == "__main__":
    main()
