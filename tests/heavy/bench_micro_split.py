# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Micro-split: probes vs conversion vs kernel on 10M snapshot (NEW file).
Sequential, snapshot once. Quantifies: range_probe, int_exact_ok,
i32->f64 conversion, np.stack, ST kernel, per-worker overhead.
Usage: timeout ... python tests/heavy/bench_micro_split.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
FORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, "C:/App/numfast/app-builder-ponytail")
sys.path.insert(0, "C:/App/numfast/numfast-ponytail")
import numpy as np

SNAP = FORK / "scratch" / "snap_G1_1e7_1e2_0_0"
N = 10_000_000


def t(fn, reps=5):
    for _ in range(2):
        fn()
    best = 1e18
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - s) * 1000)
    return best


def main():
    from builder import MAIN
    a = MAIN["build"]("C:/App/numfast/numfast-ponytail").alias
    K = {k: np.load(str(SNAP / f"{k}.npy")) for k in
         ("K1", "K2", "K3", "K4", "K6", "V1", "V2", "V3")}
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gi_mod", str(FORK / "src" / "Drivers" / "CPU" / "_lib" / "groupindex.py"))
    gi_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gi_mod)
    rp = gi_mod.range_probe
    print("range_probe K6:", round(t(lambda: rp(K["K6"])), 1), "ms", flush=True)
    print("K6.min+max:", round(t(lambda: (K["K6"].min(), K["K6"].max())), 1), "ms", flush=True)
    print("V1.min+max+V2.min+max:",
          round(t(lambda: (K["V1"].min(), K["V1"].max(), K["V2"].min(), K["V2"].max())), 1), "ms", flush=True)
    print("V1 i32->f64:",
          round(t(lambda: np.ascontiguousarray(K["V1"], dtype=np.float64)), 1), "ms", flush=True)
    V1f = np.ascontiguousarray(K["V1"], dtype=np.float64)
    V2f = np.ascontiguousarray(K["V2"], dtype=np.float64)
    V3 = K["V3"]
    print("stack 3xf64:",
          round(t(lambda: np.ascontiguousarray(np.stack([V1f, V2f, V3]).reshape(-1))), 1), "ms", flush=True)
    spec = importlib.util.spec_from_file_location(
        "nc", str(FORK / "src" / "Drivers" / "CPU" / "_lib" / "native_cpu.py"))
    nc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nc)
    m6 = int(K["K6"].max()) + 1
    print("ST multi kernel Q5:",
          round(t(lambda: nc.multi_sum_count(K["K6"], [V1f, V2f, V3], m6)), 1), "ms", flush=True)
    print("ST multi kernel Q5 int-in (conv inside):",
          round(t(lambda: nc.multi_sum_count(K["K6"], [K["V1"], K["V2"], V3], m6)), 1), "ms", flush=True)
    m4 = int(K["K4"].max()) + 1
    V1f4 = V1f
    print("ST multi kernel Q4:",
          round(t(lambda: nc.multi_sum_count(K["K4"], [V1f4, V2f, V3], m4)), 1), "ms", flush=True)
    print("K1.max+K2.max+K1.min+K2.min (Q2 pack probes):",
          round(t(lambda: (K["K1"].max(), K["K2"].max(), K["K1"].min(), K["K2"].min())), 1), "ms", flush=True)
    print("pack_i32_direct 10M:",
          round(t(lambda: nc.pack_i32_direct(K["K1"], K["K2"], int(K["K2"].max()) + 1)), 1), "ms", flush=True)


if __name__ == "__main__":
    main()
