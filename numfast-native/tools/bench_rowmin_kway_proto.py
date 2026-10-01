# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""PROTOTYPE bench: K4 specialized vs K-way generic (K=4) rowwise
time-argmin + gather. Additive research script; existing symbols untouched.

Compares nf_rowwise_min4_time_argmin_gather (K4) vs
nf_rowwise_kway_time_argmin_gather with k=4 (generic) via ctypes direct.
Exact parity (bit-exact t_best/d_best/m_best), warm median timing.

Seed 42. Sizes: 50k/100k/200k/500k/1M/5M. Writes
results/bench_rowmin_kway_proto.json. Exit 0 = PASS (parity + json).
"""
import ctypes
import json
import os
import statistics
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

DLL = os.path.join(ROOT, "target", "x86_64-pc-windows-gnu", "release", "numfast_native.dll")

SIZES = [50_000, 100_000, 200_000, 500_000, 1_000_000, 5_000_000]
WARM_REPS = 7

lib = ctypes.CDLL(DLL)
lib.nf_rowwise_min4_time_argmin_gather.argtypes = (
    [ctypes.c_void_p] * 8 + [ctypes.c_size_t] + [ctypes.c_void_p] * 3)
lib.nf_rowwise_min4_time_argmin_gather.restype = ctypes.c_int32
lib.nf_rowwise_kway_time_argmin_gather.argtypes = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
lib.nf_rowwise_kway_time_argmin_gather.restype = ctypes.c_int32


def make_data(n, rng):
    T = [np.ascontiguousarray(
        rng.integers(-2 ** 31, 2 ** 31, size=n).astype(np.int32))
        for _ in range(4)]
    D = [np.ascontiguousarray(rng.normal(0, 20, size=n).astype(np.float32))
        for _ in range(4)]
    tie = rng.random(size=n) < 0.3
    T[1][tie] = T[0][tie]
    tie2 = rng.random(size=n) < 0.2
    T[3][tie2] = T[2][tie2]
    return T, D


def call_k4(T, D):
    n = T[0].size
    tb = np.empty(n, dtype=np.int32)
    db = np.empty(n, dtype=np.float32)
    mb = np.empty(n, dtype=np.uint8)
    rc = lib.nf_rowwise_min4_time_argmin_gather(
        T[0].ctypes.data, T[1].ctypes.data, T[2].ctypes.data, T[3].ctypes.data,
        D[0].ctypes.data, D[1].ctypes.data, D[2].ctypes.data, D[3].ctypes.data,
        n, tb.ctypes.data, db.ctypes.data, mb.ctypes.data)
    assert rc == 0, rc
    return tb, db, mb


def call_kway(T, D):
    n = T[0].size
    k = len(T)
    assert k == len(D) == 4
    t_ptrs = (ctypes.c_void_p * k)(*[a.ctypes.data for a in T])
    d_ptrs = (ctypes.c_void_p * k)(*[a.ctypes.data for a in D])
    tb = np.empty(n, dtype=np.int32)
    db = np.empty(n, dtype=np.float32)
    mb = np.empty(n, dtype=np.uint8)
    rc = lib.nf_rowwise_kway_time_argmin_gather(
        t_ptrs, d_ptrs, k, n, tb.ctypes.data, db.ctypes.data, mb.ctypes.data)
    assert rc == 0, rc
    return tb, db, mb


def bit_exact(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def time_interleaved(T, D):
    """Unbiased timing: alternate K4/K-way per rep (no order bias)."""
    call_k4(T, D)  # warmup both (page faults + icache outside the median)
    call_kway(T, D)
    r4, rkw = [], []
    for _ in range(WARM_REPS):
        t = time.perf_counter()
        call_k4(T, D)
        r4.append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter()
        call_kway(T, D)
        rkw.append((time.perf_counter() - t) * 1e3)
    # cold: first timed call per lane on warm pages (icache already hot)
    t = time.perf_counter()
    call_k4(T, D)
    ck4 = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    call_kway(T, D)
    ckw = (time.perf_counter() - t) * 1e3
    return ck4, statistics.median(r4), ckw, statistics.median(rkw)


def main():
    rng = np.random.default_rng(42)
    fails = []
    table = []

    # --- edge parity first (NaN/Inf payload, ties, extremes), small N ---
    re = np.random.default_rng(42)
    n = 4096
    T = [np.ascontiguousarray(
        re.integers(-2 ** 31, 2 ** 31, size=n).astype(np.int32))
        for _ in range(4)]
    T[1][:512] = T[0][:512]
    T[3][1024:1536] = T[2][1024:1536]
    T[0][0] = np.int32(-2 ** 31)
    T[1][1] = np.int32(2 ** 31 - 1)
    D = [np.ascontiguousarray(re.normal(0, 5, size=n).astype(np.float32))
         for _ in range(4)]
    D[0][::3] = np.nan
    D[1][1::4] = np.nan
    D[2][::5] = np.inf
    D[3][2::7] = -np.inf
    D[0][10] = np.float32(-0.0)
    a = call_k4(T, D)
    b = call_kway(T, D)
    edge_ok = all(bit_exact(x, y) for x, y in zip(a, b))
    print(("PASS " if edge_ok else "FAIL ") + "edge parity NaN/Inf/ties/extremes N=4096")
    if not edge_ok:
        fails.append("edge-parity")

    for size in SIZES:
        try:
            T, D = make_data(size, rng)
        except MemoryError:
            print(f"SKIP N={size} (data alloc failed)")
            continue
        try:
            a = call_k4(T, D)
            b = call_kway(T, D)
        except MemoryError:
            print(f"SKIP N={size} (lane alloc failed)")
            continue
        ok = all(bit_exact(x, y) for x, y in zip(a, b))
        print(("PASS " if ok else "FAIL ") + f"N={size} K4==Kway bit-exact")
        if not ok:
            fails.append(f"parity@{size}")
        # time K4 vs K-way interleaved (same buffers, same process); warm median
        ck4, wk4, ckw, wkw = time_interleaved(T, D)
        diff_pct = (wkw - wk4) / wk4 * 100.0
        row = {"n": size, "parity_bitexact": bool(ok),
               "k4_cold_ms": ck4, "k4_warm_ms": wk4,
               "kway_cold_ms": ckw, "kway_warm_ms": wkw,
               "diff_pct": diff_pct}
        table.append(row)
        print(f"N={size:>8} warm_ms K4={wk4:.3f} Kway={wkw:.3f} "
              f"diff={diff_pct:+.2f}% cold K4={ck4:.3f} Kway={ckw:.3f}")
        del T, D, a, b

    diffs = [r["diff_pct"] for r in table]
    mx = max(diffs) if diffs else float("nan")
    mn = min(diffs) if diffs else float("nan")
    if all(-0.5 <= d <= 2.0 for d in diffs):
        verdict = "K-WAY (K4 thin wrappers)"
    elif all(15.0 <= d <= 30.0 for d in diffs):
        verdict = "K4+K-WAY"
    elif all(d <= 2.0 for d in diffs):
        verdict = "K-WAY (K4 thin wrappers)"
    elif any(d >= 15.0 for d in diffs):
        verdict = "K4+K-WAY"
    else:
        verdict = "UNDECIDED (outside 0-2% / 15-30% bands)"
    print(f"diff range [{mn:+.2f}%, {mx:+.2f}%] -> verdict: {verdict}")
    json.dump({"seed": 42, "warm_reps": WARM_REPS, "dll": DLL,
               "edge_parity": bool(edge_ok), "table": table,
               "diff_min_pct": mn, "diff_max_pct": mx,
               "verdict": verdict,
               "pass": not fails, "fails": fails},
              open(os.path.join(RES, "bench_rowmin_kway_proto.json"), "w"),
              indent=1)
    print("PASS" if not fails else f"FAIL {fails}")
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
