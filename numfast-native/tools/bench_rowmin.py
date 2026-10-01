# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Rowwise MIN4 + argmin + gather benchmark: numpy-ref vs native vs
segmented-composition vs stacked-(N,4) alternative.

Lanes (all vectorized, no Python loops):
- numpy-ref: (4, N) stack + argmin(axis=0) + fancy gather.
- native: rowwise_min4 wrapper (Rust kernel when the rebuilt DLL is
  present, else the numpy fallback -- the run records which).
- segmented: flat (4N) + bounds + minimum.reduceat for d_best, m via
  first-equal cascade, t via where-cascade (composition of reduce +
  select primitives).
- stacked-N4: (N, 4) row-major stack + argmin(axis=1) +
  take_along_axis gather (layout-alternative formulation).

Data per N is finite (seeded normal keys + exact ties), so all lanes
agree bit-exactly (asserted per N); NaN/Inf semantics are proven by
tools/parity_rowmin.py, not re-timed here.

Reports per N: total ms (median of warm reps), speedup vs numpy-ref,
analytic temp bytes + tracemalloc peak at N=1M, cold (first call) vs
warm, native path vs forced fallback. Kernel is single-threaded by
design (streaming, row-independent) -> no thread scaling to report.

Seed 42 everywhere. Prints table + stage breakdown, writes
results/bench_rowmin.json. Exit 0 = PASS (all lanes agree + json).
"""
import importlib.util
import json
import os
import statistics
import time
import tracemalloc

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WRAP = _load("nf_rowwise_min4",
             os.path.normpath(os.path.join(ROOT, "..", "src", "Drivers",
                                           "CPU", "_lib", "rowwise_min4.py")))

SIZES = [50_000, 100_000, 200_000, 500_000, 1_000_000, 5_000_000]
WARM_REPS = 7


def make_data(n, rng):
    T = [np.ascontiguousarray(
        rng.integers(-2 ** 31, 2 ** 31, size=n).astype(np.int32))
        for _ in range(4)]
    D = [np.ascontiguousarray(rng.normal(0, 20, size=n).astype(np.float32))
         for _ in range(4)]
    tie = rng.random(size=n) < 0.3
    D[1][tie] = D[0][tie]
    tie2 = rng.random(size=n) < 0.2
    D[3][tie2] = D[2][tie2]
    return T, D


def lane_ref(T, D):
    Ts = np.stack(T)
    Ds = np.stack(D)
    m = np.argmin(Ds, axis=0).astype(np.uint8)
    idx = np.arange(T[0].size)
    return (np.ascontiguousarray(Ts[m, idx]),
            np.ascontiguousarray(Ds[m, idx]),
            np.ascontiguousarray(m))


def lane_native(T, D):
    return WRAP.rowwise_min4_argmin_gather(*T, *D)


def lane_segmented(T, D):
    n = T[0].size
    flat = np.concatenate(D)  # (4N,) lane-major
    bounds = np.arange(0, 4 * n + 1, n)
    dmin = np.minimum.reduceat(flat, bounds[:-1])
    # first-equal cascade, NaN-free data: earliest lane equal to dmin wins
    m = np.zeros(n, dtype=np.uint8)
    seen = np.zeros(n, dtype=bool)
    for k in range(4):
        take = (D[k] == dmin) & ~seen
        m = np.where(take, np.uint8(k), m)
        seen = seen | take
    tb = T[0].copy()
    for k in (1, 2, 3):
        tb = np.where(m == k, T[k], tb)
    db = D[0].copy()
    for k in (1, 2, 3):
        db = np.where(m == k, D[k], db)
    return (np.ascontiguousarray(tb), np.ascontiguousarray(db),
            np.ascontiguousarray(m))


def lane_stacked_n4(T, D):
    Tn = np.stack(T, axis=1)  # (N, 4) row-major
    Dn = np.stack(D, axis=1)
    m = np.argmin(Dn, axis=1).astype(np.uint8)
    mi = m.astype(np.int64)[:, None]
    tb = np.take_along_axis(Tn, np.broadcast_to(mi, (Tn.shape[0], 1)),
                            axis=1)[:, 0]
    db = np.take_along_axis(Dn, np.broadcast_to(mi, (Dn.shape[0], 1)),
                            axis=1)[:, 0]
    return (np.ascontiguousarray(tb), np.ascontiguousarray(db),
            np.ascontiguousarray(m))


def bit_exact(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def time_lane(fn, T, D):
    fn(*T, *D)  # cold (probe + page faults included)
    t0 = time.perf_counter()
    fn(*T, *D)
    cold_ms = (time.perf_counter() - t0) * 1e3
    reps = []
    for _ in range(WARM_REPS):
        t = time.perf_counter()
        fn(*T, *D)
        reps.append((time.perf_counter() - t) * 1e3)
    return cold_ms, statistics.median(reps)


def main():
    rng = np.random.default_rng(42)
    native_on = WRAP.rowmin_available()
    print(f"native backend: {native_on} ({WRAP.why()})")
    fails = []
    table = []
    for n in SIZES:
        try:
            T, D = make_data(n, rng)
        except MemoryError:
            print(f"SKIP N={n} (data alloc failed)")
            continue
        row = {"n": n}
        try:
            ref = lane_ref(T, D)
            seg = lane_segmented(T, D)
            alt = lane_stacked_n4(T, D)
            nat = lane_native(T, D)
        except MemoryError:
            print(f"SKIP N={n} (lane alloc failed)")
            continue
        agree = all(bit_exact(a, b)
                    for a, b in (*zip(ref, seg), *zip(ref, alt), *zip(ref, nat)))
        row["agree_bitexact"] = bool(agree)
        print(("PASS " if agree else "FAIL ") + f"N={n} lanes-agree")
        if not agree:
            fails.append(f"agree@{n}")
        for label, fn in (("ref", lane_ref), ("native", lane_native),
                          ("segmented", lane_segmented),
                          ("stacked-N4", lane_stacked_n4)):
            cold_ms, warm_ms = time_lane(fn, T, D)
            row[f"{label}_cold_ms"] = cold_ms
            row[f"{label}_warm_ms"] = warm_ms
        row["speedup_native"] = row["ref_warm_ms"] / row["native_warm_ms"]
        row["speedup_segmented"] = (row["ref_warm_ms"]
                                    / row["segmented_warm_ms"])
        row["speedup_stackedN4"] = (row["ref_warm_ms"]
                                    / row["stacked-N4_warm_ms"])
        # analytic bytes: inputs 32N + outputs 9N + lane temps
        row["input_bytes"] = 8 * n * 4
        row["output_bytes"] = 4 * n + 4 * n + n
        row["temp_bytes"] = {"ref": 2 * 4 * n * 4,
                             "native": 0,
                             "segmented": 4 * n * 4 + 2 * n,
                             "stacked-N4": 2 * 4 * n * 4 + 8 * n}
        table.append(row)
        print(f"N={n:>8} warm_ms ref={row['ref_warm_ms']:.3f} "
              f"native={row['native_warm_ms']:.3f} "
              f"(x{row['speedup_native']:.2f}) "
              f"segmented={row['segmented_warm_ms']:.3f} "
              f"(x{row['speedup_segmented']:.2f}) "
              f"stacked-N4={row['stacked-N4_warm_ms']:.3f} "
              f"(x{row['speedup_stackedN4']:.2f}) "
              f"cold_native={row['native_cold_ms']:.3f}")
        del T, D, ref, seg, alt, nat

    # tracemalloc peak at N=1M (grounds the analytic temps)
    mem = {}
    T, D = make_data(1_000_000, np.random.default_rng(42))
    for label, fn in (("ref", lane_ref), ("native", lane_native),
                      ("segmented", lane_segmented),
                      ("stacked-N4", lane_stacked_n4)):
        tracemalloc.start()
        fn(*T, *D)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem[label] = peak
    print(f"tracemalloc peak @1M: {mem}")

    # native vs forced-fallback @1M (same process, env-aware probe)
    T, D = make_data(1_000_000, np.random.default_rng(7))
    _, w_native = time_lane(lane_native, T, D)
    os.environ["NUMFAST_NATIVE_DISABLE"] = "1"
    try:
        _, w_fallback = time_lane(lane_native, T, D)
    finally:
        del os.environ["NUMFAST_NATIVE_DISABLE"]
    print(f"@1M native_path={w_native:.3f}ms forced_fallback={w_fallback:.3f}ms "
          f"native_backend={native_on}")

    json.dump({"seed": 42, "warm_reps": WARM_REPS,
               "native_backend": native_on, "why": WRAP.why(),
               "table": table, "tracemalloc_peak_1M": mem,
               "at_1M": {"native_path_ms": w_native,
                         "forced_fallback_ms": w_fallback},
               "threads": "single-threaded kernel, no scaling surface",
               "pass": not fails, "fails": fails},
              open(os.path.join(RES, "bench_rowmin.json"), "w"), indent=1)
    print("PASS" if not fails else f"FAIL {fails}")
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
