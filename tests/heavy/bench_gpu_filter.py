# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU Filter/Mask-scan bench (SPEC-DELTA-8, NEW file, bench-only).

Compares compact paths on RTX 2060 (Vulkan, warm, seed 42):
  A host-assisted (baseline): full-mask D2H + host flatnonzero + GPU gather.
  B block-partials (new): GPU counts (mask resident) + tiny W D2H + host
    W-prefix + GPU scatter (single M-sized D2H).
  CPU: numpy boolean-mask reference (NUMFAST_THREADS=1 and =16 sections;
    numpy take is ST -- reported verbatim, no fake scaling).
Stages for B@10M/50%: H2D / counts-dispatch / W-D2H / host-prefix /
scatter-dispatch / M-D2H separately. Chain: Filter->GroupBy sum fully GPU
(resident, no N round-trip) vs CPU round-trip vs B+std-groupby.
N-scale 256K->10M. Crossover 256..8192 validates _FILTER_HOST_MAX=2048.
Integrity: exact match vs numpy before timing (loss>1% = STOP).

Usage (Git Bash, sequential, timeout -- heavy jobs strictly serial):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    timeout 550 /c/Users/Mikech/AppData/Local/Programs/Python/Python314/python.exe \
    tests/heavy/bench_gpu_filter.py
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
for _p in (str(FORK.parent / "app-builder-ponytail"), str(FORK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

OUT = FORK / "tests" / "heavy" / "bench_gpu_filter.json"
SEED = 42
REPS = 5
WARM = 3
WG = 256

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "nfgpu_bench", str(FORK / "src" / "Drivers" / "GPU" / "_lib" / "gpu.py"))
G = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(G)


def med(fn, reps=REPS, warm=WARM):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def cpu_filter(V, m):
    return V[m.astype(bool)]


def gpu_a(V, m):
    """Baseline A: host flatnonzero + GPU gather (mask D2H implicit in path)."""
    ix = np.flatnonzero(m.astype(bool)).astype(np.int32)
    return G.gather(V, ix, "int32")


def gpu_b(V, m):
    return G.filter_compact(V, m, "int32")


def staged_b(V, m):
    """Stage split of path B at fixed buffers (ms + bytes)."""
    import wgpu

    dev = G._device()
    st = {}
    mk = np.ascontiguousarray(m.astype(np.int32))
    n = int(mk.size)
    w = (n + WG - 1) // WG
    t = time.perf_counter()
    b_mk = dev.create_buffer_with_data(
        data=mk.tobytes(), usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
    b_v = dev.create_buffer_with_data(
        data=V.tobytes(), usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
    st["h2d_ms"] = (time.perf_counter() - t) * 1000
    st["h2d_bytes"] = 2 * n * 4
    t = time.perf_counter()
    counts = G._filter_counts_gpu(dev, mk)
    st["counts_ms"] = (time.perf_counter() - t) * 1000
    st["counts_d2h_bytes"] = w * 4
    t = time.perf_counter()
    offs, total = G._filter_prefix(counts)
    st["prefix_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    src = G._FILTER_SCATTER_WGSL.replace("{t}", "i32").replace(
        "{_WG}", str(WG)).replace("{n}", str(n))
    out0 = np.zeros(total, dtype=np.int32)
    (rb,) = G._run(dev, src, [(V.tobytes(), True), (mk.tobytes(), True),
                              (offs.tobytes(), True),
                              (out0.tobytes(), False)], n, [])
    st["scatter_ms"] = (time.perf_counter() - t) * 1000
    st["d2h_bytes"] = total * 4
    st["W"] = w
    st["M"] = total
    del b_mk, b_v
    return np.frombuffer(rb, dtype=np.int32).copy(), st


def main():
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "device": "RTX 2060 Vulkan", "reps": REPS,
           "warm": WARM, "sections": {}}
    # ---- integrity gate (exact, STOP on mismatch) ----
    Vg = rng.integers(0, 1000, 1_000_000, dtype=np.int32)
    mg = (Vg > 500).astype(np.int32)
    assert (gpu_b(Vg, mg) == Vg[Vg > 500]).all(), "B integrity FAIL"
    assert (gpu_a(Vg, mg) == Vg[Vg > 500]).all(), "A integrity FAIL"
    print("integrity gate: exact 1M/50% OK", flush=True)

    # ---- selectivity ladder @10M ----
    N = 10_000_000
    V = rng.integers(0, 1000, N, dtype=np.int32)
    lad = {}
    for sel, thr in (("0", 1001), ("1", 990), ("10", 900), ("50", 500),
                     ("90", 100), ("100", -1)):
        m = (V > thr).astype(np.int32)
        M = int(m.sum())
        assert (gpu_b(V, m) == V[V > thr]).all(), f"B {sel}% FAIL"
        row = {"M": M}
        row["cpu_ms"] = med(lambda: cpu_filter(V, m))
        row["gpuA_ms"] = med(lambda: gpu_a(V, m))
        row["gpuB_ms"] = med(lambda: gpu_b(V, m))
        row["speedup_B_vs_cpu"] = row["cpu_ms"] / max(row["gpuB_ms"], 1e-9)
        lad[sel] = row
        print(f"ladder {sel}% M={M} cpu={row['cpu_ms']:.1f} "
              f"A={row['gpuA_ms']:.1f} B={row['gpuB_ms']:.1f} "
              f"x{row['speedup_B_vs_cpu']:.2f}", flush=True)
        gc.collect()
    res["sections"]["ladder_10M"] = lad

    # ---- staged split B @10M/50% ----
    m50 = (V > 500).astype(np.int32)
    got, st = staged_b(V, m50)
    assert (got == V[V > 500]).all()
    res["sections"]["staged_10M_50"] = st
    print("staged: " + json.dumps(st), flush=True)

    # ---- N-scale @50% ----
    scale = {}
    for N2 in (262144, 1_048_576, 4_194_304, 10_000_000):
        V2 = V[:N2].copy()
        m2 = (V2 > 500).astype(np.int32)
        assert (gpu_b(V2, m2) == V2[V2 > 500]).all(), f"B N={N2} FAIL"
        row = {"M": int(m2.sum())}
        row["cpu_ms"] = med(lambda: cpu_filter(V2, m2))
        row["gpuB_ms"] = med(lambda: gpu_b(V2, m2))
        scale[str(N2)] = row
        print(f"N={N2} cpu={row['cpu_ms']:.1f} B={row['gpuB_ms']:.1f}",
              flush=True)
        gc.collect()
    res["sections"]["nscale_50"] = scale

    # ---- crossover 256..8192 (validates _FILTER_HOST_MAX=2048) ----
    cross = {}
    for N3 in (256, 512, 1024, 2048, 4096, 8192):
        V3 = rng.integers(0, 100, N3, dtype=np.int32)
        m3 = (V3 > 50).astype(np.int32)
        cross[str(N3)] = {"A_ms": med(lambda: gpu_a(V3, m3)),
                          "B_ms": med(lambda: gpu_b(V3, m3))}
        print(f"cross N={N3} A={cross[str(N3)]['A_ms']:.2f} "
              f"B={cross[str(N3)]['B_ms']:.2f}", flush=True)
    res["sections"]["crossover"] = cross

    # ---- CPU 1T vs 16T note (numpy take is ST) ----
    os.environ["NUMFAST_THREADS"] = "16"
    t16 = med(lambda: cpu_filter(V, m50))
    os.environ["NUMFAST_THREADS"] = "1"
    t1 = med(lambda: cpu_filter(V, m50))
    res["sections"]["cpu_threads"] = {"cpu_1T_ms": t1, "cpu_16T_ms": t16,
                                      "note": "numpy take ST; env tribal only"}
    print(f"cpu 1T={t1:.1f} 16T={t16:.1f}", flush=True)

    # ---- chain Filter->GroupBy @10M sel=10% ----
    K = rng.integers(0, 1000, N, dtype=np.int32)
    mc = (V > 900).astype(np.int32)
    Vs, Ks = V[mc.astype(bool)], K[mc.astype(bool)]
    ref_counts = np.bincount(Ks.astype(np.int64), minlength=1000)
    nz = int((ref_counts > 0).sum())

    def chain_cpu_rt():
        ix = np.flatnonzero(mc.astype(bool)).astype(np.int32)
        ck = K[ix]
        cv = V[ix]
        counts, (sums,) = G.groupby_dense_impl(
            ck, [(cv, "int32")])
        return counts, sums

    def chain_b_std():
        ck = gpu_b(K, mc)
        cv = gpu_b(V, mc)
        counts, (sums,) = G.groupby_dense_impl(
            ck, [(cv, "int32")])
        return counts, sums

    def chain_resident():
        return G.filter_groupby_resident(K, V, mc, "sum")

    c0, s0 = chain_cpu_rt()
    c1, s1 = chain_b_std()
    uk2, cc2, ss2 = chain_resident()
    assert (c0 == c1).all() and (s0 == s1).all(), "chain B!=CPU"
    d2 = {int(k): int(v) for k, v in zip(uk2.tolist(), ss2.tolist())}
    d0 = {i: int(s0[i]) for i in range(len(s0)) if c0[i] > 0}
    assert d2 == d0, "resident chain != CPU"
    ch = {"groups": nz, "M": int(mc.sum())}
    ch["cpu_roundtrip_ms"] = med(chain_cpu_rt)
    ch["gpuB_stdgroupby_ms"] = med(chain_b_std)
    ch["resident_ms"] = med(chain_resident)
    ch["speedup_resident_vs_cpu_rt"] = (ch["cpu_roundtrip_ms"]
                                        / max(ch["resident_ms"], 1e-9))
    res["sections"]["chain_10M_sel10"] = ch
    print("chain: " + json.dumps(ch), flush=True)

    # ---- full chain predicate-inclusive: CPU IR vs GPU graph vs resident ----
    from builder import MAIN as _MAIN

    _k = _MAIN["build"](str(FORK))
    _a = _k.alias
    _jobs = [_a["ir_series"]("v", V), _a["ir_series"]("k", K),
             _a["ir_compare"]("m", "v", 900, ">"),
             _a["ir_filter"]("fv", "v", "m"),
             _a["ir_filter"]("fk", "k", "m")]
    _g = _a["compile"](_jobs)

    def chain_cpu_full():
        b = _a["cpu_execute"](_g["nodes"])
        kk = np.asarray(b["fk"]).astype(np.int64)
        vv = np.asarray(b["fv"])
        return (np.bincount(kk, minlength=1000),
                np.bincount(kk, weights=vv.astype(np.float64), minlength=1000))

    def chain_gpu_graph():
        b = _a["gpu_execute"](_g["nodes"])
        ck = np.asarray(b["fk"], dtype=np.int32)
        cv = np.asarray(b["fv"], dtype=np.int32)
        return G.groupby_dense_impl(ck, [(cv, "int32")])

    def chain_resident_full():
        mk = G.compare(V, 900, ">", "int32")
        return G.filter_groupby_resident(K, V, mk, "sum")

    c0, s0 = chain_cpu_full()
    c1, (s1,) = chain_gpu_graph()
    uk3, cc3, ss3 = chain_resident_full()
    assert (c0 == c1).all() and (s0 == s1).all(), "gpu graph != cpu"
    d3 = {int(k): int(v) for k, v in zip(uk3.tolist(), ss3.tolist())}
    assert d3 == {i: int(s0[i]) for i in range(len(s0)) if c0[i] > 0}
    full = {"groups": int((c0 > 0).sum()), "M": int((V > 900).sum())}
    full["cpu_full_ms"] = med(chain_cpu_full)
    full["gpu_graph_ms"] = med(chain_gpu_graph)
    full["resident_full_ms"] = med(chain_resident_full)
    res["sections"]["chain_full_10M_sel10"] = full
    print("chain_full: " + json.dumps(full), flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
