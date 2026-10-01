# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU Lookup / semi-join bench (NEW file, bench-only, no core change).

N-scale 256K/1M/4M/10M, RTX 2060 Vulkan, warm, seed 42. Standalone
lookup_mask (kernel vs transfer split) + resident chain
lookup_filter_resident vs CPU numpy isin reference. Integrity exact
before timing (mismatch = STOP). No optimization-at-any-cost.

Usage (Git Bash, sequential, timeout):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    timeout 550 /c/App/numfast/.venv/Scripts/python tests/heavy/bench_gpu_lookup.py
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

OUT = FORK / "tests" / "heavy" / "bench_gpu_lookup.json"
SEED = 42
REPS = 5
WARM = 3
WG = 256
NS = [256_000, 1_000_000, 4_000_000, 10_000_000]

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "nfgpu_lookup_bench", str(FORK / "src" / "Drivers" / "GPU" / "_lib"
                               / "gpu.py"))
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


def staged_probe(U, P):
    """Kernel-vs-transfer split: H2D / probe-dispatch / mask-D2H (ms)."""
    import wgpu

    dev = G._device()
    n, k = int(P.size), int(U.size)
    st = {}
    t = time.perf_counter()
    b_p = dev.create_buffer_with_data(
        data=P.tobytes(), usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
    ub = U if k else np.zeros(1, dtype=np.int32)
    b_u = dev.create_buffer_with_data(
        data=ub.tobytes(), usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
    b_m = dev.create_buffer(
        size=n * 4, usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
    st["h2d_ms"] = (time.perf_counter() - t) * 1000
    st["h2d_bytes"] = n * 4 + max(k, 1) * 4
    t = time.perf_counter()
    G._chain_dispatch(dev, G._lookup_src(n, k),
                      [(b_p, True), (b_u, True), (b_m, False)], n)
    st["kernel_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    m = np.frombuffer(bytes(dev.queue.read_buffer(b_m)),
                      dtype=np.int32).copy()
    st["d2h_ms"] = (time.perf_counter() - t) * 1000
    st["d2h_bytes"] = n * 4
    del b_p, b_u, b_m
    return m, st


def main():
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "device": "RTX 2060 Vulkan", "reps": REPS,
           "warm": WARM, "rows": []}
    # fixed build domain (resident dict reused across N)
    B = rng.integers(0, 1_000_000, 1_000_000).astype(np.int32)
    U = G.lookup_build(B)
    res["build_M"] = int(B.size)
    res["dict_K"] = int(U.size)
    for n in NS:
        P = rng.integers(0, 1_000_000, n).astype(np.int32)
        V = rng.integers(-500, 500, n).astype(np.int32)
        # integrity exact before timing
        ref = G.lookup_ref(B, P)
        got, st = staged_probe(U, P)
        if got.shape != ref.shape or not (got == ref).all():
            print(f"INTEGRITY STOP at N={n}: mismatch")
            sys.exit(1)
        ch = G.lookup_filter_resident(B, P, V)
        refc = V[np.isin(P, U)]
        if ch.shape != refc.shape or not (ch == refc).all():
            print(f"CHAIN INTEGRITY STOP at N={n}")
            sys.exit(1)
        sel = float(got.mean()) if n else 0.0
        row = {"n": n, "selectivity": sel, "staged_ms": st}
        row["standalone_ms"] = med(lambda: G.lookup_mask(B, P))
        row["chain_ms"] = med(lambda: G.lookup_filter_resident(B, P, V))
        t = time.perf_counter()
        _ = np.isin(P, U).astype(np.int32)
        row["cpu_isin_ms"] = (time.perf_counter() - t) * 1000
        row["probe_Mrows_s"] = n / max(row["standalone_ms"], 1e-9) / 1000
        print(f"N={n} sel={sel:.3f} standalone={row['standalone_ms']:.1f}ms "
              f"(h2d={st['h2d_ms']:.1f} kernel={st['kernel_ms']:.1f} "
              f"d2h={st['d2h_ms']:.1f}) chain={row['chain_ms']:.1f}ms "
              f"cpu={row['cpu_isin_ms']:.1f}ms")
        res["rows"].append(row)
        del P, V, got, ch, ref, refc
        gc.collect()
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
