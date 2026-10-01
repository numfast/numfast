# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scaling audit: resident-accumulate chunk-size x Q grid, N=100M synthetic seed 42.

Bench-only (no core/WGSL/Planner/Runtime/calibration change). Reuses
groupby_resident_accumulate / _packed from src/Drivers/GPU/_lib/gpu.py.
Grid: Q1 M=100 1-sum | Q2 M=10K packed 1-sum | Q3 M=1M 1-sum |
Q5 M=1M 2-sum (v1,v2, 2x resident calls on same keys) --
each x chunk_rows 5M/10M/20M (20/10/5 chunks). Per point: fwd stats
(total/H2D/kernel/D2H/merge, d2h_bytes=16*M, num_chunks), exact vs numpy
bincount, reversed-order bit-exact check. No chunked baseline (already in
bench_gpu_resident_accum_100M.py), no 1B, no ClickBench.

Usage (Git Bash, strictly sequential):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    timeout 550 /c/Users/Mikech/AppData/Local/Programs/Python/Python314/python.exe \
    tests/heavy/bench_gpu_resident_scaling_100M.py
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

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "nfgpu_resident_accum",
    str(FORK / "src" / "Drivers" / "GPU" / "_lib" / "gpu.py"))
G = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(G)

OUT = FORK / "tests" / "heavy" / "bench_gpu_resident_scaling_100M.json"
SEED = 42
N = 100_000_000
# 20M chunk requested but NOT runnable: single dispatch per chunk,
# 20M/256 = 78125 groups > WebGPU limit 65535 (ceiling 65535*256 =
# 16776960 rows). Observed GPUValidationError on Q1-20M (evidence below).
# Grid uses max feasible 16M (62500 groups, 7 chunks) instead. No core change.
CHUNKS = (5_000_000, 10_000_000, 16_000_000)
CHUNK20M_EVIDENCE = {
    "supported": False,
    "reason": "dispatch [78125,1,1] > 65535 (20M/256); ceiling 16776960 rows",
    "observed": "Q1-20M fwd GPUValidationError wgpuCommandEncoderFinish/dispatch",
}


def exact_ok(uk, cc, ss, ref_counts, ref_sums):
    return bool(((uk == np.flatnonzero(ref_counts)).all()
                 and (cc == ref_counts[uk]).all()
                 and (ss == ref_sums[uk]).all()))


def total_of(st):
    return st["h2d_ms"] + st["kernel_ms"] + st["d2h_ms"] + st["merge_ms"]


def run_point(label, keys, vals, m, ref_counts, ref_sums, chunk, packed=None):
    """One (Q, chunk) point: fwd + reversed. Returns dict row."""
    nch = (N + chunk - 1) // chunk
    rev = list(reversed(range(nch)))
    if packed is None:
        uk, cc, ss, st = G.groupby_resident_accumulate(
            keys, vals, m=m, chunk_rows=chunk)
        ukr, ccr, ssr, _ = G.groupby_resident_accumulate(
            keys, vals, m=m, chunk_rows=chunk, chunk_order=rev)
    else:
        a1, a2 = packed
        uk, cc, ss, st = G.groupby_resident_accumulate_packed(
            a1, a2, vals, chunk_rows=chunk)
        ukr, ccr, ssr, _ = G.groupby_resident_accumulate_packed(
            a1, a2, vals, chunk_rows=chunk, chunk_order=rev)
    ok = exact_ok(uk, cc, ss, ref_counts, ref_sums)
    rev_ok = bool((uk == ukr).all() and (cc == ccr).all() and (ss == ssr).all())
    if not ok:
        print(f"{label} INTEGRITY STOP", flush=True)
        sys.exit(1)
    row = {"chunk_rows": chunk, "num_chunks": st["num_chunks"],
           "m": st["m"], "h2d_ms": st["h2d_ms"], "kernel_ms": st["kernel_ms"],
           "d2h_ms": st["d2h_ms"], "merge_ms": st["merge_ms"],
           "total_ms": total_of(st), "d2h_bytes": st["d2h_bytes"],
           "exact_vs_numpy": ok, "reversed_bitexact": rev_ok}
    print(f"  {label} chunk={chunk//1_000_000}M nch={st['num_chunks']} "
          f"total={row['total_ms']:.0f} h2d={st['h2d_ms']:.0f} "
          f"k={st['kernel_ms']:.0f} d2h={st['d2h_ms']:.1f} "
          f"merge={st['merge_ms']:.3f} d2hB={st['d2h_bytes']} "
          f"exact={ok} rev={rev_ok}", flush=True)
    return row


def main():
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "n": N, "chunks": list(CHUNKS),
           "device": "RTX 2060 Vulkan", "queries": {},
           "chunk_20M": CHUNK20M_EVIDENCE}
    wk = np.arange(1024, dtype=np.int32)
    wv = np.arange(1024, dtype=np.int32)
    G.groupby_resident_accumulate(wk, wv, chunk_rows=512)
    del wk, wv
    print("warmup done", flush=True)

    # ---- Q1 M=100 ----
    print("=== Q1 M=100 ===", flush=True)
    t = time.perf_counter()
    k1 = rng.integers(0, 100, N, dtype=np.int32)
    v1 = rng.integers(-500, 500, N, dtype=np.int32)
    print(f"  gen {(time.perf_counter() - t) * 1000:.0f}ms", flush=True)
    t = time.perf_counter()
    rc = np.bincount(k1.astype(np.int64), minlength=100)
    rs = np.bincount(k1.astype(np.int64), weights=v1.astype(np.float64),
                     minlength=100).astype(np.int64)
    print(f"  numpy-ref {(time.perf_counter() - t) * 1000:.0f}ms", flush=True)
    res["queries"]["Q1"] = {"m": 100, "points": [
        run_point("Q1", k1, v1, 100, rc, rs, c) for c in CHUNKS]}
    del k1, v1, rc, rs
    gc.collect()

    # ---- Q2 M=10K packed ----
    print("=== Q2 M=10K packed ===", flush=True)
    t = time.perf_counter()
    a1 = rng.integers(0, 100, N, dtype=np.int32)
    a2 = rng.integers(0, 100, N, dtype=np.int32)
    w1 = rng.integers(-500, 500, N, dtype=np.int32)
    print(f"  gen {(time.perf_counter() - t) * 1000:.0f}ms", flush=True)
    m2 = int(a2.max()) + 1
    t = time.perf_counter()
    pk = a1.astype(np.int64) * m2 + a2.astype(np.int64)
    rc = np.bincount(pk)
    rs = np.bincount(pk, weights=w1.astype(np.float64)).astype(np.int64)
    print(f"  numpy-ref {(time.perf_counter() - t) * 1000:.0f}ms", flush=True)
    res["queries"]["Q2"] = {"m": 10_000, "m2": m2, "points": [
        run_point("Q2", None, w1, None, rc, rs, c, packed=(a1, a2))
        for c in CHUNKS]}
    del a1, a2, w1, pk, rc, rs
    gc.collect()

    # ---- Q3 M=1M ----
    print("=== Q3 M=1M ===", flush=True)
    t = time.perf_counter()
    k3 = rng.integers(0, 1_000_000, N, dtype=np.int32)
    v3 = rng.integers(-500, 500, N, dtype=np.int32)
    print(f"  gen {(time.perf_counter() - t) * 1000:.0f}ms", flush=True)
    t = time.perf_counter()
    rc = np.bincount(k3.astype(np.int64), minlength=1_000_000)
    rs = np.bincount(k3.astype(np.int64), weights=v3.astype(np.float64),
                     minlength=1_000_000).astype(np.int64)
    print(f"  numpy-ref {(time.perf_counter() - t) * 1000:.0f}ms", flush=True)
    res["queries"]["Q3"] = {"m": 1_000_000, "points": [
        run_point("Q3", k3, v3, 1_000_000, rc, rs, c) for c in CHUNKS]}
    del k3, v3, rc, rs
    gc.collect()

    # ---- Q5 M=1M 2-sum (v1+v2, 2x resident calls, same keys) ----
    print("=== Q5 M=1M 2-sum ===", flush=True)
    t = time.perf_counter()
    k5 = rng.integers(0, 1_000_000, N, dtype=np.int32)
    u1 = rng.integers(-500, 500, N, dtype=np.int32)
    u2 = rng.integers(-500, 500, N, dtype=np.int32)
    print(f"  gen {(time.perf_counter() - t) * 1000:.0f}ms", flush=True)
    t = time.perf_counter()
    rc = np.bincount(k5.astype(np.int64), minlength=1_000_000)
    rs1 = np.bincount(k5.astype(np.int64), weights=u1.astype(np.float64),
                      minlength=1_000_000).astype(np.int64)
    rs2 = np.bincount(k5.astype(np.int64), weights=u2.astype(np.float64),
                      minlength=1_000_000).astype(np.int64)
    print(f"  numpy-ref {(time.perf_counter() - t) * 1000:.0f}ms", flush=True)
    pts = []
    for c in CHUNKS:
        r1 = run_point("Q5-s1", k5, u1, 1_000_000, rc, rs1, c)
        r2 = run_point("Q5-s2", k5, u2, 1_000_000, rc, rs2, c)
        pts.append({"chunk_rows": c, "num_chunks": r1["num_chunks"],
                    "m": 1_000_000, "nsums": 2,
                    "h2d_ms": r1["h2d_ms"] + r2["h2d_ms"],
                    "kernel_ms": r1["kernel_ms"] + r2["kernel_ms"],
                    "d2h_ms": r1["d2h_ms"] + r2["d2h_ms"],
                    "merge_ms": r1["merge_ms"] + r2["merge_ms"],
                    "total_ms": r1["total_ms"] + r2["total_ms"],
                    "d2h_bytes": r1["d2h_bytes"] + r2["d2h_bytes"],
                    "exact_vs_numpy": r1["exact_vs_numpy"] and r2["exact_vs_numpy"],
                    "reversed_bitexact": r1["reversed_bitexact"] and r2["reversed_bitexact"],
                    "per_sum": [r1, r2]})
        r = pts[-1]
        print(f"  Q5-combined chunk={c//1_000_000}M nch={r['num_chunks']} "
              f"total={r['total_ms']:.0f} h2d={r['h2d_ms']:.0f} "
              f"k={r['kernel_ms']:.0f} d2h={r['d2h_ms']:.1f} "
              f"merge={r['merge_ms']:.3f} d2hB={r['d2h_bytes']} "
              f"exact={r['exact_vs_numpy']} rev={r['reversed_bitexact']}",
              flush=True)
    res["queries"]["Q5"] = {"m": 1_000_000, "nsums": 2, "points": pts}
    del k5, u1, u2, rc, rs1, rs2
    gc.collect()

    with open(OUT, "w") as f:
        json.dump(json.loads(json.dumps(res, default=float)), f, indent=1)
    print(f"JSON written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
