# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streaming GroupBy resident-accumulate vs chunked-with-merge, 100M (heavy).

SPEC-DELTA-11 path (new, beside existing chunked): counts[M]+hi[M]+lo[M]+
carry[M] allocated ONCE on device; each chunk H2D keys/hi/lo -> atomicAdd
into resident accums -> chunk bufs freed; end: single M-sized D2H (4*M)
-> host int64 finalize. Baseline = existing chunked-with-merge (per-chunk
groupby_dense_impl + host _merge_partial; its internals NOT instrumented --
existing code must not be rewritten, so chunked reports total + merge +
analytic D2H bytes, resident reports full H2D/kernel/D2H/merge split).

Synthetic seed 42, int32, N=100M, chunk_rows=10M (10 chunks, streaming):
  Q1 M=100 | Q2 M=10K packed (global m2) | Q3 M=1M.
Integrity exact first (mismatch = STOP). Contention micro-probe: fixed
N=4M, M in (100, 10K, 1M) -> kernel ns/row. No 1B, no ClickBench, no NFS,
no Planner/calibration change, no CPU fallback.

Usage (Git Bash, strictly sequential, timeout):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    timeout 1500 /c/Users/Mikech/AppData/Local/Programs/Python/Python314/python.exe \
    tests/heavy/bench_gpu_resident_accum_100M.py
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

OUT = FORK / "tests" / "heavy" / "bench_gpu_resident_accum_100M.json"
SEED = 42
N = 100_000_000
CHUNK = 10_000_000
TILE = 2048  # mirrors _GB_TILE for analytic chunked D2H (tiled path M<=2048)


def chunked_merge(keys, vals, m, chunk):
    """Existing-path baseline: per-chunk groupby_dense_impl + host merge."""
    t0 = time.perf_counter()
    merge_ms = 0.0
    acc = {}
    n = int(keys.size)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        c, (s,) = G.groupby_dense_impl(keys[a:b], [(vals[a:b], "int32")])
        t1 = time.perf_counter()
        G._merge_partial(acc, "__counts__", c)
        G._merge_partial(acc, "__sums__", s)
        merge_ms += (time.perf_counter() - t1) * 1000
    total_ms = (time.perf_counter() - t0) * 1000
    counts = acc["__counts__"]
    sums = acc["__sums__"]
    uk, cc, (ss,) = G._gb_compact(counts, [sums])
    return uk, cc, ss, total_ms, merge_ms


def chunked_d2h_bytes(m, n, chunk):
    """Analytic per-chunk D2H of the existing path (no rewrite => analytic)."""
    nch = (n + chunk - 1) // chunk
    if m <= 2048:
        per = ((chunk + TILE - 1) // TILE) * m * 8  # phi i32 + plo u32
    else:
        per = m * 8  # ghi i32 + glo u32, one readback per chunk-call
    return nch * per, nch


def check_exact(label, uk, cc, ss, ref_counts, ref_sums):
    if not ((uk == np.flatnonzero(ref_counts)).all()
            and (cc == ref_counts[uk]).all() and (ss == ref_sums[uk]).all()):
        print(f"{label} INTEGRITY STOP")
        sys.exit(1)


def main():
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "n": N, "chunk_rows": CHUNK, "device": "RTX 2060 Vulkan",
           "queries": {}, "contention": {}}
    # GPU warmup (compile resident kernel once, off the clock)
    wk = np.arange(1024, dtype=np.int32)
    wv = np.arange(1024, dtype=np.int32)
    G.groupby_resident_accumulate(wk, wv, chunk_rows=512)
    G.groupby_dense_impl(wk, [(wv, "int32")])
    print("warmup done", flush=True)

    # ---- contention micro-probe first (cheap, fixed N=4M) ----
    print("=== contention probe N=4M ===", flush=True)
    nm = 4_000_000
    vm = rng.integers(-500, 500, nm, dtype=np.int32)
    for m in (100, 10_000, 1_000_000):
        km = rng.integers(0, m, nm, dtype=np.int32)
        G.groupby_resident_accumulate(km, vm, chunk_rows=1_000_000)  # warm
        _, _, _, st = G.groupby_resident_accumulate(km, vm, chunk_rows=1_000_000)
        ns = st["kernel_ms"] * 1e6 / nm
        res["contention"][f"M={m}"] = {"kernel_ms": st["kernel_ms"],
                                       "ns_per_row": ns,
                                       "h2d_ms": st["h2d_ms"],
                                       "d2h_ms": st["d2h_ms"]}
        print(f"  M={m}: kernel={st['kernel_ms']:.1f}ms "
              f"{ns:.1f}ns/row h2d={st['h2d_ms']:.1f}ms", flush=True)
    del vm
    gc.collect()

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
    uk, cc, ss, st = G.groupby_resident_accumulate(k1, v1, chunk_rows=CHUNK)
    check_exact("Q1-resident", uk, cc, ss, rc, rs)
    t = time.perf_counter()
    uk0, cc0, ss0, bt, bm = chunked_merge(k1, v1, 100, CHUNK)
    bd, bnc = chunked_d2h_bytes(100, N, CHUNK)
    check_exact("Q1-chunked", uk0, cc0, ss0, rc, rs)
    assert (uk == uk0).all() and (cc == cc0).all() and (ss == ss0).all()
    res["queries"]["Q1"] = {
        "m": 100, "resident": {**st, "total_ms": st["h2d_ms"] + st["kernel_ms"]
                               + st["d2h_ms"] + st["merge_ms"]},
        "chunked": {"total_ms": bt, "merge_ms": bm, "d2h_bytes": bd,
                    "num_chunks": bnc,
                    "note": "internals not instrumented (no rewrite)"},
        "speedup": bt / max(st["h2d_ms"] + st["kernel_ms"] + st["d2h_ms"]
                            + st["merge_ms"], 1e-9),
    }
    print(f"  resident total={res['queries']['Q1']['resident']['total_ms']:.0f}ms "
          f"(h2d={st['h2d_ms']:.0f} k={st['kernel_ms']:.0f} "
          f"d2h={st['d2h_ms']:.0f} merge={st['merge_ms']:.3f}) d2hB={st['d2h_bytes']}",
          flush=True)
    print(f"  chunked total={bt:.0f}ms merge={bm:.0f}ms d2hB={bd} "
          f"speedup={res['queries']['Q1']['speedup']:.2f}x", flush=True)
    del k1, v1, rc, rs, uk, cc, ss, uk0, cc0, ss0
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
    uk, cc, ss, st = G.groupby_resident_accumulate_packed(a1, a2, w1,
                                                          chunk_rows=CHUNK)
    check_exact("Q2-resident", uk, cc, ss, rc, rs)
    t = time.perf_counter()
    acc = {}
    merge_ms = 0.0
    nch = (N + CHUNK - 1) // CHUNK
    for a in range(0, N, CHUNK):
        b = min(N, a + CHUNK)
        pkk = (a1[a:b].astype(np.int64) * m2 + a2[a:b].astype(np.int64)).astype(
            np.int32)
        c, (s,) = G.groupby_dense_impl(pkk, [(w1[a:b], "int32")])
        t1 = time.perf_counter()
        G._merge_partial(acc, "__counts__", c)
        G._merge_partial(acc, "__sums__", s)
        merge_ms += (time.perf_counter() - t1) * 1000
    bt = (time.perf_counter() - t) * 1000
    uk0, cc0, (ss0,) = G._gb_compact(acc["__counts__"], [acc["__sums__"]])
    check_exact("Q2-chunked", uk0, cc0, ss0, rc, rs)
    assert (uk == uk0).all() and (cc == cc0).all() and (ss == ss0).all()
    bd, _ = chunked_d2h_bytes(10_000, N, CHUNK)
    res["queries"]["Q2"] = {
        "m": 10_000, "m2": m2,
        "resident": {**st, "total_ms": st["h2d_ms"] + st["kernel_ms"]
                     + st["d2h_ms"] + st["merge_ms"]},
        "chunked": {"total_ms": bt, "merge_ms": merge_ms, "d2h_bytes": bd,
                    "num_chunks": nch,
                    "note": "per-chunk pack host + dense_impl + merge"},
        "speedup": bt / max(st["h2d_ms"] + st["kernel_ms"] + st["d2h_ms"]
                            + st["merge_ms"], 1e-9),
    }
    print(f"  resident total={res['queries']['Q2']['resident']['total_ms']:.0f}ms "
          f"(h2d={st['h2d_ms']:.0f} k={st['kernel_ms']:.0f} "
          f"d2h={st['d2h_ms']:.0f} merge={st['merge_ms']:.3f}) d2hB={st['d2h_bytes']}",
          flush=True)
    print(f"  chunked total={bt:.0f}ms merge={merge_ms:.0f}ms d2hB={bd} "
          f"speedup={res['queries']['Q2']['speedup']:.2f}x", flush=True)
    del a1, a2, w1, pk, rc, rs, uk, cc, ss, uk0, cc0, ss0
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
    uk, cc, ss, st = G.groupby_resident_accumulate(k3, v3, chunk_rows=CHUNK)
    check_exact("Q3-resident", uk, cc, ss, rc, rs)
    uk0, cc0, ss0, bt, bm = chunked_merge(k3, v3, 1_000_000, CHUNK)
    bd, bnc = chunked_d2h_bytes(1_000_000, N, CHUNK)
    check_exact("Q3-chunked", uk0, cc0, ss0, rc, rs)
    assert (uk == uk0).all() and (cc == cc0).all() and (ss == ss0).all()
    res["queries"]["Q3"] = {
        "m": 1_000_000,
        "resident": {**st, "total_ms": st["h2d_ms"] + st["kernel_ms"]
                     + st["d2h_ms"] + st["merge_ms"]},
        "chunked": {"total_ms": bt, "merge_ms": bm, "d2h_bytes": bd,
                    "num_chunks": bnc,
                    "note": "internals not instrumented (no rewrite)"},
        "speedup": bt / max(st["h2d_ms"] + st["kernel_ms"] + st["d2h_ms"]
                            + st["merge_ms"], 1e-9),
    }
    print(f"  resident total={res['queries']['Q3']['resident']['total_ms']:.0f}ms "
          f"(h2d={st['h2d_ms']:.0f} k={st['kernel_ms']:.0f} "
          f"d2h={st['d2h_ms']:.0f} merge={st['merge_ms']:.3f}) d2hB={st['d2h_bytes']}",
          flush=True)
    print(f"  chunked total={bt:.0f}ms merge={bm:.0f}ms d2hB={bd} "
          f"speedup={res['queries']['Q3']['speedup']:.2f}x", flush=True)
    del k3, v3, rc, rs, uk, cc, ss, uk0, cc0, ss0
    gc.collect()

    with open(OUT, "w") as f:
        json.dump(json.loads(json.dumps(res, default=float)), f, indent=1)
    print(f"JSON written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
