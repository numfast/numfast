# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent GPU graph bench (SPEC-DELTA-9, NEW file, bench-only).

Stages 1-6 evidence on RTX 2060 Vulkan, seed 42, warm, sequential:
  1. pipeline cache: same-shape repeat dispatch (pack/compare/sort/lookup)
     first-compile ms vs cached ms + pipeline_stats hits/misses.
  2. uniform N/K/M: different N reuses the SAME pipeline (misses flat).
  4. pack->groupby resident vs old pack-D2H-reupload path (Q2 round-trip).
  5. lookup intended use: cold one-shot vs upload-once probe-many (P1..P3).
  6. fused graphs: compare->mask->filter->groupby, sort->slice->gather.
Integrity exact before timing (mismatch = STOP). No 1B, no ClickBench,
no H2O polish, no Planner ML.

Usage (Git Bash, sequential, timeout):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    timeout 550 /c/App/numfast/.venv/Scripts/python tests/heavy/bench_gpu_persistent.py
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
    "nfgpu_persistent_bench",
    str(FORK / "src" / "Drivers" / "GPU" / "_lib" / "gpu.py"))
G = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(G)

OUT = FORK / "tests" / "heavy" / "bench_gpu_persistent.json"
SEED = 42
N = 1_000_000
N2 = 1_000_001  # different N: uniform path must NOT recompile
REPS = 5
WARM = 2


def best(fn, reps=REPS, warm=WARM):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def main():
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "device": "RTX 2060 Vulkan", "n": N,
           "reps": REPS, "warm": WARM, "stages": {}}
    st = res["stages"]

    k1 = rng.integers(0, 50, N, dtype=np.int32)
    k2 = rng.integers(0, 100, N, dtype=np.int32)
    m2 = int(k2.max()) + 1
    vv = rng.integers(0, 10, N, dtype=np.int32)
    A = rng.integers(0, 100, N, dtype=np.int32)

    # ---- 1+2. pipeline cache + uniform shapes ----
    s0 = G.pipeline_stats()
    t_first = time.perf_counter()
    G.pack_i32(k1, k2, m2)
    first_ms = (time.perf_counter() - t_first) * 1000
    s1 = G.pipeline_stats()
    mn, md = best(lambda: G.pack_i32(k1, k2, m2))
    s2 = G.pipeline_stats()
    # different N: literal path would compile a new pipeline; uniform reuses
    k1b = rng.integers(0, 50, N2, dtype=np.int32)
    k2b = rng.integers(0, 100, N2, dtype=np.int32)
    G.pack_i32(k1b, k2b, m2)
    s3 = G.pipeline_stats()
    st["pipeline_cache"] = {
        "first_compile_ms": first_ms, "cached_min_ms": mn,
        "cached_med_ms": md,
        "misses_after_first": s1["misses"] - s0["misses"],
        "hits_after_repeats": s2["hits"] - s1["hits"],
        "misses_after_new_shape": s3["misses"] - s2["misses"],
        "pipelines": s3["pipelines"],
    }
    # sort pipeline hit: two sorts share one pipeline
    S = rng.integers(0, 1000, 1 << 20, dtype=np.int32)
    G.sort_perm_impl(S)
    ss = G.pipeline_stats()
    G.sort_perm_impl(S)
    se = G.pipeline_stats()
    st["sort_pipeline_hit"] = {
        "misses_second_sort": se["misses"] - ss["misses"],
        "hits_second_sort": se["hits"] - ss["hits"],
    }

    # ---- 4. pack round-trip removed ----
    pk = (k1.astype(np.int64) * m2 + k2.astype(np.int64))
    refc = np.bincount(pk)
    nz = np.flatnonzero(refc)
    uk, cc, ss_ = G.pack_groupby_resident(k1, k2, vv, "sum")
    refs = np.bincount(pk, weights=vv.astype(np.float64)).astype(np.int64)
    if not ((uk == nz).all() and (cc == refc[nz]).all()
            and (ss_ == refs[nz]).all()):
        print("PACKGB INTEGRITY STOP")
        sys.exit(1)

    def old_path():
        p = G.pack_i32(k1, k2, m2)  # N-i32 D2H round-trip (old Q2 half)
        c, (s_,) = G.groupby_dense_impl(p, [(vv, "int32")])
        u2, c2, (s2_,) = G._gb_compact(c, [s_])
        return u2, c2, s2_

    mn_o, md_o = best(old_path)
    mn_n, md_n = best(lambda: G.pack_groupby_resident(k1, k2, vv, "sum"))
    st["pack_groupby"] = {"old_pack_roundtrip_med_ms": md_o,
                          "old_pack_roundtrip_min_ms": mn_o,
                          "resident_med_ms": md_n,
                          "resident_min_ms": mn_n,
                          "ratio_old_new": md_o / max(md_n, 1e-9)}

    # ---- 5. lookup intended use ----
    B = rng.integers(0, 1_000_000, 1_000_000, dtype=np.int32)
    P1 = rng.integers(0, 1_000_000, N, dtype=np.int32)
    P2 = rng.integers(0, 1_000_000, N, dtype=np.int32)
    P3 = rng.integers(0, 1_000_000, N, dtype=np.int32)
    V = rng.integers(-500, 500, N, dtype=np.int32)
    ref = G.lookup_ref(B, P1)
    T = G.lookup_build_resident(B)
    if not (T.probe_mask(P1) == ref).all():
        print("LOOKUP INTEGRITY STOP")
        sys.exit(1)
    if not (T.probe_filter(P1, V) == V[np.isin(P1, T.u)]).all():
        print("LOOKUP-CHAIN INTEGRITY STOP")
        sys.exit(1)
    _, cold_md = best(lambda: G.lookup_mask(B, P1))
    t = time.perf_counter()
    T2 = G.lookup_build_resident(B)
    build_ms = (time.perf_counter() - t) * 1000
    _, p1 = best(lambda: T2.probe_mask(P1))
    _, p2 = best(lambda: T2.probe_mask(P2))
    _, p3 = best(lambda: T2.probe_mask(P3))
    amort = (build_ms + p1 + p2 + p3) / 3
    st["lookup"] = {"cold_oneshot_med_ms": cold_md,
                    "build_upload_once_ms": build_ms,
                    "probe_p1_med_ms": p1, "probe_p2_med_ms": p2,
                    "probe_p3_med_ms": p3,
                    "amortized_per_probe_ms": amort,
                    "dict_K": int(T2.k)}
    del B, P2, P3
    gc.collect()

    # ---- 6. fused graphs ----
    K = rng.integers(0, 1000, N, dtype=np.int32)
    sel = A > 50
    cn = np.bincount(K[sel].astype(np.int64))
    nz6 = np.flatnonzero(cn)
    g = G.graph_compare_mask_filter_groupby(A, 50, V, K, "sum")
    cf = np.bincount(K[sel].astype(np.int64),
                     weights=V[sel].astype(np.float64)).astype(np.int64)
    if not ((g[0] == nz6).all() and (g[1] == cn[nz6]).all()
            and (g[2] == cf[nz6]).all()):
        print("GRAPH INTEGRITY STOP")
        sys.exit(1)
    _, g_md = best(lambda: G.graph_compare_mask_filter_groupby(
        A, 50, V, K, "sum"))
    st["graph_cmf_g"] = {"resident_med_ms": g_md}
    S2 = rng.integers(0, 1000, 1 << 20, dtype=np.int32)
    if not (G.graph_sort_slice_gather(S2, 100, 10)
            == np.sort(S2)[10:110]).all():
        print("GRAPH-SORT INTEGRITY STOP")
        sys.exit(1)
    _, gs_md = best(lambda: G.graph_sort_slice_gather(S2, 100, 10))
    st["graph_sort_slice_gather"] = {"resident_med_ms": gs_md}

    st["pipeline_final"] = G.pipeline_stats()
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    main()
