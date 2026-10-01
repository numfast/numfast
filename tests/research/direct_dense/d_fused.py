# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""D FUSED MULTI-AGGREGATE DIRECT: one njit pass -> sums+counts+mins+maxs
(Q1+Q4 fused state). SoA variant (separate arrays) + AoS variant (interleaved
row per key, legacy AoS-stream idea clean re-implemented). Caller picks
layout='soa'|'aos'."""

import time

import numpy as np

from .contract import State, empty_state
from .kernels import HAS_NUMBA, scatter_fused_aos, scatter_fused_soa

NAME = "D_fused_direct"


def run(keys, vals, m=None, layout="soa"):
    k = np.asarray(keys).astype(np.int32, copy=False)
    v = np.asarray(vals).astype(np.int32, copy=False)
    n = k.size
    if n == 0:
        return empty_state(True), {"gi_ms": 0.0, "agg_ms": 0.0, "total_ms": 0.0,
                                   "tmp_bytes": 0, "traffic_bytes": 0,
                                   "layout": layout}
    if not HAS_NUMBA:
        raise RuntimeError("D needs numba")
    t0 = time.perf_counter()
    kmin = int(k.min())
    base = 0 if (m is not None and kmin >= 0) else kmin
    M = int(m) if m is not None else (int(k.max()) - kmin + 1)
    kk = (k.astype(np.int64, copy=False) - np.int64(base)).astype(np.int64)
    gate_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    if layout == "soa":
        sums_m = np.zeros(M, dtype=np.int64)
        counts_m = np.zeros(M, dtype=np.int64)
        mins_m = np.full(M, np.iinfo(np.int32).max, dtype=np.int32)
        maxs_m = np.full(M, np.iinfo(np.int32).min, dtype=np.int32)
        scatter_fused_soa(kk, v, sums_m, counts_m, mins_m, maxs_m)
        mask = counts_m > 0
        ukeys = np.flatnonzero(mask).astype(np.int64) + np.int64(base)
        st = State(ukeys, sums_m[mask], counts_m[mask],
                   mins_m[mask], maxs_m[mask])
        tmp_b = int(sums_m.nbytes + counts_m.nbytes + mins_m.nbytes + maxs_m.nbytes)
    else:
        W = 4
        aos = np.zeros(M * W, dtype=np.int64)
        aos[2::W] = np.int64(np.iinfo(np.int32).max)
        aos[3::W] = np.int64(np.iinfo(np.int32).min)
        scatter_fused_aos(kk, v.astype(np.int64, copy=False), aos, np.int64(W))
        counts_m = aos[1::W]
        mask = counts_m > 0
        idx = np.flatnonzero(mask)
        ukeys = idx.astype(np.int64) + np.int64(base)
        sums = aos[idx * W].copy()
        counts = counts_m[mask].copy()
        mins = aos[idx * W + 2].astype(np.int32)
        maxs = aos[idx * W + 3].astype(np.int32)
        st = State(ukeys, sums, counts, mins, maxs)
        tmp_b = int(aos.nbytes)
    agg_ms = (time.perf_counter() - t1) * 1000.0
    traffic = int(k.nbytes + v.nbytes + tmp_b + 8 * st.ngroups * 2 + 4 * st.ngroups * 2)
    return st, {"gi_ms": gate_ms, "agg_ms": agg_ms, "total_ms": gate_ms + agg_ms,
                "tmp_bytes": tmp_b, "traffic_bytes": traffic,
                "range": M, "layout": layout}
