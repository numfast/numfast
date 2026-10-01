# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""E CURRENT OPTIMIZED WITHOUT COPIES: same math as A but zero-copy views:
astype(copy=False), shifted view reuse, lut built once, inverse via fancy
lut[sh] (one alloc, unavoidable), bincount minlength=ng (no oversize).
Measures whether copies (not algorithm) dominate A."""

import time

import numpy as np

from .contract import State, empty_state

NAME = "E_nocopy"


def run(keys, vals, fused=False):
    k = np.asarray(keys)
    if k.dtype != np.int32:
        k = k.astype(np.int32, copy=False)
    v = np.asarray(vals)
    if v.dtype != np.int32:
        v = v.astype(np.int32, copy=False)
    n = k.size
    if n == 0:
        return empty_state(fused), {"gi_ms": 0.0, "agg_ms": 0.0, "total_ms": 0.0,
                                    "tmp_bytes": 0, "traffic_bytes": 0}
    t0 = time.perf_counter()
    kmin = int(k.min())
    kmax = int(k.max())
    r = kmax - kmin + 1
    sh = k.astype(np.int64, copy=False) - np.int64(kmin)  # view-ish, 1 temp
    cnt_r = np.bincount(sh, minlength=r)
    nz = np.flatnonzero(cnt_r)
    ng = nz.size
    lut = np.full(r, -1, dtype=np.int64)
    lut[nz] = np.arange(ng, dtype=np.int64)
    inverse = lut[sh]
    ukeys = nz.astype(np.int64, copy=False) + np.int64(kmin)
    gi_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    w = v.astype(np.int64, copy=False)
    sums = np.bincount(inverse, weights=w, minlength=ng).astype(np.int64)
    counts = np.bincount(inverse, minlength=ng).astype(np.int64)
    extra = None
    if fused:
        o = np.argsort(inverse, kind="stable")
        sv = v[o]
        bounds = np.flatnonzero(inverse[o][1:] != inverse[o][:-1]).astype(np.int64) + 1
        starts = np.empty(ng + 1, dtype=np.int64)
        starts[0] = 0
        starts[1:-1] = bounds
        starts[ng] = n
        mins = np.empty(ng, dtype=np.int32)
        maxs = np.empty(ng, dtype=np.int32)
        for g in range(ng):
            seg = sv[starts[g]:starts[g + 1]]
            mins[g] = seg.min()
            maxs[g] = seg.max()
        extra = (mins, maxs)
    agg_ms = (time.perf_counter() - t1) * 1000.0
    st = State(ukeys, sums, counts, extra[0], extra[1]) if fused else State(ukeys, sums, counts)
    tmp_b = int(sh.nbytes + cnt_r.nbytes + lut.nbytes + inverse.nbytes)
    traffic = int(k.nbytes + v.nbytes + 8 * r + 8 * n + 8 * ng * 2)
    return st, {"gi_ms": gi_ms, "agg_ms": agg_ms, "total_ms": gi_ms + agg_ms,
                "tmp_bytes": tmp_b, "traffic_bytes": traffic, "range": r}
