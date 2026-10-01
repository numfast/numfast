# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""A CURRENT DENSE (shift+bincount+inverse): range probe -> shift -> bincount
sums+counts (R) -> ukeys via flatnonzero -> inverse via lut[shifted] ->
aggregate via bincount on inverse (mirrors production dense_direct_index +
fused sums/counts split). Stages timed separately: gi_ms / agg_ms."""

import time

import numpy as np

from .contract import State, empty_state

NAME = "A_shift_bincount_inverse"


def run(keys, vals, fused=False):
    k = np.asarray(keys).astype(np.int32, copy=False)
    v = np.asarray(vals).astype(np.int32, copy=False)
    n = k.size
    if n == 0:
        return empty_state(fused), {"gi_ms": 0.0, "agg_ms": 0.0, "total_ms": 0.0,
                                    "tmp_bytes": 0, "traffic_bytes": 0}
    t0 = time.perf_counter()
    kmin, kmax = int(k.min()), int(k.max())
    r = kmax - kmin + 1
    sh = (k.astype(np.int64, copy=False) - np.int64(kmin)).astype(np.int64)
    cnt_r = np.bincount(sh, minlength=r)
    nz = np.flatnonzero(cnt_r)
    lut = np.full(r, -1, dtype=np.int64)
    lut[nz] = np.arange(nz.size, dtype=np.int64)
    inverse = lut[sh]
    ukeys = nz.astype(np.int64) + np.int64(kmin)
    gi_ms = (time.perf_counter() - t0) * 1000.0
    tmp_gi = int(sh.nbytes + cnt_r.nbytes + lut.nbytes * 2 + inverse.nbytes)
    t1 = time.perf_counter()
    ng = nz.size
    w = v.astype(np.int64, copy=False)
    sums = np.bincount(inverse, weights=w, minlength=ng).astype(np.int64)
    counts = np.bincount(inverse, minlength=ng).astype(np.int64)
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
        st = State(ukeys, sums, counts, mins, maxs)
    else:
        st = State(ukeys, sums, counts)
    agg_ms = (time.perf_counter() - t1) * 1000.0
    traffic = int(k.nbytes + v.nbytes + 8 * r * 2 + 8 * n + 8 * ng * 2)
    return st, {"gi_ms": gi_ms, "agg_ms": agg_ms, "total_ms": gi_ms + agg_ms,
                "tmp_bytes": tmp_gi, "traffic_bytes": traffic,
                "range": r, "kmin": kmin}
