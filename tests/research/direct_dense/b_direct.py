# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""B DIRECT DENSE-BY-CODE: resident int32 codes already 0..M-1 -> NO shift,
NO inverse, NO lut. Gate: min>=0 and max<M (cheap vectorized probe, timed
separately as gate_ms). Aggregation = bincount(keys, minlength=M) directly,
compact via flatnonzero mask. Falls back flag when gate fails (no silent
wrong result)."""

import time

import numpy as np

from .contract import State, empty_state

NAME = "B_direct_by_code"


def run(keys, vals, m=None, fused=False):
    k = np.asarray(keys).astype(np.int32, copy=False)
    v = np.asarray(vals).astype(np.int32, copy=False)
    n = k.size
    if n == 0:
        return empty_state(fused), {"gi_ms": 0.0, "agg_ms": 0.0, "total_ms": 0.0,
                                    "tmp_bytes": 0, "traffic_bytes": 0,
                                    "gate_ok": True}
    t0 = time.perf_counter()
    kmin, kmax = (int(k.min()), int(k.max())) if m is None or True else (0, m - 1)
    gate_ms = (time.perf_counter() - t0) * 1000.0
    M = int(m) if m is not None else kmax + 1
    gate_ok = (kmin >= 0) and (kmax < M)
    if not gate_ok:
        return empty_state(fused), {"gi_ms": gate_ms, "agg_ms": 0.0,
                                    "total_ms": gate_ms, "tmp_bytes": 0,
                                    "traffic_bytes": int(k.nbytes + v.nbytes),
                                    "gate_ok": False, "range": M}
    t1 = time.perf_counter()
    w = v.astype(np.int64, copy=False)
    ki = k.astype(np.int64, copy=False)
    sums_m = np.bincount(ki, weights=w, minlength=M).astype(np.int64)
    counts_m = np.bincount(ki, minlength=M).astype(np.int64)
    mask = counts_m > 0
    ukeys = np.flatnonzero(mask).astype(np.int64)
    sums, counts = sums_m[mask], counts_m[mask]
    if fused:
        # min/max still need one indirect pass over M-sized aux; use
        # sort-free segment pass on compacted order via argsort of keys
        o = np.argsort(ki, kind="stable")
        sk, sv = ki[o], v[o]
        bounds = np.flatnonzero(sk[1:] != sk[:-1]).astype(np.int64) + 1
        ng = ukeys.size
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
    tmp_b = int(sums_m.nbytes + counts_m.nbytes)
    traffic = int(k.nbytes + v.nbytes + 8 * M * 2 + 8 * st.ngroups * 2)
    return st, {"gi_ms": gate_ms, "agg_ms": agg_ms, "total_ms": gate_ms + agg_ms,
                "tmp_bytes": tmp_b, "traffic_bytes": traffic,
                "gate_ok": True, "range": M}
