# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""B RANGE-PART: P mask scans -> per-partition small bincounts -> concat merge.

Partition cost honest: P full key scans + N key/val copies.
Aggregate parallelises over partitions with T threads (bincount releases GIL).
Merge trivial (disjoint ranges -> concat)."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .contract import State, empty_state, info
from .kernels import part_bounds, ws_bytes

NAME = "B_range_part"


def _agg_one(args):
    sub_k, sub_v, w = args
    s = np.bincount(sub_k, weights=sub_v, minlength=w).astype(np.int64)
    c = np.bincount(sub_k, minlength=w).astype(np.int64)
    return s, c


def run(keys, vals, g, p=16, threads=1):
    k = np.asarray(keys).astype(np.int32, copy=False)
    v = np.asarray(vals).astype(np.int32, copy=False)
    n = k.size
    if n == 0:
        return empty_state(), info(0.0, 0.0, 0.0, 0, 0, {"P": p})
    lo, hi = part_bounds(g, p)
    t0 = time.perf_counter()
    parts = []
    for q in range(p):
        m = (k >= lo[q]) & (k < hi[q])
        sk = (k[m].astype(np.int64, copy=False) - lo[q])
        sv = v[m].astype(np.int64, copy=False)
        parts.append((sk, sv, int(hi[q] - lo[q])))
    part_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    if threads > 1:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            res = list(ex.map(_agg_one, parts))
    else:
        res = [_agg_one(a) for a in parts]
    agg_ms = (time.perf_counter() - t1) * 1000.0
    t2 = time.perf_counter()
    uk, ss, cc = [], [], []
    for q in range(p):
        s_q, c_q = res[q]
        nz = np.flatnonzero(c_q > 0)
        if nz.size:
            uk.append(nz + lo[q])
            ss.append(s_q[nz])
            cc.append(c_q[nz])
    st = State(np.concatenate(uk).astype(np.int64) if uk else
               np.empty(0, dtype=np.int64),
               np.concatenate(ss) if ss else np.empty(0, dtype=np.int64),
               np.concatenate(cc) if cc else np.empty(0, dtype=np.int64)) \
        if uk else empty_state()
    merge_ms = (time.perf_counter() - t2) * 1000.0
    tmp_b = int(n * 8 + sum(int(s.nbytes + c.nbytes) for s, c in res))
    traffic = int(p * n * 4 + n * 8 + 8 * g * 2 + 8 * st.ngroups * 2)
    return st, info(part_ms, agg_ms, merge_ms, tmp_b, traffic,
                    {"P": p, "threads": threads, "ws_bytes": ws_bytes(g, p)})
