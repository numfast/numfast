# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""D SCATTER-PART hybrid: one counting scatter by pid=(k*P)//G, then small
bincounts over contiguous slices, concat merge. Partition = 2 passes
(pid + scatter) instead of B's P mask scans."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .contract import State, empty_state, info
from .kernels import HAS_NUMBA, part_bounds, scatter_by_pid, ws_bytes

NAME = "D_scatter_part"


def _agg_slice(args):
    sk, sv, w = args
    s = np.bincount(sk, weights=sv, minlength=w).astype(np.int64)
    c = np.bincount(sk, minlength=w).astype(np.int64)
    return s, c


def run(keys, vals, g, p=16, threads=1):
    if not HAS_NUMBA:
        raise RuntimeError("D needs numba")
    k = np.asarray(keys).astype(np.int32, copy=False)
    v = np.asarray(vals).astype(np.int32, copy=False)
    n = k.size
    if n == 0:
        return empty_state(), info(0.0, 0.0, 0.0, 0, 0, {"P": p})
    lo, hi = part_bounds(g, p)
    t0 = time.perf_counter()
    pid = ((k.astype(np.int64, copy=False) * np.int64(p)) // np.int64(g)) \
        .astype(np.int64)
    cnt = np.bincount(pid, minlength=p).astype(np.int64)
    starts = np.empty(p + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(cnt, out=starts[1:])
    cursor = starts[:-1].copy()
    out_k = np.empty(n, dtype=np.int32)
    out_v = np.empty(n, dtype=np.int32)
    scatter_by_pid(k, v, pid, cursor, out_k, out_v)
    part_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    jobs = [(out_k[starts[q]:starts[q + 1]].astype(np.int64, copy=False)
             - lo[q],
             out_v[starts[q]:starts[q + 1]].astype(np.int64, copy=False),
             int(hi[q] - lo[q])) for q in range(p)]
    if threads > 1:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            res = list(ex.map(_agg_slice, jobs))
    else:
        res = [_agg_slice(j) for j in jobs]
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
    tmp_b = int(pid.nbytes + out_k.nbytes + out_v.nbytes +
                sum(int(s.nbytes + c.nbytes) for s, c in res))
    traffic = int(k.nbytes * 2 + v.nbytes * 2 + n * 8 + 8 * g * 2 +
                  8 * st.ngroups * 2)
    return st, info(part_ms, agg_ms, merge_ms, tmp_b, traffic,
                    {"P": p, "threads": threads, "ws_bytes": ws_bytes(g, p)})
