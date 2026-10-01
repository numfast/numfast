# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""C NUMBA DIRECT DENSE: single njit pass state[key]+=value over preallocated
M arrays (legacy scatter idea, clean re-implementation). No shift when
resident codes 0..M-1; shifted variant when generic range. Compact at end."""

import time

import numpy as np

from .contract import State, empty_state
from .kernels import HAS_NUMBA, scatter_sums_counts

NAME = "C_numba_scatter"


def run(keys, vals, m=None, fused=False):
    k = np.asarray(keys).astype(np.int32, copy=False)
    v = np.asarray(vals).astype(np.int32, copy=False)
    n = k.size
    if n == 0:
        return empty_state(False), {"gi_ms": 0.0, "agg_ms": 0.0, "total_ms": 0.0,
                                    "tmp_bytes": 0, "traffic_bytes": 0}
    if not HAS_NUMBA:
        raise RuntimeError("C needs numba")
    t0 = time.perf_counter()
    kmin = int(k.min())
    base = 0 if (m is not None and kmin >= 0) else kmin
    M = int(m) if m is not None else (int(k.max()) - kmin + 1)
    kk = (k.astype(np.int64, copy=False) - np.int64(base)).astype(np.int64)
    gate_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    sums_m = np.zeros(M, dtype=np.int64)
    counts_m = np.zeros(M, dtype=np.int64)
    scatter_sums_counts(kk, v.astype(np.int64, copy=False), sums_m, counts_m)
    mask = counts_m > 0
    ukeys = np.flatnonzero(mask).astype(np.int64) + np.int64(base)
    st = State(ukeys, sums_m[mask], counts_m[mask])
    agg_ms = (time.perf_counter() - t1) * 1000.0
    traffic = int(k.nbytes + v.nbytes + 8 * M * 2 + 8 * st.ngroups * 2)
    return st, {"gi_ms": gate_ms, "agg_ms": agg_ms, "total_ms": gate_ms + agg_ms,
                "tmp_bytes": int(sums_m.nbytes + counts_m.nbytes),
                "traffic_bytes": traffic, "range": M}
