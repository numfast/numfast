# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""A FLAT dense baseline: gate probe + full-G bincount + compact."""

import time

import numpy as np

from .contract import State, empty_state, info

NAME = "A_flat"


def run(keys, vals, g, p=1, threads=1):
    k = np.asarray(keys).astype(np.int32, copy=False)
    v = np.asarray(vals).astype(np.int32, copy=False)
    n = k.size
    if n == 0:
        return empty_state(), info(0.0, 0.0, 0.0, 0, 0, {"range": g})
    t0 = time.perf_counter()
    kmin, kmax = int(k.min()), int(k.max())
    part_ms = (time.perf_counter() - t0) * 1000.0
    assert 0 <= kmin and kmax < g, "keys out of [0,G)"
    t1 = time.perf_counter()
    ki = k.astype(np.int64, copy=False)
    w = v.astype(np.int64, copy=False)
    sums_m = np.bincount(ki, weights=w, minlength=g).astype(np.int64)
    counts_m = np.bincount(ki, minlength=g).astype(np.int64)
    mask = counts_m > 0
    ukeys = np.flatnonzero(mask).astype(np.int64)
    st = State(ukeys, sums_m[mask], counts_m[mask])
    agg_ms = (time.perf_counter() - t1) * 1000.0
    tmp_b = int(sums_m.nbytes + counts_m.nbytes)
    traffic = int(k.nbytes + v.nbytes + 8 * g * 2 + 8 * st.ngroups * 2)
    return st, info(part_ms, agg_ms, 0.0, tmp_b, traffic,
                    {"range": g, "ws_bytes": int(8 * g * 2)})
