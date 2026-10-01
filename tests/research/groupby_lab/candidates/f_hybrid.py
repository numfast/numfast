# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""F HYBRID ADAPTIVE: per-block dispatch from cheap metadata (no full unique):
dense span -> bincount; sorted -> single run pass; est uniq>0.9 -> hash;
else radix-partition. Merge = golden funnel. Records chosen path.
"""

import time

import numpy as np

from ..contract import State, empty_state, merge_states
from ..kernels import HAS_NUMBA, is_sorted_asc, radix_partition_local, run_encode

NAME = "F_hybrid"
DENSE_SPAN_CAP = 1_000_000


def _estimate(keys):
    k = np.asarray(keys).astype(np.int64, copy=False)
    n = k.size
    if n == 0:
        return {"rows": 0, "span": 0, "sorted": True, "est_uniq": 1.0}
    mn = int(k.min())
    mx = int(k.max())
    span = mx - mn + 1
    srt = bool(is_sorted_asc(k)) if HAS_NUMBA else bool(np.all(np.diff(k) >= 0))
    step = max(1, n // 4096)
    samp = k[::step]
    u = np.unique(samp).size
    return {"rows": n, "span": int(span), "min": mn, "sorted": srt,
            "est_uniq": float(u / samp.size)}


def local(keys, vals):
    t0 = time.perf_counter()
    k = np.asarray(keys)
    n = k.size
    if n == 0:
        return empty_state(), {"local_ms": 0.0, "path": "empty"}
    e = _estimate(k)
    path = None
    if e["span"] <= min(4 * n, DENSE_SPAN_CAP):
        path = "dense"
        ki = k.astype(np.int64, copy=False) - e["min"]
        v = np.asarray(vals).astype(np.float64)
        su = np.bincount(ki, weights=v, minlength=e["span"]).astype(np.int64)
        co = np.bincount(ki, minlength=e["span"]).astype(np.int64)
        mask = co > 0
        uk = (np.nonzero(mask)[0].astype(np.int64) + e["min"])
        st = State(uk, su[mask], co[mask])
    elif e["sorted"]:
        path = "sorted"
        kk = k.astype(np.int64, copy=False)
        vv = np.asarray(vals).astype(np.float64)
        if HAS_NUMBA:
            ok, ss, cc, m = run_encode(kk, vv)
            st = State(ok[:m], ss[:m].astype(np.int64), cc[:m])
        else:
            u, inv = np.unique(kk, return_inverse=True)
            st = State(u.astype(np.int64),
                       np.bincount(inv, weights=vv).astype(np.int64),
                       np.bincount(inv, minlength=u.size).astype(np.int64))
    elif e["est_uniq"] > 0.9:
        path = "hash"
        u, inv = np.unique(k, return_inverse=True)
        v = np.asarray(vals).astype(np.float64)
        st = State(u.astype(np.int64),
                   np.bincount(inv, weights=v).astype(np.int64),
                   np.bincount(inv, minlength=u.size).astype(np.int64))
    else:
        path = "part"
        st, _ = radix_partition_local(k, vals, 2)
    ms = (time.perf_counter() - t0) * 1000.0
    return st, {"local_ms": ms, "path": path, "est_uniq": e["est_uniq"],
                "span": e["span"], "was_sorted": e["sorted"]}
def merge(states):
    return merge_states(states)
