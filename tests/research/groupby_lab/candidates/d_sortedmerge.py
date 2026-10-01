# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""D SORTED PARTIAL STATE + LINEAR MERGE: local emits sorted unique via
single run pass (argsort first if unsorted); merge is pairwise linear
two-pointer tree counting TOTAL work (comparisons) over ALL levels.
Bit-identical to golden (checked in bench).
"""

import time

import numpy as np

from ..contract import State, empty_state
from ..kernels import HAS_NUMBA, is_sorted_asc, merge2, run_encode

NAME = "D_sorted_linear"


def local(keys, vals):
    t0 = time.perf_counter()
    k = np.asarray(keys).astype(np.int64, copy=False)
    v = np.asarray(vals).astype(np.float64, copy=False)
    n = k.size
    if n == 0:
        return empty_state(), {"local_ms": 0.0, "was_sorted": True}
    if HAS_NUMBA:
        srt = bool(is_sorted_asc(k))
    else:
        srt = bool(np.all(np.diff(k) >= 0))
    if srt:
        sk, sv = k, v
    else:
        o = np.argsort(k, kind="stable")
        sk, sv = k[o], v[o]
    if HAS_NUMBA:
        ok, ss, cc, m = run_encode(sk, sv)
        uk, su, co = ok[:m], ss[:m].astype(np.int64), cc[:m]
    else:
        u, inv = np.unique(sk, return_inverse=True)
        uk = u.astype(np.int64)
        su = np.bincount(inv, weights=sv).astype(np.int64)
        co = np.bincount(inv, minlength=u.size).astype(np.int64)
    ms = (time.perf_counter() - t0) * 1000.0
    return State(uk, su, co), {"local_ms": ms, "was_sorted": srt}


def _merge_pair(a, b):
    na, nb = a.ngroups, b.ngroups
    ok = np.empty(na + nb, dtype=np.int64)
    os = np.empty(na + nb, dtype=np.int64)
    oc = np.empty(na + nb, dtype=np.int64)
    m, comps = merge2(a.ukeys, a.sums, a.counts, b.ukeys, b.sums, b.counts,
                      ok, os, oc)
    return State(ok[:m], os[:m], oc[:m]), int(comps)


def merge(states):
    import time as _t
    t0 = _t.perf_counter()
    cur = [s for s in states if s.ngroups > 0]
    comps = 0
    rounds = 0
    if not cur:
        return empty_state(), {"merge_ms": 0.0, "comparisons": 0, "rounds": 0}
    if not HAS_NUMBA:  # honest fallback: golden funnel, same result
        from ..contract import merge_states
        st, info = merge_states(cur)
        info = dict(info, rounds=0, linear=False)
        return st, info
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            if i + 1 < len(cur):
                m, c = _merge_pair(cur[i], cur[i + 1])
                comps += c
                nxt.append(m)
            else:
                nxt.append(cur[i])
        cur = nxt
        rounds += 1
    ms = (_t.perf_counter() - t0) * 1000.0
    return cur[0], {"merge_ms": ms, "comparisons": comps, "rounds": rounds,
                    "linear": True}
