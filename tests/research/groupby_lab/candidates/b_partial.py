# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""B PARTIAL COALESCE: block split into SEG segments, unique each segment,
concat -> reduced but NOT unique stream (unique=False). Global funnel =
merge_states. Metric: next_level_cost (partial stream rows the merger must
re-hash) + reduction vs direct full unique.
"""

import time

import numpy as np

from ..contract import State, empty_state, merge_states

NAME = "B_partial_coalesce"


def local(keys, vals, seg=None):
    t0 = time.perf_counter()
    k = np.asarray(keys)
    v = np.asarray(vals).astype(np.float64)
    n = k.size
    if n == 0:
        return empty_state(), {"local_ms": 0.0, "partial_rows": 0}
    if seg is None:
        seg = 2 if n <= 32_768 else 4
    idx = np.array_split(np.arange(n), seg)
    uks, sus, cos = [], [], []
    for ix in idx:
        u, inv = np.unique(k[ix], return_inverse=True)
        uks.append(u.astype(np.int64))
        sus.append(np.bincount(inv, weights=v[ix]).astype(np.int64))
        cos.append(np.bincount(inv, minlength=u.size).astype(np.int64))
    uk = np.concatenate(uks)
    su = np.concatenate(sus)
    co = np.concatenate(cos)
    ms = (time.perf_counter() - t0) * 1000.0
    return State(uk, su, co, unique=False), {
        "local_ms": ms, "seg": seg, "partial_rows": int(uk.size),
    }


def describe(keys, partial_rows):
    """Reduction metric vs direct full unique. Call OUTSIDE timers (does its
    own np.unique pass); reports next_level_cost (rows the merger re-hashes)
    vs direct_cost. 65K->~40K->10K regime probes use seg=2/4/8."""
    n = np.asarray(keys).size
    full_g = int(np.unique(np.asarray(keys)).size)
    partial = int(partial_rows)
    return {
        "full_groups": full_g,
        "reduction_vs_raw": 1.0 - partial / max(n, 1),
        "residual_vs_direct": (partial - full_g) / max(full_g, 1),
        "next_level_cost": partial, "direct_cost": full_g,
    }


def merge(states):
    return merge_states(states)
