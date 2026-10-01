# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""A FULL LOCAL UNIQUE: block -> complete local aggregation -> unique states.

Baseline: np.unique + bincount per block (exact, sorted), global funnel =
merge_states. Measures local_ms vs merge_ms split.
"""

import time

import numpy as np

from ..contract import State, merge_states

NAME = "A_full_unique"


def local(keys, vals):
    t0 = time.perf_counter()
    k = np.asarray(keys)
    v = np.asarray(vals).astype(np.float64)
    if k.size == 0:
        z = np.empty(0, dtype=np.int64)
        return State(z, z.copy(), z.copy()), {"local_ms": 0.0}
    uk, inv = np.unique(k, return_inverse=True)
    sums = np.bincount(inv, weights=v).astype(np.int64)
    counts = np.bincount(inv, minlength=uk.size).astype(np.int64)
    ms = (time.perf_counter() - t0) * 1000.0
    return State(uk.astype(np.int64), sums, counts), {"local_ms": ms}


def merge(states):
    return merge_states(states)
