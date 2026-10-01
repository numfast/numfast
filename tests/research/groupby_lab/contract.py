# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Shared contract for all lab candidates.

State: sorted unique keys + per-key (sum int64, count int64). Mean is always
derived sum/count (same rule as production CPU driver), never a second pass.
B-partial local states may carry unique=False (reduced but not unique stream);
the FINAL merged state must always be unique=True.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    ukeys: np.ndarray   # int64, ascending iff unique=True
    sums: np.ndarray    # int64 accumulator
    counts: np.ndarray  # int64 accumulator
    unique: bool = True

    def __post_init__(self):
        self.ukeys = np.ascontiguousarray(self.ukeys, dtype=np.int64)
        self.sums = np.ascontiguousarray(self.sums, dtype=np.int64)
        self.counts = np.ascontiguousarray(self.counts, dtype=np.int64)
        assert self.ukeys.shape == self.sums.shape == self.counts.shape

    @property
    def ngroups(self):
        return int(self.ukeys.size)

    def state_bytes(self):
        # ukeys int64 + sums int64 + counts int64 (accumulator discipline)
        return int(self.ukeys.nbytes + self.sums.nbytes + self.counts.nbytes)

    def means(self):
        return self.sums.astype(np.float64) / np.maximum(self.counts, 1).astype(np.float64)


def empty_state():
    z = np.empty(0, dtype=np.int64)
    return State(z, z.copy(), z.copy(), unique=True)


def merge_states(states):
    """Golden combiner: concat partials (unique or not) -> one unique State.

    Single funnel used by every candidate merge path unless the candidate
    overrides it with its own measured merge (D linear merge, E tree merge).
    Override results must be bit-identical to this (checked in tests).
    """
    states = [s for s in states if s.ngroups > 0]
    if not states:
        return empty_state(), {"merge_ms": 0.0, "comparisons": 0}
    import time
    t0 = time.perf_counter()
    ku = np.concatenate([s.ukeys for s in states])
    su = np.concatenate([s.sums for s in states])
    cu = np.concatenate([s.counts for s in states])
    uk, inv = np.unique(ku, return_inverse=True)
    sums = np.bincount(inv, weights=su.astype(np.float64)).astype(np.int64)
    counts = np.bincount(inv, weights=cu.astype(np.float64)).astype(np.int64)
    ms = (time.perf_counter() - t0) * 1000.0
    return State(uk.astype(np.int64), sums, counts, unique=True), {
        "merge_ms": ms, "comparisons": int(ku.size),
    }
