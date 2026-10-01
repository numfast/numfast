# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""One input/output contract for fused_part lab. Local copy (no cross-lab imports)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    ukeys: np.ndarray
    sums: np.ndarray
    counts: np.ndarray
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
        return int(self.ukeys.nbytes + self.sums.nbytes + self.counts.nbytes)


def empty_state():
    z = np.empty(0, dtype=np.int64)
    return State(z, z.copy(), z.copy())


def info(part_ms, agg_ms, merge_ms, tmp_bytes, traffic_bytes, extra=None):
    d = {"part_ms": float(part_ms), "agg_ms": float(agg_ms),
         "merge_ms": float(merge_ms),
         "total_ms": float(part_ms + agg_ms + merge_ms),
         "tmp_bytes": int(tmp_bytes), "traffic_bytes": int(traffic_bytes)}
    if extra:
        d.update(extra)
    return d
