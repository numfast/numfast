# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""One input/output contract for direct-dense A-E. No dict/list hot paths."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class State:
    ukeys: np.ndarray
    sums: np.ndarray
    counts: np.ndarray
    mins: np.ndarray = field(default=None)
    maxs: np.ndarray = field(default=None)
    unique: bool = True

    def __post_init__(self):
        self.ukeys = np.ascontiguousarray(self.ukeys, dtype=np.int64)
        self.sums = np.ascontiguousarray(self.sums, dtype=np.int64)
        self.counts = np.ascontiguousarray(self.counts, dtype=np.int64)
        if self.mins is not None:
            self.mins = np.ascontiguousarray(self.mins, dtype=np.int32)
        if self.maxs is not None:
            self.maxs = np.ascontiguousarray(self.maxs, dtype=np.int32)
        assert self.ukeys.shape == self.sums.shape == self.counts.shape

    @property
    def ngroups(self):
        return int(self.ukeys.size)

    def state_bytes(self):
        n = int(self.ukeys.nbytes + self.sums.nbytes + self.counts.nbytes)
        if self.mins is not None:
            n += int(self.mins.nbytes + self.maxs.nbytes)
        return n


def empty_state(fused=False):
    z64 = np.empty(0, dtype=np.int64)
    if fused:
        z32 = np.empty(0, dtype=np.int32)
        return State(z64, z64.copy(), z64.copy(), z32, z32.copy())
    return State(z64, z64.copy(), z64.copy())
