# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden verifier: exact vs flat bincount reference (bit-identical int64)."""

import numpy as np


def reference(keys, vals, g):
    k = np.asarray(keys).astype(np.int64, copy=False)
    w = np.asarray(vals).astype(np.int64, copy=False)
    sums_m = np.bincount(k, weights=w, minlength=g).astype(np.int64)
    counts_m = np.bincount(k, minlength=g).astype(np.int64)
    mask = counts_m > 0
    return (np.flatnonzero(mask).astype(np.int64), sums_m[mask],
            counts_m[mask].astype(np.int64))


def verify(state, keys, vals, g, label=""):
    rk, rs, rc = reference(keys, vals, g)
    n = np.asarray(keys).size
    assert state.unique, f"{label}: final state must be unique"
    assert int(state.counts.sum()) == int(n), \
        f"{label}: counts sum {state.counts.sum()} != rows {n}"
    assert rk.shape == state.ukeys.shape, \
        f"{label}: ngroups {state.ngroups} != golden {rk.size}"
    assert np.array_equal(state.ukeys, rk), f"{label}: ukeys != golden"
    assert np.array_equal(state.counts, rc), f"{label}: counts != golden"
    assert np.array_equal(state.sums, rs), f"{label}: sums not bit-exact"
    return {"ngroups": int(state.ngroups), "rows": int(n),
            "state_bytes": int(state.state_bytes())}
