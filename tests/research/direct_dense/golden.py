# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden verifier: exact vs plain np.unique+bincount reference. Fused min/max
via sorted-run slices (no dict). Raises on mismatch."""

import numpy as np


def golden_reference(keys, vals, fused=False):
    k = np.asarray(keys)
    v = np.asarray(vals)
    uk, inv = np.unique(k, return_inverse=True)
    sums = np.bincount(inv, weights=v.astype(np.float64)).astype(np.int64)
    counts = np.bincount(inv, minlength=uk.size).astype(np.int64)
    if not fused:
        return uk.astype(np.int64), sums, counts, None, None
    o = np.argsort(inv, kind="stable")
    sk, sv = k[o], v.astype(np.int32, copy=False)[o]
    bounds = np.flatnonzero(np.diff(inv[o]).astype(np.int8) != 0) + 1 \
        if sk.size else np.empty(0, dtype=np.int64)
    starts = np.empty(bounds.size + 1, dtype=np.int64)
    starts[0] = 0
    if bounds.size:
        starts[1:] = bounds
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = sk.size
    mins = np.empty(uk.size, dtype=np.int32)
    maxs = np.empty(uk.size, dtype=np.int32)
    for g in range(uk.size):
        seg = sv[starts[g]:ends[g]]
        mins[g] = int(seg.min())
        maxs[g] = int(seg.max())
    return uk.astype(np.int64), sums, counts, mins, maxs


def verify(state, keys, vals, fused=False, label=""):
    k = np.asarray(keys)
    v = np.asarray(vals)
    n = k.size
    assert state.unique, f"{label}: final state must be unique"
    assert int(state.counts.sum()) == int(n), \
        f"{label}: counts sum {state.counts.sum()} != rows {n}"
    if n:
        assert np.all(np.diff(state.ukeys) > 0), f"{label}: ukeys not sorted+unique"
        assert np.all(state.counts > 0), f"{label}: non-positive count"
    rk, rs, rc, rmin, rmax = golden_reference(k, v, fused=fused)
    assert rk.shape == state.ukeys.shape, \
        f"{label}: ngroups {state.ngroups} != golden {rk.size}"
    assert np.array_equal(state.ukeys, rk), f"{label}: ukeys != golden"
    assert np.array_equal(state.counts, rc), f"{label}: counts != golden"
    assert np.array_equal(state.sums, rs), f"{label}: sums not bit-exact"
    if fused:
        assert state.mins is not None and state.maxs is not None, \
            f"{label}: fused state missing mins/maxs"
        assert np.array_equal(state.mins, rmin), f"{label}: mins != golden"
        assert np.array_equal(state.maxs, rmax), f"{label}: maxs != golden"
    return {"ngroups": int(state.ngroups), "rows": int(n),
            "state_bytes": int(state.state_bytes())}
