# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden verifier: exact vs flat bincount reference (bit-identical int64).
Plus pid-mapping verifier: partition mapping must be bijective
(key_of(pid(k)) == k for every row) with pid in [0,P), li in [0,W)."""

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


def verify_pid(pid, keys, variant, par, g, p, label=""):
    """Exactness of the pid mapping itself: bounds + bijective round-trip."""
    k = np.asarray(keys).astype(np.int64, copy=False)
    q = np.ascontiguousarray(pid, dtype=np.int64)
    assert q.shape == k.shape, f"{label}: pid shape mismatch"
    assert int(q.min()) >= 0 and int(q.max()) < p, \
        f"{label}: pid out of [0,P)"
    if variant == "shift":
        w = par["width"]
        li = k - q * np.int64(w)
    elif variant == "scaled":
        li = k - par["lo"][q]
        w = par["width"]
    elif variant == "div":
        w = par["ws"]
        li = k - q * np.int64(w)
    elif variant == "mask":
        w = par["width"]
        li = k >> np.int64(par["lbits"])
        assert np.array_equal(q, k & np.int64(p - 1)), \
            f"{label}: mask pid != k&(P-1)"
        assert np.array_equal(li * np.int64(p) + q, k), \
            f"{label}: mask round-trip failed"
        assert int(li.min()) >= 0 and int(li.max()) < w, \
            f"{label}: mask li out of bounds"
        return {"pids_used": int(np.unique(q).size)}
    else:
        raise ValueError(variant)
    assert int(li.min()) >= 0 and int(li.max()) < w, \
        f"{label}: li out of [0,W)"
    assert np.array_equal(key_of_vec(variant, par, q, li, g, p), k), \
        f"{label}: round-trip key_of(pid(k)) != k"
    return {"pids_used": int(np.unique(q).size)}


def key_of_vec(variant, par, q, li, g, p):
    q = np.asarray(q, dtype=np.int64)
    li = np.asarray(li, dtype=np.int64)
    if variant == "shift":
        return q * np.int64(par["width"]) + li
    if variant == "scaled":
        return par["lo"][q] + li
    if variant == "div":
        return q * np.int64(par["ws"]) + li
    if variant == "mask":
        return li * np.int64(p) + q
    raise ValueError(variant)
