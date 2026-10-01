# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden verifier: one checker for ALL candidates.

Checks (int exact; float via specs/conformance-profile.toml f64):
  - ukeys unique + ascending, counts>0
  - counts.sum() == N rows
  - per-key sum exact vs reference, per-key count exact
  - mean within tolerance (derived sum/count)
No dataset hacks: reference is plain np.unique+bincount on the same input.
"""

import tomllib
from pathlib import Path

import numpy as np

PROFILE_PATH = (
    Path(__file__).resolve().parents[3] / "specs-rebuilt" / "conformance-profile.toml"
)


def load_profile():
    with open(PROFILE_PATH, "rb") as f:
        return tomllib.load(f)


def golden_reference(keys, vals):
    keys = np.asarray(keys)
    vals = np.asarray(vals)
    uk, inv = np.unique(keys, return_inverse=True)
    sums = np.bincount(inv, weights=vals.astype(np.float64)).astype(np.int64)
    counts = np.bincount(inv, minlength=uk.size).astype(np.int64)
    return uk.astype(np.int64), sums, counts


def verify(state, keys, vals, profile=None, label=""):
    """Raise AssertionError on any mismatch; return check dict otherwise."""
    from .contract import State  # noqa: F401  (type check only)
    profile = profile or load_profile()
    tol = profile["tolerance"]["f64"]
    keys = np.asarray(keys)
    vals = np.asarray(vals)
    n = keys.size
    assert state.unique, f"{label}: final state must be unique"
    assert state.ngroups > 0 or n == 0, f"{label}: empty state for {n} rows"
    d = np.diff(state.ukeys)
    assert np.all(d > 0), f"{label}: ukeys not strictly ascending+unique"
    assert np.all(state.counts > 0), f"{label}: non-positive count present"
    assert int(state.counts.sum()) == int(n), (
        f"{label}: counts sum {state.counts.sum()} != rows {n}")
    ref_k, ref_s, ref_c = golden_reference(keys, vals)
    assert ref_k.shape == state.ukeys.shape, (
        f"{label}: ngroups {state.ngroups} != golden {ref_k.size}")
    assert np.array_equal(state.ukeys, ref_k), f"{label}: ukeys != golden"
    assert np.array_equal(state.counts, ref_c), f"{label}: counts != golden"
    assert np.array_equal(state.sums, ref_s), f"{label}: sums not bit-exact"
    mean = state.means()
    ref_mean = ref_s.astype(np.float64) / np.maximum(ref_c, 1).astype(np.float64)
    diff = np.abs(mean - ref_mean)
    allowed = np.maximum(tol["atol"], tol["rtol"] * np.abs(ref_mean))
    assert np.all(diff <= allowed), f"{label}: mean exceeds f64 tolerance"
    return {
        "ngroups": int(state.ngroups), "rows": int(n),
        "max_mean_diff": float(diff.max(initial=0.0)),
        "state_bytes": int(state.state_bytes()),
    }
