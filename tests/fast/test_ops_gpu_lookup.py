# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""GPU lookup / semi-join golden (standalone, no IR wiring).

Semantics (fixed): build dupes collapse (existence, no multiplicity);
probe dupes independent; M=0 -> all zeros; N=0 -> empty.
Gates: exact vs lookup_ref/numpy isin; empty; 1-elem; all/no-match;
dupes both sides; non-pow2; big N; 1000 random seed 42; resident chain.
"""

import importlib.util as _ilu
import time
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])


def _mod():
    spec = _ilu.spec_from_file_location(
        "nfgpu_lookup", str(Path(APP_DIR) / "src" / "Drivers" / "GPU" / "_lib"
                             / "gpu.py"))
    G = _ilu.module_from_spec(spec)
    spec.loader.exec_module(G)
    return G


@pytest.fixture(scope="module")
def G():
    return _mod()


def _stages(fn, label):
    s = time.perf_counter()
    out = fn()
    dt = (time.perf_counter() - s) * 1000
    print(f"\ngpu lookup stages ms: {label}={dt:.3f}")
    return out


def test_lookup_dup_semantics_fixed(G):
    assert list(G.lookup_mask([5, 5, 7], [5])) == [1]  # build dupes: x1
    assert list(G.lookup_mask([5, 7], [5, 5])) == [1, 1]  # probe dupes: x2
    assert list(G.lookup_ref([5, 5, 7], [5, 6])) == [1, 0]


def test_lookup_edges(G):
    z = np.zeros(0, dtype=np.int32)
    assert G.lookup_mask([], [1, 2, 3]).tolist() == [0, 0, 0]  # M=0
    assert G.lookup_mask([1, 2], z).size == 0  # N=0
    assert G.lookup_mask([], z).size == 0
    assert G.lookup_mask([7], [7]).tolist() == [1]  # 1-elem hit
    assert G.lookup_mask([7], [8]).tolist() == [0]  # 1-elem miss
    n = np.arange(100, dtype=np.int32)
    assert (G.lookup_mask(n, n) == 1).all()  # all-match
    assert (G.lookup_mask(n, n + 1000) == 0).all()  # no-match
    lo = np.array([-2147483648, 2147483647], dtype=np.int32)
    assert G.lookup_mask(lo, lo).tolist() == [1, 1]  # int32 extremes


def test_lookup_1000_random_exact(G):
    rng = np.random.default_rng(42)
    bad = 0
    for t in range(1000):
        m = int(rng.integers(0, 300))
        n = int(rng.integers(0, 300))
        if t % 100 == 0:  # non-pow2 + dupes forced
            B = rng.integers(0, 7, max(m, 1)).astype(np.int32)
            P = rng.integers(0, 7, 100003 % 1000 + 3).astype(np.int32)
        else:
            B = rng.integers(-50, 50, m).astype(np.int32)
            P = rng.integers(-50, 50, n).astype(np.int32)
        got = _stages(lambda: G.lookup_mask(B, P), "probe") if False else G.lookup_mask(B, P)
        ref = G.lookup_ref(B, P)
        if got.shape != ref.shape or not (got == ref).all():
            bad += 1
    assert bad == 0, f"{bad}/1000 mismatch"


def test_lookup_big_n_nonpow2(G):
    rng = np.random.default_rng(42)
    B = rng.integers(0, 50000, 50000).astype(np.int32)
    P = rng.integers(0, 50000, 100003).astype(np.int32)  # non-pow2
    got = _stages(lambda: G.lookup_mask(B, P), "probe100k")
    assert (got == G.lookup_ref(B, P)).all()


def test_lookup_positions_feed_gather(G):
    rng = np.random.default_rng(42)
    B = rng.integers(0, 20, 30).astype(np.int32)
    P = rng.integers(0, 20, 500).astype(np.int32)
    V = rng.integers(0, 1000, 500).astype(np.int32)
    mask = G.lookup_mask(B, P)
    ix = G.filter_indices(mask)
    assert (G.gather(V, ix, "int32") == V[np.flatnonzero(mask.astype(bool))]).all()


def test_lookup_chain_resident_no_n_roundtrip(G):
    rng = np.random.default_rng(42)
    B = rng.integers(0, 1000, 2000).astype(np.int32)
    P = rng.integers(0, 1000, 20000).astype(np.int32)
    V = rng.integers(-500, 500, 20000).astype(np.int32)
    got = _stages(lambda: G.lookup_filter_resident(B, P, V), "chain20k")
    ref = V[np.isin(P, np.unique(B))]
    assert got.shape == ref.shape and (got == ref).all()
    assert G.lookup_filter_resident(B, P[:0], V[:0]).size == 0
    assert G.lookup_filter_resident([], P, V).size == 0
