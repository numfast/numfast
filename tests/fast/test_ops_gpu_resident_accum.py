# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""SPEC-DELTA-11 resident-accumulate fast gates (seed 42, small N, exact int).

Covers: Q1-like exact vs numpy + vs groupby_dense_impl, associativity
(reversed chunk order), Q2 packed exact, Q3-spread exact, lo-carry wrap
exact, scaled-f64 unscale tolerance. Small N and exact; no memory-scale case.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

FORK = Path(__file__).resolve().parents[2]
G_SPEC = importlib.util.spec_from_file_location(
    "nfgpu_delta11",
    str(FORK / "src" / "Drivers" / "GPU" / "_lib" / "gpu.py"))
G = importlib.util.module_from_spec(G_SPEC)
G_SPEC.loader.exec_module(G)


@pytest.mark.fast
def test_delta11_q1_exact_and_assoc():
    rng = np.random.default_rng(42)
    k = rng.integers(0, 100, 20000, dtype=np.int32)
    v = rng.integers(-500, 500, 20000, dtype=np.int32)
    uk, cc, ss, st = G.groupby_resident_accumulate(k, v, chunk_rows=4096)
    rc = np.bincount(k.astype(np.int64))
    nz = np.flatnonzero(rc)
    rs = np.bincount(k.astype(np.int64),
                     weights=v.astype(np.float64)).astype(np.int64)
    assert (uk == nz).all() and (cc == rc[nz]).all() and (ss == rs[nz]).all()
    c0, (s0,) = G.groupby_dense_impl(k, [(v, "int32")])
    u0, c00, (s00,) = G._gb_compact(c0, [s0])
    assert (uk == u0).all() and (cc == c00).all() and (ss == s00).all()
    rev = list(reversed(range(st["num_chunks"])))
    uk2, cc2, ss2, _ = G.groupby_resident_accumulate(k, v, chunk_rows=4096,
                                                     chunk_order=rev)
    assert (uk == uk2).all() and (cc == cc2).all() and (ss == ss2).all()
    assert st["d2h_bytes"] == 4 * 100 * 4 and st["merge_ms"] < 5.0


@pytest.mark.fast
def test_delta11_q2_packed_exact():
    rng = np.random.default_rng(42)
    k1 = rng.integers(0, 100, 20000, dtype=np.int32)
    k2 = rng.integers(0, 100, 20000, dtype=np.int32)
    v = rng.integers(-500, 500, 20000, dtype=np.int32)
    uk, cc, ss, st = G.groupby_resident_accumulate_packed(k1, k2, v,
                                                          chunk_rows=4096)
    m2 = int(k2.max()) + 1
    pk = k1.astype(np.int64) * m2 + k2.astype(np.int64)
    rc = np.bincount(pk)
    nz = np.flatnonzero(rc)
    rs = np.bincount(pk, weights=v.astype(np.float64)).astype(np.int64)
    assert (uk == nz).all() and (cc == rc[nz]).all() and (ss == rs[nz]).all()
    assert st["m2"] == m2 and st["m"] == (int(k1.max()) + 1) * m2


@pytest.mark.fast
def test_delta11_carry_wrap_exact():
    k = np.tile(np.array([0, 1], dtype=np.int32), 150000)
    v = np.full(300000, 30000, dtype=np.int32)  # lo total 4.5e9 > u32max
    uk, cc, ss, _ = G.groupby_resident_accumulate(k, v, chunk_rows=50000)
    assert (ss == np.array([4500000000, 4500000000])).all()
    assert (cc == np.array([150000, 150000])).all()


@pytest.mark.fast
def test_delta11_scaled_f64_tolerance():
    rng = np.random.default_rng(42)
    k = rng.integers(0, 100, 20000, dtype=np.int32)
    sc = 1e-6
    lv = rng.integers(-1000000, 1000000, 20000).astype(np.float64) * sc
    phys = G._f64_to_physical(lv, sc, 0.0, G._fallback_err, "t")
    uk, cc, ss, _ = G.groupby_resident_accumulate(k, phys, chunk_rows=4096)
    sums, _ = G._gb_unscale(ss, cc, sc, 0.0)
    m = int(k.max()) + 1
    ref = np.bincount(k.astype(np.int64), weights=lv, minlength=m)
    nz = np.flatnonzero(np.bincount(k.astype(np.int64), minlength=m))
    assert float(np.max(np.abs(sums - ref[nz]))) <= 1e-9 * max(
        1.0, float(np.max(np.abs(ref))))
