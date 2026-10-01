# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Public Lookup API gate: nf.lookup(build, probe) thin over ir_lookup.

Contract (fixed): inputs both int32 Series (ndarray/float/bool/text ->
explicit ValueError with fix); output struct {positions int32 Series
(index into SORTED-UNIQUE build order, miss -> -1; int64 driver output
narrowed losslessly, max pos = k-1), hit bool Series, k int metadata};
duplicate VALID build keys -> the same ValueError contract as direct
ir_lookup and the Join build ("not unique"); invalid rows never match
(miss, not error): invalid build rows excluded from the table, invalid
probe rows forced to (-1, False). Payload take is composition
(lookup -> gather), shown in two chain examples (inner + left).
Gates: parity API vs direct ir_lookup bit-by-bit + fuzz small (seed 42);
fixed seed 42, stage breakdown (ms) printed per run.
"""

import importlib.util as _ilu
import sys
import time
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[1] / "src")
_FORK = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def nf():
    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)
    import numfast as _nf
    _nf.get_kernel(fresh=True)
    return _nf


def _join_mod():
    spec = _ilu.spec_from_file_location(
        "nflookupapi_join", str(_FORK / "src" / "Relational" / "Join" / "_lib" / "join.py"))
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_lookup_in_all_and_signature(nf):
    t0 = time.perf_counter()
    assert "lookup" in nf.__all__ and "lookup" in dir(nf)
    b = nf.from_numpy(np.array([30, 10, 20], dtype=np.int32), name="b")
    p = nf.from_numpy(np.array([20, 99, 10, 20], dtype=np.int32), name="p")
    r = nf.lookup(b, p)
    assert sorted(r.keys()) == ["hit", "k", "positions"]
    assert r["positions"].dtype == "int32" and r["hit"].dtype == "bool"
    assert isinstance(r["k"], int)
    assert len(nf.to_numpy(r["positions"])) == len(p)
    assert len(nf.to_numpy(r["hit"])) == len(p)
    print(f"\nlookup api stages ms: total={(time.perf_counter()-t0)*1000:.2f}")


def test_lookup_input_contract(nf):
    b = nf.from_numpy(np.array([1, 2], dtype=np.int32))
    # ndarray (not Series) rejected with fix
    with pytest.raises(ValueError, match="NumFast Series"):
        nf.lookup(np.array([1, 2], dtype=np.int32), b)
    with pytest.raises(ValueError, match="NumFast Series"):
        nf.lookup(b, np.array([1, 2], dtype=np.int32))
    # float / bool / text keys rejected (no key semantics)
    for arr in (np.array([1.0, 2.0]), np.array([True, False])):
        with pytest.raises(ValueError, match="int32 keys"):
            nf.lookup(nf.from_numpy(arr), b)
    with pytest.raises(ValueError, match="int32 keys"):
        nf.lookup(b, nf.from_numpy(np.array(["a", "b"], dtype=object)))
    with pytest.raises(ValueError, match="int32 keys"):
        nf.lookup(b, nf.from_numpy(np.array([1.5, 2.5])))


def test_lookup_miss_contract(nf):
    t0 = time.perf_counter()
    b = nf.from_numpy(np.array([30, 10, 20], dtype=np.int32))
    p = nf.from_numpy(np.array([20, 99, 10, 20], dtype=np.int32))
    r = nf.lookup(b, p)
    pos, hit = nf.to_numpy(r["positions"]), nf.to_numpy(r["hit"])
    # sorted-U order: U=[10,20,30]
    assert pos.tolist() == [1, -1, 0, 1]
    assert hit.tolist() == [True, False, True, True]
    assert pos.dtype == np.int32 and hit.dtype == bool and r["k"] == 3
    assert (pos[~hit] == -1).all() and (pos[hit] >= 0).all()
    # edges: empty probe, all-miss, int32 extremes, probe dupes
    e = nf.lookup(b, nf.from_numpy(np.zeros(0, dtype=np.int32)))
    assert len(nf.to_numpy(e["positions"])) == 0 and e["k"] == 3
    m = nf.lookup(b, nf.from_numpy(np.array([1000, 1001], dtype=np.int32)))
    assert (nf.to_numpy(m["positions"]) == -1).all()
    assert not nf.to_numpy(m["hit"]).any()
    lo = np.array([-2147483648, 2147483647], dtype=np.int32)
    g = nf.lookup(nf.from_numpy(lo), nf.from_numpy(lo))
    assert nf.to_numpy(g["hit"]).all()
    print(f"\nlookup miss stages ms: total={(time.perf_counter()-t0)*1000:.2f}")


def test_lookup_dupe_build_contract(nf):
    # Same ValueError contract as direct ir_lookup and the Join build.
    a = nf.get_kernel().alias
    dup = np.array([5, 5, 7], dtype=np.int32)
    bd = nf.from_numpy(dup)
    pr = nf.from_numpy(np.array([5], dtype=np.int32))
    with pytest.raises(ValueError, match="not unique"):
        nf.lookup(bd, pr)
    jobs = [bd._source("b"), pr._source("p"), a["ir_lookup"]("L", "b", "p")]
    graph = a["optimize"](a["compile"](jobs))
    with pytest.raises(ValueError, match="not unique"):
        a["cpu_execute"](graph["nodes"])
    _J = _join_mod()
    with pytest.raises(ValueError, match="not unique"):
        _J.JoinBuild(dup, np.array([1, 2, 3], dtype=np.int32))


def test_lookup_validity_never_matches(nf):
    t0 = time.perf_counter()
    # invalid probe rows forced to (-1, False) even when the value matches
    b = nf.from_numpy(np.array([10, 20, 30], dtype=np.int32))
    pv = nf.from_numpy(np.array([20, 99, 10], dtype=np.int32),
                       validity=np.array([False, True, True]))
    r = nf.lookup(b, pv)
    assert nf.to_numpy(r["positions"]).tolist() == [-1, -1, 0]
    assert nf.to_numpy(r["hit"]).tolist() == [False, False, True]
    # invalid build rows excluded from the table (k counts valid only)
    bv = nf.from_numpy(np.array([10, 20, 30], dtype=np.int32),
                       validity=np.array([True, False, True]))
    p = nf.from_numpy(np.array([20, 10, 30], dtype=np.int32))
    r2 = nf.lookup(bv, p)
    assert nf.to_numpy(r2["positions"]).tolist() == [-1, 0, 1]
    assert nf.to_numpy(r2["hit"]).tolist() == [False, True, True]
    assert r2["k"] == 2
    print(f"\nlookup validity stages ms: total={(time.perf_counter()-t0)*1000:.2f}")


def test_lookup_chain_gather_inner(nf):
    """Chain 1 (inner): lookup -> gather build payload at hits == Join inner."""
    t0 = time.perf_counter()
    _J = _join_mod()
    xk, xv, rk, rv = _J.make_pair(seed=42, n=1000, s=10)
    build = _J.JoinBuild(rk, rv)
    r = nf.lookup(nf.from_numpy(rk, name="rk"), nf.from_numpy(xk, name="xk"))
    pos, hit = nf.to_numpy(r["positions"]), nf.to_numpy(r["hit"])
    # payload take is composition: sorted-U payload gathered at hit positions
    order = np.argsort(rk.astype(np.int64), kind="stable")
    sorted_payload = np.ascontiguousarray(rv[order])
    got_v2 = np.ascontiguousarray(sorted_payload[pos[hit]])
    (ok, o1, o2), _ = _J.join_inner(xk, xv, build, threads=2)
    assert np.array_equal(xk[hit], ok) and np.array_equal(xv[hit], o1)
    assert np.array_equal(got_v2, o2)
    print(f"\nlookup gather-inner stages ms: total={(time.perf_counter()-t0)*1000:.2f}")


def test_lookup_chain_gather_left(nf):
    """Chain 2 (left): lookup -> payload Series with validity=hit == Join left."""
    t0 = time.perf_counter()
    _J = _join_mod()
    xk, xv, rk, rv = _J.make_pair(seed=42, n=1000, s=10)
    build = _J.JoinBuild(rk, rv)
    r = nf.lookup(nf.from_numpy(rk, name="rk"), nf.from_numpy(xk, name="xk"))
    pos, hit = nf.to_numpy(r["positions"]), nf.to_numpy(r["hit"])
    order = np.argsort(rk.astype(np.int64), kind="stable")
    sorted_payload = np.ascontiguousarray(rv[order])
    full = np.where(hit, sorted_payload[np.where(hit, pos, 0)], np.int32(0))
    out = nf.from_numpy(full, name="v2", validity=hit)
    (lk, l1, l2, valid), _ = _J.join_left(xk, xv, build, threads=2)
    assert np.array_equal(nf.to_numpy(out), np.where(valid, l2, 0))
    assert np.array_equal(out.validity, valid)
    assert np.array_equal(xk, lk) and np.array_equal(xv, l1)
    print(f"\nlookup gather-left stages ms: total={(time.perf_counter()-t0)*1000:.2f}")


def test_lookup_parity_direct_bitwise_and_fuzz(nf):
    """API result vs direct ir_lookup bit-by-bit (valid domain) + fuzz small."""
    t0 = time.perf_counter()
    a = nf.get_kernel().alias
    rng = np.random.default_rng(42)
    bad = 0
    for t in range(200):
        m = int(rng.integers(0, 64))
        n = int(rng.integers(0, 64))
        B = np.unique(rng.integers(-50, 50, max(m, 1)).astype(np.int32))
        P = rng.integers(-50, 50, max(n, 1)).astype(np.int32)
        b, p = nf.from_numpy(B), nf.from_numpy(P)
        r = nf.lookup(b, p)
        jobs = [b._source("b"), p._source("p"), a["ir_lookup"]("L", "b", "p")]
        graph = a["optimize"](a["compile"](jobs))
        bufs = a["cpu_execute"](graph["nodes"])
        api_pos = nf.to_numpy(r["positions"]).astype(np.int64)
        api_hit = nf.to_numpy(r["hit"])
        if not (np.array_equal(api_pos, np.asarray(bufs["L"]))
                and np.array_equal(api_hit, np.asarray(bufs["L#hit"]))
                and r["k"] == int(bufs["L#k"])):
            bad += 1
    assert bad == 0, f"{bad}/200 API vs ir_lookup mismatch"
    print(f"\nlookup parity stages ms: total={(time.perf_counter()-t0)*1000:.2f}")
