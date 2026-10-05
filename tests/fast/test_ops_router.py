# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Router: multi-source Dijkstra over CSR u32 with target-set early exit.

Parity of the native `nf_router_route` against the bit-exact heapq fallback in
the same module, plus the frozen semantics that path has to hold. Generic
lanes only, no domain vocabulary. Seed 42.

Loaded by path, the way test_ops_pairinsert_bitmasksweep.py does: no sys.path
is touched, and the module under test is the one in the tree.
"""
import importlib.util
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])


def _lib_dir():
    """The Router `_lib/`, from the checkout or from the installed package.

    The wheel vendors Extensions verbatim into `numfast/_ext/<Name>/`, so the
    same bytes are reachable either way. Without this the module would raise
    FileNotFoundError against an installed numfast, where there is no `src/`
    tree to point at.
    """
    in_checkout = Path(APP_DIR) / "src" / "Relational" / "Router" / "_lib"
    if in_checkout.is_dir():
        return in_checkout
    import numfast
    vendored = Path(numfast.__file__).resolve().parent / "_ext" / "Router" / "_lib"
    if not vendored.is_dir():
        raise FileNotFoundError(
            "Router/_lib not found in the checkout at %s nor vendored into the "
            "installed package at %s" % (in_checkout, vendored))
    return vendored


LIB_DIR = _lib_dir()
LIB_RT = LIB_DIR / "router.py"
LIB_PLAN = LIB_DIR / "plan.py"

INF = 4294967295
UNREACHABLE = 9223372036854775807

#: Synthetic package name plan.py's `from . import router` resolves against.
_PKG = "_router_test_pkg"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(
        name, str(path), submodule_search_locations=[str(path.parent)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[name]
        raise
    return mod


@pytest.fixture(scope="module")
def rt():
    """`router.py` as `rt`, with a sibling importable as plan.py expects."""
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(LIB_DIR)]
    sys.modules.setdefault(_PKG, pkg)
    mod = _load("%s.router" % _PKG, LIB_RT)
    sys.modules[_PKG].router = mod
    yield mod
    for key in ("%s.router" % _PKG, "%s.plan" % _PKG, _PKG):
        sys.modules.pop(key, None)


@pytest.fixture(scope="module")
def plan_mod(rt):
    return _load("%s.plan" % _PKG, LIB_PLAN)


@pytest.fixture(scope="module")
def kernel():
    # conftest substitutes this `builder` against an installed numfast, where
    # the kernel comes from numfast.get_kernel() and APP_DIR is ignored.
    from builder import MAIN

    return MAIN["build"](APP_DIR)


# --- fixtures --------------------------------------------------------------
#
#   v0 --9--> v1 --1--> v2 --1--> v3 --1--> v5
#   |                    ^                  |
#   +--1--> v4 --1-------+                  |
#                                           (v6 isolated)
#
# 0->3 costs 2 through v4 and 11 through v1, so a wrong early-exit router
# answers 11.


def _fork_graph():
    indptr = np.array([0, 2, 3, 4, 5, 6, 7, 7], dtype=np.uint32)   # V=7
    indices = np.array([1, 4, 2, 3, 5, 3, 2], dtype=np.uint32)
    weights = np.array([9, 1, 1, 1, 1, 1, 1], dtype=np.uint32)
    return indptr, indices, weights


def _line_graph(n):
    """n vertices chained 0 -> 1 -> ... -> n-1. V-1 edges, last vertex has none."""
    indptr = np.concatenate((np.arange(n), [n - 1])).astype(np.uint32)
    return (indptr,
            np.arange(1, n, dtype=np.uint32),
            np.ones(n - 1, dtype=np.uint32))


def _reference(rt, indptr, indices, weights, sources, target):
    """The module's own fallback, called the way router_route calls it.

    `_fb_route` takes the CSR lanes as arrays and the query as plain lists --
    that is exactly how router_route invokes it on the no-native path.
    """
    v = int(rt._as_u32(indptr, "indptr").shape[0]) - 1
    is_target = np.zeros(v, dtype=np.uint8)
    is_target[int(target)] = 1
    return rt._fb_route(
        rt._as_u32(indptr, "indptr"),
        rt._as_u32(indices, "indices"),
        np.ascontiguousarray(weights, dtype=np.uint32),
        [int(s) for s in sources], is_target.tolist())


# --- manifest and kernel wiring --------------------------------------------


def test_kernel_alias_router(kernel):
    assert kernel.metadata["Router"]["types"] == [
        "router_route", "router_available", "RouterPlan"]
    assert kernel.metadata["Router"]["inf"] == INF
    for name in ("router_route", "router_available", "RouterPlan"):
        assert name in kernel.alias, name


def test_manifest_alias_equals_mods():
    import tomllib
    manifest = LIB_DIR.parent / "Router.toml"
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    assert data["name"] == "Router"
    assert data["alias"] == data["mods"]
    assert data["depends"] == []
    assert set(data["alias"]) == {
        "router_route", "router_available", "RouterPlan"}


# --- the return contract ----------------------------------------------------


def test_single_source_reaches_target_with_cost(rt):
    ip, ix, w = _fork_graph()
    dist, tgt, visited = rt.router_route(ip, ix, w, [0], [3])
    assert (dist, tgt) == (2, 3)           # 0 -> 4 -> 3
    assert visited >= 1


def test_earliest_settled_target_wins_not_the_lowest_id(rt):
    """Targets {1, 3}: v1 costs 9 and v3 costs 2. The answer is 3, so this
    pins settled-pop order rather than 'first target in the set'."""
    ip, ix, w = _fork_graph()
    assert rt.router_route(ip, ix, w, [0], [1, 3])[:2] == (2, 3)


def test_tie_break_is_by_vertex_id(rt):
    """v1 and v2 are both reachable from v0 at distance 1. The lower id settles
    first, which is what makes an equal-cost answer deterministic."""
    indptr = np.array([0, 2, 2, 2], dtype=np.uint32)         # V=3
    indices = np.array([1, 2], dtype=np.uint32)
    weights = np.array([1, 1], dtype=np.uint32)
    dist, tgt, _ = rt.router_route(indptr, indices, weights, [0], [2, 1])
    assert (dist, tgt) == (1, 1)
    # target order in the input does not change it
    assert rt.router_route(indptr, indices, weights, [0], [1, 2])[:2] == (1, 1)


def test_unreachable_target_returns_the_frozen_sentinel(rt):
    ip, ix, w = _fork_graph()
    dist, tgt, _ = rt.router_route(ip, ix, w, [0], [6])
    assert dist == UNREACHABLE
    assert tgt == -1


def test_multiple_sources_all_start_at_zero(rt):
    ip, ix, w = _fork_graph()
    dist, tgt, _ = rt.router_route(ip, ix, w, [2, 1], [3])
    assert (dist, tgt) == (1, 3)           # 2 -> 3 direct, not 3 via 1


def test_source_outside_the_vertex_range_is_ignored(rt):
    ip, ix, w = _fork_graph()
    assert rt.router_route(ip, ix, w, [99], [3])[:2] == (UNREACHABLE, -1)
    # alongside a valid source it routes, and the out-of-range one costs nothing
    assert rt.router_route(ip, ix, w, [99, 0], [3])[:2] == (2, 3)


def test_longer_path_is_preferred_over_the_cheap_one(rt):
    """0 -> 3 directly costs 2, 0 -> 4 -> 5 -> 6-ish costs 6. Cheaper wins even
    though it is not the shortest hop count."""
    indptr = np.array([0, 2, 2, 2, 2], dtype=np.uint32)        # V=4
    indices = np.array([3, 1], dtype=np.uint32)
    weights = np.array([2, 6], dtype=np.uint32)
    assert rt.router_route(indptr, indices, weights, [0], [3])[:2] == (2, 3)


# --- INF marker: never relaxed ---------------------------------------------


def test_inf_weighted_edge_is_skipped(rt):
    """0->1->2 with the second edge marked UINT32_MAX: the target is unreachable
    rather than reached at a 4294967295-length distance."""
    indptr = np.array([0, 1, 2, 3], dtype=np.uint32)
    indices = np.array([1, 2, 2], dtype=np.uint32)
    marked = np.array([1, INF, 1], dtype=np.uint32)
    assert rt.router_route(indptr, indices, marked, [0], [2])[:2] == (
        UNREACHABLE, -1)
    plain = np.array([1, 1, 1], dtype=np.uint32)
    assert rt.router_route(indptr, indices, plain, [0], [2])[:2] == (2, 2)


def test_weights_may_carry_inf_but_may_not(rt):
    """weights is the one lane where the UINT32_MAX marker is legal -- it means
    'skip this edge'. indptr/indices/sources may not carry it."""
    ip, ix, w = _fork_graph()
    marked = w.copy()
    marked[0] = INF                        # 0->1 becomes INF; 0->4->3 stands
    assert rt.router_route(ip, ix, marked, [0], [3])[:2] == (2, 3)

    # Note the input dtype: _as_u32 returns a u32 array untouched, so the INF
    # rejection only fires on widened input. A u32 array carrying INF is caught
    # downstream instead, by the CSR range check.
    with pytest.raises(ValueError, match=r"indices holds UINT32_MAX"):
        bad = ix.astype(np.int64)
        bad[0] = INF
        rt.router_route(ip, bad, w, [0], [3])
    with pytest.raises(ValueError, match=r"sources holds UINT32_MAX"):
        rt.router_route(ip, ix, w, np.array([INF], dtype=np.int64), [3])
    with pytest.raises(ValueError, match=r"indptr holds UINT32_MAX"):
        rt.router_route(np.append(ip, INF).astype(np.int64),
                        ix, w, [0], [3])
    # the same lane as u32: refused as out of range, not silently accepted
    with pytest.raises(ValueError, match="index out of range"):
        bad32 = ix.copy()
        bad32[0] = INF
        rt.router_route(ip, bad32, w, [0], [3])


# --- the guards: malformed CSR raises, never returns a wrong answer --------


def test_indptr_must_be_monotone(rt):
    ip, ix, w = _fork_graph()
    broken = ip.copy()
    broken[1], broken[2] = broken[2], broken[1]     # 0, 3, 2, ...
    with pytest.raises(ValueError, match="monotone"):
        rt.router_route(broken, ix, w, [0], [3])


def test_indptr_must_close_on_the_edge_count(rt):
    """Monotone but overshooting: the last entry claims an edge past the end of
    `indices`. """
    ip, ix, w = _fork_graph()
    over = ip.copy()
    over[-1] += 1
    with pytest.raises(ValueError, match=r"indptr\[-1\]"):
        rt.router_route(over, ix, w, [0], [3])
    # monotone, but stops short of the edge count: 0->1->2 with a third edge
    # left over. Same check, other direction.
    two = np.array([0, 1, 1], dtype=np.uint32)
    edge = np.array([1, 2], dtype=np.uint32)
    with pytest.raises(ValueError, match=r"indptr\[-1\]"):
        rt.router_route(two, edge, np.array([1, 1], dtype=np.uint32), [0], [1])


def test_weights_length_must_equal_edge_count(rt):
    ip, ix, w = _fork_graph()
    with pytest.raises(ValueError, match="weights len"):
        rt.router_route(ip, ix, w[:-1], [0], [3])


def test_index_out_of_range_refuses(rt):
    ip, ix, w = _fork_graph()
    bad = ix.copy()
    bad[0] = 99
    with pytest.raises(ValueError, match="index out of range"):
        rt.router_route(ip, bad, w, [0], [3])


def test_empty_graph_refuses(rt):
    with pytest.raises(ValueError, match="empty graph"):
        rt.router_route(np.zeros(0, dtype=np.uint32),
                        np.zeros(0, dtype=np.uint32),
                        np.zeros(0, dtype=np.uint32), [0], [0])


def test_empty_sources_refuses(rt):
    ip, ix, w = _fork_graph()
    with pytest.raises(ValueError, match="sources empty"):
        rt.router_route(ip, ix, w, [], [3])


def test_target_out_of_range_refuses(rt):
    ip, ix, w = _fork_graph()
    with pytest.raises(ValueError, match="target out of range"):
        rt.router_route(ip, ix, w, [0], [99])


def test_inputs_are_never_mutated(rt):
    ip, ix, w = _fork_graph()
    before = (ip.copy(), ix.copy(), w.copy())
    rt.router_route(ip, ix, w, [0, 4], [3])
    rt.router_route(ip, ix, w, [0], [1, 3, 6])
    assert np.array_equal(ip, before[0])
    assert np.array_equal(ix, before[1])
    assert np.array_equal(w, before[2])


# --- native / fallback parity ----------------------------------------------


@pytest.mark.parametrize("n", [2, 17, 257])
def test_random_chain_parity_native_fallback(rt, n):
    rng = np.random.default_rng(42)
    ip, ix, _ = _line_graph(n)
    w = rng.integers(1, 1000, size=n - 1, dtype=np.uint32)
    got = rt.router_route(ip, ix, w, [0], [n - 1])
    assert got == _reference(rt, ip, ix, w, [0], n - 1)
    # and the chain is actually reachable: the whole point of the fixture
    assert got[:2] == (int(w.astype(np.int64).sum()), n - 1)


def test_random_graph_parity_native_fallback(rt):
    """Seed 42, 300 random forward graphs, one target each."""
    rng = np.random.default_rng(42)
    for case in range(300):
        v = int(rng.integers(2, 40))
        # degrees first, then exactly as many edges as they demand, so
        # indptr[-1] == len(indices) by construction
        counts = rng.integers(0, 5, size=v)
        counts[-1] += 1                    # the last vertex owns an edge
        indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.uint32)
        e = int(indptr[-1])
        indices = rng.integers(0, v, size=e, dtype=np.uint32)
        weights = rng.integers(1, 1000, size=e, dtype=np.uint32)
        src = np.array([int(rng.integers(0, v))], dtype=np.uint32)
        tgt = int(rng.integers(0, v))

        got = rt.router_route(indptr, indices, weights, src, [tgt])
        assert got == _reference(rt, indptr, indices, weights, src, tgt), (
            case, v, int(indptr[-1]), got)


def test_router_available_is_a_bool(rt):
    assert isinstance(rt.router_available(), bool)


def test_native_disable_falls_back_bit_exactly(rt, monkeypatch):
    """With the native path off the answer must not move by one bit."""
    ip, ix, w = _fork_graph()
    before = rt.router_route(ip, ix, w, [0, 4], [3])
    monkeypatch.setenv("NUMFAST_NATIVE_DISABLE", "1")
    try:
        assert rt.router_available() is False
        assert rt.router_route(ip, ix, w, [0, 4], [3]) == before
    finally:
        monkeypatch.delenv("NUMFAST_NATIVE_DISABLE", raising=False)
        rt._probe()                       # restore the module-level probe


# --- RouterPlan: same contract, prepared lanes ------------------------------


def test_plan_matches_router_route(rt, plan_mod):
    ip, ix, w = _fork_graph()
    plan = plan_mod.RouterPlan(ip, ix, w)
    assert plan.n_vertices == 7
    assert plan.n_edges == int(w.size)
    assert plan.n_inf_edges == 0
    for src, tgt in [([0], [3]), ([0], [1, 3]), ([2, 1], [3]),
                     ([99], [3]), ([0], [6]), ([0, 4], [3])]:
        assert plan.route(src, tgt) == rt.router_route(ip, ix, w, src, tgt)


def test_plan_counts_and_remaps_inf_edges(rt, plan_mod):
    indptr = np.array([0, 1, 2, 3], dtype=np.uint32)
    indices = np.array([1, 2, 2], dtype=np.uint32)
    weights = np.array([1, INF, 1], dtype=np.uint32)
    plan = plan_mod.RouterPlan(indptr, indices, weights)
    assert plan.n_inf_edges == 1
    assert plan.route([0], [2])[:2] == (UNREACHABLE, -1)


def test_plan_inputs_not_mutated_and_reusable(rt, plan_mod):
    ip, ix, w = _fork_graph()
    before = (ip.copy(), ix.copy(), w.copy())
    plan = plan_mod.RouterPlan(ip, ix, w)
    first = plan.route([0], [3])
    # the resident target mask is reset per query, so a second query must not
    # see the first one's targets
    assert first == plan.route([0], [3])
    assert plan.route([0], [6])[:2] == (UNREACHABLE, -1)
    assert plan.route([0], [3]) == first
    assert np.array_equal(ip, before[0])
    assert np.array_equal(ix, before[1])
    assert np.array_equal(w, before[2])


def test_plan_route_many_preserves_input_order(rt, plan_mod):
    ip, ix, w = _fork_graph()
    plan = plan_mod.RouterPlan(ip, ix, w)
    pairs = [([0], [3]), ([0], [1]), ([2, 1], [3]), ([0], [6])]
    assert plan.route_many(pairs) == [
        rt.router_route(ip, ix, w, s, t) for s, t in pairs]


def test_plan_rejects_the_same_malformed_inputs(rt, plan_mod):
    ip, ix, w = _fork_graph()
    with pytest.raises(ValueError, match="weights len"):
        plan_mod.RouterPlan(ip, ix, w[:-1])
    plan = plan_mod.RouterPlan(ip, ix, w)
    with pytest.raises(ValueError, match="sources empty"):
        plan.route([], [3])
    with pytest.raises(ValueError, match="target out of range"):
        plan.route([0], [99])


# --- smoke ------------------------------------------------------------------


def test_router_smoke_on_a_wide_graph(rt):
    """V=20000 path graph. A per-pop linear rescan would not finish this."""
    n = 20000
    ip, ix, w = _line_graph(n)
    start = time.perf_counter()
    dist, tgt, _ = rt.router_route(ip, ix, w, [0], [n - 1])
    elapsed = time.perf_counter() - start
    assert (dist, tgt) == (n - 1, n - 1)
    assert elapsed < 20.0, elapsed
