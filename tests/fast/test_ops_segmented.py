# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""P1 segmented_reduce + P2 adjacency_slice: unit/property + workload checks.

Fast tests (seed 42). Every case is a small runnable example over generic
integer lanes -- no dataset names, no domain vocabulary.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from Relational.Segmented._lib.adjacency import (  # noqa: E402
    adjacency_flat,
    adjacency_slice,
)
from Relational.Segmented._lib.segmented import (  # noqa: E402
    MAX_DISPATCH_N,
    segmented_reduce,
)

RNG = np.random.default_rng(42)


def _naive(values, bounds, op):
    vals = np.asarray(values)
    b = np.asarray(bounds).astype(np.int64)
    out = []
    for i in range(b.size - 1):
        seg = vals[b[i]:b[i + 1]]
        if op == "sum":
            out.append(seg.sum() if seg.size else 0)
        elif op == "count":
            out.append(seg.size)
        elif op == "min":
            out.append(seg.min())
        elif op == "max":
            out.append(seg.max())
    return out


@pytest.mark.fast
def test_p1_one_group_sum_count_min_max():
    v = np.array([3, 1, 2], dtype=np.int32)
    b = np.array([0, 3], dtype=np.uint32)
    assert segmented_reduce(v, b, "sum").tolist() == [6]
    assert segmented_reduce(v, b, "count").tolist() == [3]
    assert segmented_reduce(v, b, "min").tolist() == [1]
    assert segmented_reduce(v, b, "max").tolist() == [3]


@pytest.mark.fast
def test_p1_m_equals_n_singletons():
    v = np.array([5, -2, 7], dtype=np.int32)
    b = np.array([0, 1, 2, 3], dtype=np.uint32)
    assert segmented_reduce(v, b, "sum").tolist() == [5, -2, 7]
    assert segmented_reduce(v, b, "count").tolist() == [1, 1, 1]
    assert segmented_reduce(v, b, "min").tolist() == [5, -2, 7]


@pytest.mark.fast
def test_p1_empty_group_sum0_count0_minmax_raise():
    v = np.array([1, 2], dtype=np.int32)
    b = np.array([0, 1, 1, 2], dtype=np.uint32)
    assert segmented_reduce(v, b, "sum").tolist() == [1, 0, 2]
    assert segmented_reduce(v, b, "count").tolist() == [1, 0, 1]
    with pytest.raises(ValueError):
        segmented_reduce(v, b, "min")
    with pytest.raises(ValueError):
        segmented_reduce(v, b, "max")


@pytest.mark.fast
def test_p1_n0_m0():
    v = np.zeros(0, dtype=np.int32)
    b = np.array([0], dtype=np.uint32)
    assert segmented_reduce(v, b, "sum").size == 0
    assert segmented_reduce(v, b, "count").dtype == np.uint32
    vf = np.zeros(0, dtype=np.float32)
    assert segmented_reduce(vf, b, "sum").dtype == np.float32


@pytest.mark.fast
def test_p1_bounds_validation():
    v = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(ValueError):
        segmented_reduce(v, np.array([0, 2, 1, 3], dtype=np.uint32), "sum")
    with pytest.raises(ValueError):
        segmented_reduce(v, np.array([0, 2], dtype=np.uint32), "sum")
    with pytest.raises(ValueError):
        segmented_reduce(v, np.array([0, 4], dtype=np.uint32), "sum")
    with pytest.raises(ValueError):
        segmented_reduce(v, np.array([0, 3], dtype=np.uint32), "mean")


@pytest.mark.fast
def test_p1_i32_saturating():
    big = np.array([2000000000, 2000000000], dtype=np.int32)
    b = np.array([0, 2], dtype=np.uint32)
    assert segmented_reduce(big, b, "sum").tolist() == [2147483647]
    neg = np.array([-2000000000, -2000000000], dtype=np.int32)
    assert segmented_reduce(neg, b, "sum").tolist() == [-2147483648]


@pytest.mark.fast
def test_p1_f32_ieee_nan_propagates():
    v = np.array([1.0, float("nan"), 2.0], dtype=np.float32)
    b = np.array([0, 3], dtype=np.uint32)
    out = segmented_reduce(v, b, "sum")
    assert bool(np.isnan(out[0]))


@pytest.mark.fast
def test_p1_count_u32():
    v = np.array([9, 8, 7, 6], dtype=np.int32)
    b = np.array([0, 2, 4], dtype=np.uint32)
    out = segmented_reduce(v, b, "count")
    assert out.dtype == np.uint32
    assert out.tolist() == [2, 2]


@pytest.mark.fast
def test_p1_limit_flag():
    assert MAX_DISPATCH_N == 4194240
    v = np.zeros(3, dtype=np.int32)
    with pytest.raises(ValueError):
        segmented_reduce(np.zeros(MAX_DISPATCH_N + 1, dtype=np.int32),
                         np.array([0, MAX_DISPATCH_N + 1],
                                  dtype=np.int64), "sum")
    assert v.size <= MAX_DISPATCH_N


@pytest.mark.fast
def test_p1_property_vs_naive_seed42():
    rng = np.random.default_rng(42)
    for trial in range(30):
        n = int(rng.integers(0, 40))
        m = int(rng.integers(0, 8)) if n else 0
        cuts = sorted(rng.integers(0, n + 1, size=m).tolist()) if m else []
        if n == 0:
            bounds = np.array([0], dtype=np.uint32)
            vals = np.zeros(0, dtype=np.int32)
        else:
            bounds = np.array([0] + cuts + [n], dtype=np.uint32)
            bounds = np.unique(bounds)
            vals = rng.integers(-50, 50, size=n).astype(np.int32)
        for op in ("sum", "count"):
            got = segmented_reduce(vals, bounds, op).tolist()
            want = _naive(vals, bounds, op)
            assert [int(x) for x in got] == [int(x) for x in want]
        non_empty = bool(np.all(np.diff(bounds.astype(np.int64)) > 0))
        if non_empty and len(bounds) > 1:
            for op in ("min", "max"):
                got = segmented_reduce(vals, bounds, op).tolist()
                assert [int(x) for x in got] == \
                    [int(x) for x in _naive(vals, bounds, op)]


@pytest.mark.fast
def test_p1_determinism():
    v = RNG.integers(-100, 100, size=200).astype(np.int32)
    cuts = np.sort(RNG.integers(0, 201, size=9))
    b = np.unique(np.concatenate(([0], cuts, [200]))).astype(np.uint32)
    a = segmented_reduce(v, b, "sum")
    c = segmented_reduce(v, b, "sum")
    assert np.array_equal(a, c)


@pytest.mark.fast
def test_p2_basic_begins_ends_flat():
    indptr = np.array([0, 2, 2, 5], dtype=np.uint32)
    indices = np.array([1, 2, 0, 1, 2], dtype=np.uint32)
    query = np.array([0, 1, 2], dtype=np.uint32)
    b, e = adjacency_slice(indptr, indices, query)
    assert b.tolist() == [0, 2, 2]
    assert e.tolist() == [2, 2, 5]
    assert b.dtype == np.uint32 and e.dtype == np.uint32
    flat = adjacency_flat(indices, b, e)
    assert flat.tolist() == [1, 2, 0, 1, 2]


@pytest.mark.fast
def test_p2_empty_safe():
    b, e = adjacency_slice(np.array([0], dtype=np.uint32),
                           np.zeros(0, dtype=np.uint32),
                           np.zeros(0, dtype=np.uint32))
    assert b.size == 0 and e.size == 0
    b, e = adjacency_slice(np.array([0, 0], dtype=np.uint32),
                           np.zeros(0, dtype=np.uint32),
                           np.array([0], dtype=np.uint32))
    assert (b.tolist(), e.tolist()) == ([0], [0])


@pytest.mark.fast
def test_p2_inf_rejected():
    inf = np.array([0, 4294967295], dtype=np.uint32)
    with pytest.raises(ValueError):
        adjacency_slice(inf, np.zeros(0, dtype=np.uint32),
                        np.array([0], dtype=np.uint32))
    with pytest.raises(ValueError):
        adjacency_slice(np.array([0, 1], dtype=np.uint32),
                        np.array([4294967295], dtype=np.uint32),
                        np.array([0], dtype=np.uint32))
    with pytest.raises(ValueError):
        adjacency_slice(np.array([0, 0], dtype=np.uint32),
                        np.zeros(0, dtype=np.uint32),
                        np.array([4294967295], dtype=np.uint32))


@pytest.mark.fast
def test_p2_monotone_and_range():
    with pytest.raises(ValueError):
        adjacency_slice(np.array([0, 2, 1], dtype=np.uint32),
                        np.array([0, 0], dtype=np.uint32),
                        np.array([0], dtype=np.uint32))
    with pytest.raises(ValueError):
        adjacency_slice(np.array([0, 0], dtype=np.uint32),
                        np.zeros(0, dtype=np.uint32),
                        np.array([5], dtype=np.uint32))


# --- workload checks (малые runnable-примеры, бенчмарки не тронуты) ---
@pytest.mark.fast
def test_h2o_groupby_jobid_via_p1():
    job = np.array([0, 1, 0, 2, 1, 0], dtype=np.int32)
    val = np.array([10, 20, 30, 40, 50, 60], dtype=np.int32)
    order = np.argsort(job, kind="stable")
    sj, sv = job[order], val[order]
    change = np.empty(sj.size, dtype=bool)
    change[0] = True
    change[1:] = sj[1:] != sj[:-1]
    starts = np.flatnonzero(change).astype(np.uint32)
    bounds = np.concatenate([starts, [sj.size]]).astype(np.uint32)
    ukeys = sj[starts]
    assert ukeys.tolist() == [0, 1, 2]
    assert segmented_reduce(sv, bounds, "count").tolist() == [3, 2, 1]
    assert segmented_reduce(sv, bounds, "sum").tolist() == [100, 70, 40]
    assert segmented_reduce(sv, bounds, "min").tolist() == [10, 20, 40]
    assert segmented_reduce(sv, bounds, "max").tolist() == [60, 50, 40]


@pytest.mark.fast
def test_clickbench_groupby_key_via_p1():
    key = np.array([2, 0, 2, 1, 0], dtype=np.int32)
    val = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    order = np.argsort(key, kind="stable")
    sk, sv = key[order], val[order]
    change = np.empty(sk.size, dtype=bool)
    change[0] = True
    change[1:] = sk[1:] != sk[:-1]
    starts = np.flatnonzero(change).astype(np.uint32)
    bounds = np.concatenate([starts, [sk.size]]).astype(np.uint32)
    assert sk[starts].tolist() == [0, 1, 2]
    assert segmented_reduce(sv, bounds, "count").tolist() == [2, 1, 2]
    assert segmented_reduce(sv, bounds, "sum").tolist() == [7, 4, 4]


@pytest.mark.fast
def test_taxi_zone_agg_via_p1():
    zone = np.array([3, 1, 3, 1, 2], dtype=np.int32)
    fare = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    order = np.argsort(zone, kind="stable")
    sz, sf = zone[order], fare[order]
    change = np.empty(sz.size, dtype=bool)
    change[0] = True
    change[1:] = sz[1:] != sz[:-1]
    starts = np.flatnonzero(change).astype(np.uint32)
    bounds = np.concatenate([starts, [sz.size]]).astype(np.uint32)
    assert segmented_reduce(sf, bounds, "sum").tolist() == [60.0, 50.0, 40.0]


@pytest.mark.fast
def test_sparse_index_via_p2():
    indptr = np.array([0, 2, 3, 3], dtype=np.uint32)
    indices = np.array([1, 2, 0], dtype=np.uint32)
    b, e = adjacency_slice(indptr, indices, np.array([0, 1, 2],
                                                     dtype=np.uint32))
    assert (b.tolist(), e.tolist()) == ([0, 2, 3], [2, 3, 3])
    assert adjacency_flat(indices, b, e).tolist() == [1, 2, 0]


@pytest.mark.fast
def test_run_boundaries_and_repeats_via_p1():
    coarse = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    w = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    order = np.argsort(coarse, kind="stable")
    sc, sw = coarse[order], w[order]
    change = np.empty(sc.size, dtype=bool)
    change[0] = True
    change[1:] = sc[1:] != sc[:-1]
    starts = np.flatnonzero(change).astype(np.uint32)
    bounds = np.concatenate([starts, [sc.size]]).astype(np.uint32)
    assert segmented_reduce(sw, bounds, "sum").tolist() == [3, 12]
    origin = np.array([0, 1, 0, 0, 2], dtype=np.int32)
    one = np.ones(5, dtype=np.int32)
    o = np.argsort(origin, kind="stable")
    so = origin[o]
    ch = np.empty(so.size, dtype=bool)
    ch[0] = True
    ch[1:] = so[1:] != so[:-1]
    st = np.flatnonzero(ch).astype(np.uint32)
    bo = np.concatenate([st, [so.size]]).astype(np.uint32)
    assert segmented_reduce(one[o], bo, "sum").tolist() == [3, 1, 1]
