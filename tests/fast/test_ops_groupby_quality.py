# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""GroupBy quality pass: sorted/hash/fallback multi-value + NA + int boundaries.

Oracle = pure-python dict grouping on small N (reference only).
Seed 42 everywhere.
"""

import time
from pathlib import Path

import pytest

from harness import assert_float_close, assert_int_exact, load_profile

APP_DIR = str(Path(__file__).resolve().parents[2])
PROFILE = load_profile()


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


def _bufs(a, jobs):
    graph = a["optimize"](a["compile"](jobs))
    return a["cpu_execute"](graph["nodes"])


def _oracle(rows):
    """rows: [(key, v1, v2, ...)] -> {key: {'sums': [...], 'count': n}}."""
    acc = {}
    for r in rows:
        k, vs = r[0], r[1:]
        e = acc.setdefault(k, {"sums": [0] * len(vs), "count": 0})
        for i, v in enumerate(vs):
            e["sums"][i] += v
        e["count"] += 1
    return acc


@pytest.mark.fast
def test_sorted_multi_value_exact(kernel):
    a = kernel.alias
    k = [0, 0, 1, 1, 2, 2]
    v1 = [1, 2, 3, 4, 5, 6]
    v2 = [10, 20, 30, 40, 50, 60]
    jobs = [
        a["ir_series"]("v1", v1), a["ir_series"]("v2", v2),
        a["ir_series"]("k", k),
        a["ir_groupby_multi"]("g", ["v1", "v2"], "k",
                              {"v1": ("sum", "count"), "v2": ("sum", "count")}),
    ]
    got = _bufs(a, jobs)["g"]
    assert got[0] == {"v1": {"sum": 3, "count": 2}, "v2": {"sum": 30, "count": 2}}
    assert got[1] == {"v1": {"sum": 7, "count": 2}, "v2": {"sum": 70, "count": 2}}
    assert got[2] == {"v1": {"sum": 11, "count": 2}, "v2": {"sum": 110, "count": 2}}
    assert list(got) == [0, 1, 2]


@pytest.mark.fast
def test_unsorted_hash_multi_value_oracle(kernel):
    import numpy as np

    a = kernel.alias
    rng = np.random.default_rng(42)
    n = 3000
    keys = rng.integers(0, 2000, size=n).tolist()
    v1 = rng.integers(-50, 50, size=n).tolist()
    v2 = (rng.random(n) * 20 - 10).tolist()
    jobs = [
        a["ir_series"]("v1", v1), a["ir_series"]("v2", v2, "float32"),
        a["ir_series"]("k", keys),
        a["ir_groupby_multi"]("g", ["v1", "v2"], "k",
                              {"v1": ("sum", "count"), "v2": ("sum", "mean")}),
    ]
    got = _bufs(a, jobs)["g"]
    ref = _oracle(list(zip(keys, v1, [0] * n)))
    ref2 = _oracle(list(zip(keys, v2, [0] * n)))
    assert set(got) == set(ref)
    assert list(got) == sorted(got)
    for key in ref:
        assert_int_exact(got[key]["v1"]["sum"], ref[key]["sums"][0], label=f"sum[{key}]")
        assert_int_exact(got[key]["v1"]["count"], ref[key]["count"], label=f"cnt[{key}]")
        assert_float_close(got[key]["v2"]["sum"], ref2[key]["sums"][0],
                           PROFILE, "f32", label=f"fsum[{key}]")


@pytest.mark.fast
def test_fallback_unique_multi_value(kernel):
    a = kernel.alias
    keys = [0, 10**9, -7, 10**9, 0, -7, 2**31 - 1]
    v1 = [1, 2, 3, 4, 5, 6, 7]
    v2 = [7, 6, 5, 4, 3, 2, 1]
    jobs = [
        a["ir_series"]("v1", v1), a["ir_series"]("v2", v2),
        a["ir_series"]("k", keys),
        a["ir_groupby_multi"]("g", ["v1", "v2"], "k",
                              {"v1": ("sum", "count"), "v2": ("sum", "count")}),
    ]
    got = _bufs(a, jobs)["g"]
    ref = _oracle(list(zip(keys, v1, v2)))
    assert set(got) == set(ref)
    for key in ref:
        assert_int_exact(got[key]["v1"]["sum"], ref[key]["sums"][0], label=f"u1[{key}]")
        assert_int_exact(got[key]["v2"]["sum"], ref[key]["sums"][1], label=f"u2[{key}]")
        assert_int_exact(got[key]["v1"]["count"], ref[key]["count"], label=f"uc[{key}]")


@pytest.mark.fast
def test_validity_na_filtered_multi(kernel):
    a = kernel.alias
    k = [0, 1, 0, 1, 2, 2]
    v1 = [10, 20, 30, 40, 50, 60]
    v2 = [1, 2, 3, 4, 5, 6]
    # drop rows 1 and 4 (0/1 validity): keep (0,10,1),(0,30,3),(1,40,4),(2,60,6)
    mv = [1, 0, 1, 1, 0, 1]
    jobs = [
        a["ir_series"]("v1", v1, validity=mv),
        a["ir_series"]("v2", v2, validity=mv),
        a["ir_series"]("k", k, validity=mv),
        a["ir_groupby_multi"]("g", ["v1", "v2"], "k",
                              {"v1": ("sum", "count"), "v2": ("sum", "count")}),
    ]
    got = _bufs(a, jobs)["g"]
    assert got[0] == {"v1": {"sum": 40, "count": 2}, "v2": {"sum": 4, "count": 2}}
    assert got[1] == {"v1": {"sum": 40, "count": 1}, "v2": {"sum": 4, "count": 1}}
    assert got[2] == {"v1": {"sum": 60, "count": 1}, "v2": {"sum": 6, "count": 1}}


@pytest.mark.fast
def test_int32_boundary_sums_int64(kernel):
    a = kernel.alias
    big = 2**30
    v1 = [big, big, big, 1]
    k = [0, 0, 1, 1]
    jobs = [
        a["ir_series"]("v1", v1), a["ir_series"]("k", k),
        a["ir_groupby_multi"]("g", "v1", "k", ("sum", "count", "mean"),
                              result="carry"),
    ]
    carry = _bufs(a, jobs)["g"]
    assert carry.sums["v1"].dtype.name == "int64"
    assert carry.counts.dtype.name == "int64"
    d = carry.to_dict_single("v1", ("sum", "count", "mean"))
    assert d[0] == {"sum": 2**31, "count": 2, "mean": 2**30}
    assert d[1] == {"sum": big + 1, "count": 2, "mean": (big + 1) / 2}


@pytest.mark.fast
def test_int32_max_values_exact_int64_sums(kernel):
    a = kernel.alias
    mx = 2**31 - 1
    vals = [mx, 1, mx, -5]
    keys = [3, 3, 4, 4]
    jobs = [
        a["ir_series"]("v", vals), a["ir_series"]("k", keys),
        a["ir_groupby_multi"]("g", "v", "k", ("sum", "count")),
    ]
    got = _bufs(a, jobs)["g"]
    assert got[3] == {"sum": mx + 1, "count": 2}
    assert got[4] == {"sum": mx - 5, "count": 2}


@pytest.mark.fast
def test_counts_shared_across_columns(kernel):
    a = kernel.alias
    k = [5, 5, 6, 6, 6]
    v1 = [1, 2, 3, 4, 5]
    v2 = [10, 20, 30, 40, 50]
    jobs = [
        a["ir_series"]("v1", v1), a["ir_series"]("v2", v2),
        a["ir_series"]("k", k),
        a["ir_groupby_multi"]("g", ["v1", "v2"], "k",
                              {"v1": ("sum", "count"), "v2": ("sum", "count")}),
        a["ir_groupby"]("c", "v1", "k", "count"),
    ]
    bufs = _bufs(a, jobs)
    got, cnt = bufs["g"], bufs["c"]
    for key in (5, 6):
        assert_int_exact(got[key]["v1"]["count"], got[key]["v2"]["count"],
                         label=f"shared[{key}]")
        assert_int_exact(got[key]["v1"]["count"], cnt[key], label=f"oracle[{key}]")


@pytest.mark.fast
def test_sorted_vs_unsorted_same_result(kernel):
    import numpy as np

    a = kernel.alias
    rng = np.random.default_rng(42)
    n = 2000
    keys = rng.integers(0, 50, size=n).tolist()
    v1 = rng.integers(0, 100, size=n).tolist()
    v2 = rng.integers(-100, 0, size=n).tolist()
    srt = sorted(range(n), key=lambda i: keys[i])
    sk, s1, s2 = [keys[i] for i in srt], [v1[i] for i in srt], [v2[i] for i in srt]

    def run(kk, aa, bb):
        return _bufs(a, [
            a["ir_series"]("v1", aa), a["ir_series"]("v2", bb),
            a["ir_series"]("k", kk),
            a["ir_groupby_multi"]("g", ["v1", "v2"], "k",
                                  {"v1": ("sum", "count"), "v2": ("sum", "count")}),
        ])["g"]

    assert run(keys, v1, v2) == run(sk, s1, s2)
