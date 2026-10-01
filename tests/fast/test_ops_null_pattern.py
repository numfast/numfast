# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: pattern-encoded strings + null bitmap (DELTA-3/DELTA-4).

encode-once prefix+int -> int32 codes + validity sidecar; downstream ints
only. NA: skip with skipna, NaN+warning without. Sort fast-path measured.
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


@pytest.mark.fast
def test_encode_pattern_golden(kernel):
    a = kernel.alias
    jobs = [a["ir_encode_pattern"]("c", ["city#1", "city#2", None, "xx", "city#x"], "city#")]
    bufs = _bufs(a, jobs)
    assert list(bufs["c"]) == [1, 2, 0, 0, 0]
    assert list(bufs["c#validity"]) == [True, True, False, False, False]


@pytest.mark.fast
def test_pattern_decode_roundtrip(kernel):
    a = kernel.alias
    bufs = _bufs(a, [a["ir_encode_pattern"]("c", ["g#7", None, "g#42"], "g#")])
    back = a["pattern_decode"](list(bufs["c"]), "g#", list(bufs["c#validity"]))
    assert back == ["g#7", None, "g#42"]


@pytest.mark.fast
def test_encode_pattern_downstream_ints_only(kernel):
    a = kernel.alias
    bufs = _bufs(a, [
        a["ir_encode_pattern"]("c", ["g#1", "g#2", "g#1", None], "g#"),
        a["ir_series"]("v", [10, 20, 30, 40]),
        a["ir_groupby"]("s", "v", "c", "sum"),
    ])
    assert bufs["s"] == {1: 40, 2: 20}  # None-key row skipped, no strings touched


@pytest.mark.fast
def test_encode_empty_prefix_rejected(kernel):
    a = kernel.alias
    with pytest.raises(ValueError, match="non-empty str"):
        a["ir_encode_pattern"]("c", ["a#1"], "")


@pytest.mark.fast
def test_reduce_validity_skipna(kernel):
    a = kernel.alias
    jobs = lambda op, skip: [a["ir_series"]("s", [10, 20, 30], validity=[1, 0, 1]),
                             a["ir_reduce"]("r", "s", op, skipna=skip)]
    assert _bufs(a, jobs("sum", True))["r"] == 40
    assert _bufs(a, jobs("count", True))["r"] == 2
    assert _bufs(a, jobs("count", False))["r"] == 3
    assert_float_close(_bufs(a, jobs("mean", True))["r"], 20.0, PROFILE, "f64", label="masked mean")


@pytest.mark.fast
def test_reduce_validity_no_skipna_warns_nan(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [10, 20], validity=[1, 0]), a["ir_reduce"]("r", "s", "sum")]
    with pytest.warns(UserWarning, match="skipna=False"):
        got = _bufs(a, jobs)["r"]
    assert got != got  # NaN


@pytest.mark.fast
def test_reduce_all_na_skipna(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [10, 20], validity=[0, 0]), a["ir_reduce"]("r", "s", "count", skipna=True)]
    assert _bufs(a, jobs)["r"] == 0
    jobs = [a["ir_series"]("s", [10, 20], validity=[0, 0]), a["ir_reduce"]("r", "s", "sum", skipna=True)]
    with pytest.warns(UserWarning, match="all-NA"):
        got = _bufs(a, jobs)["r"]
    assert got != got  # NaN


@pytest.mark.fast
def test_groupby_skips_invalid_rows(kernel):
    a = kernel.alias
    bufs = _bufs(a, [
        a["ir_series"]("v", [1, 2, 3], validity=[1, 0, 1]),
        a["ir_series"]("k", [0, 0, 1]),
        a["ir_groupby"]("g", "v", "k", "sum"),
    ])
    assert bufs["g"] == {0: 1, 1: 3}


@pytest.mark.fast
def test_map_propagates_validity(kernel):
    a = kernel.alias
    bufs = _bufs(a, [
        a["ir_series"]("s", [1, 2, 3], validity=[1, 0, 1]),
        a["ir_map"]("m", "s", "mul", 10),
        a["ir_reduce"]("r", "m", "sum", skipna=True),
    ])
    assert_int_exact(bufs["r"], 40, label="masked map+reduce")


@pytest.mark.fast
def test_microbench_sorted_fast_path(kernel):
    """Ordered scan on sorted keys must not lose vs dict+sort. Seed 42.

    Correctness compares the SAME multiset of (key,value) pairs: sorted input
    is the unsorted pairs sorted by key (vals carried along). Shuffling keys
    alone would pair vals with different keys — a different dataset, whose
    per-key sums must differ.

    Series are numpy int32 (same values, same seed): with list inputs the
    list->array conversion (~15ms at 200K) dominates both paths and the ratio
    measures conversion noise, not the grouping paths (E-track: both paths
    are sub-ms single-pass now, conversion fog hid that).
    """
    import numpy as np

    a = kernel.alias
    rng = np.random.default_rng(42)
    n = 200_000
    vals = rng.integers(0, 100, size=n).astype(np.int32)
    ukeys = rng.integers(0, 1_000, size=n).astype(np.int32)
    order = np.argsort(ukeys, kind="stable")
    skeys = ukeys[order]
    svals = vals[order]

    def build(vv, keys):
        jobs = [a["ir_series"]("v", vv), a["ir_series"]("k", keys),
                a["ir_groupby"]("g", "v", "k", "sum")]
        return a["optimize"](a["compile"](jobs))["nodes"]

    def best(nodes, reps=5):
        ts = []
        for _ in range(reps):
            s = time.perf_counter()
            a["cpu_execute"](nodes)
            ts.append((time.perf_counter() - s) * 1000)
        return min(ts)

    n_s, n_u = build(svals, skeys), build(vals, ukeys)
    t_s, t_u = best(n_s), best(n_u)
    print(f"\nmicrobench ms: sorted={t_s:.2f} unsorted={t_u:.2f} ratio={t_s / t_u:.3f}")
    assert _bufs(a, [a["ir_series"]("v", svals), a["ir_series"]("k", skeys),
                     a["ir_groupby"]("g", "v", "k", "sum")])["g"] == \
        _bufs(a, [a["ir_series"]("v", vals), a["ir_series"]("k", ukeys),
                  a["ir_groupby"]("g", "v", "k", "sum")])["g"]
    assert t_s <= 1.1 * t_u, f"sorted path {t_s:.2f}ms slower than 1.1x unsorted {t_u:.2f}ms"
