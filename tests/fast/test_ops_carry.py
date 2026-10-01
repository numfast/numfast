# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Carry result-path: ColumnCarry primary, dict explicit compat (parity + stages).

Default result stays dict (frozen shape); result='carry' returns ColumnCarry
(ukeys/counts/sums, mean derived). Compat materialization is exact in all
shapes, serial or threaded.
"""

import time
from pathlib import Path

import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


def _bufs(a, jobs):
    graph = a["optimize"](a["compile"](jobs))
    return a["cpu_execute"](graph["nodes"])


def _carry_of(bufs, out):
    c = bufs[out + "#carry"]
    for attr in ("ukeys", "counts", "sums", "means", "to_dict_flat",
                 "to_dict_single", "to_dict_multi", "ngroups"):
        assert hasattr(c, attr), attr
    return c


@pytest.mark.fast
def test_carry_flat_parity_dict(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("v", [10, 20, 30, 40]),
            a["ir_series"]("k", [0, 1, 0, 1])]
    d = _bufs(a, jobs + [a["ir_groupby"]("g", "v", "k", "sum")])["g"]
    b = _bufs(a, jobs + [a["ir_groupby"]("g", "v", "k", "sum", result="carry")])
    c = _carry_of(b, "g")
    assert b["g"] is c
    assert list(c.ukeys) == [0, 1] and list(c.counts) == [2, 2]
    assert c.to_dict_flat("sum") == d
    assert c.to_dict_flat("count") == {0: 2, 1: 2}
    assert c.to_dict_flat("mean") == {0: 20.0, 1: 30.0}


@pytest.mark.fast
def test_carry_mean_derived_no_rescan(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("v", [1, 2, 3, 4], "float32"),
            a["ir_series"]("k", [0, 1, 0, 1]),
            a["ir_groupby_multi"]("g", "v", "k", ("sum", "count", "mean"),
                                  result="carry")]
    b = _bufs(a, jobs)
    c = _carry_of(b, "g")
    m1, m2 = c.means("v"), c.means("v")
    assert m1 is m2  # derived once, cached
    assert [float(x) for x in m1] == [2.0, 3.0]
    d = _bufs(a, [a["ir_series"]("v", [1, 2, 3, 4], "float32"),
                  a["ir_series"]("k", [0, 1, 0, 1]),
                  a["ir_groupby_multi"]("g", "v", "k",
                                        ("sum", "count", "mean"))])["g"]
    assert c.to_dict_single("v", ("sum", "count", "mean")) == d


@pytest.mark.fast
def test_carry_multi_threads_parity(kernel):
    a = kernel.alias
    n = 20_000
    import numpy as np  # noqa: PLC0415 -- deterministic data only

    rng = np.random.default_rng(42)
    v1 = rng.integers(0, 100, size=n).astype(np.int32)
    v3 = (rng.standard_normal(n)).astype(np.float64)
    kk = rng.integers(0, 5_000, size=n).astype(np.int32)
    mk = lambda res: [a["ir_series"]("v1", v1), a["ir_series"]("v3", v3, "float64"),  # noqa: E731
                      a["ir_series"]("k", kk),
                      a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                            {"v1": ("sum",), "v3": ("mean",)},
                                            result=res)]
    c = _carry_of(_bufs(a, mk("carry")), "g")
    d = _bufs(a, [a["ir_series"]("v1", v1), a["ir_series"]("v3", v3, "float64"),
                  a["ir_series"]("k", kk),
                  a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                        {"v1": ("sum",), "v3": ("mean",)})])["g"]
    ops = {"v1": ("sum",), "v3": ("mean",)}
    assert c.to_dict_multi(["v1", "v3"], ops) == d
    assert c.to_dict_multi(["v1", "v3"], ops, threads=4) == d
    s = time.perf_counter()
    c.to_dict_multi(["v1", "v3"], ops, threads=1)
    t1 = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    c.to_dict_multi(["v1", "v3"], ops, threads=4)
    t4 = (time.perf_counter() - s) * 1000
    print(f"\ncompat threads=1 {t1:.1f}ms vs threads=4 {t4:.1f}ms")
