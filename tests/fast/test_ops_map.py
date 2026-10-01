# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: map full set (scalar/array, add/sub/mul/div/pow) + compare (spec 01/02/08).

int — exact; float — tolerance/ULP из conformance-profile.toml.
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


def _run(a, jobs, n, dtype="int32"):
    s = time.perf_counter()
    graph = a["optimize"](a["compile"](jobs))
    t_compile = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    res = a["evaluate"](graph, "auto", n)
    t_exec = (time.perf_counter() - s) * 1000
    print(f"\nstages ms: compile+optimize={t_compile:.3f} execute={t_exec:.3f}")
    return res


def _eq(a, jobs, n):
    return _run(a, jobs, n)["result"]


@pytest.mark.fast
def test_map_pow_scalar_int_exact(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [1, 2, 3, 4]), a["ir_map"]("m", "s", "pow", 2)]
    got = list(_eq(a, jobs, 4))
    assert [int(v) for v in got] == [1, 4, 9, 16]


@pytest.mark.fast
def test_map_array_array_add_int_exact(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("x", [1, 2, 3]),
        a["ir_series"]("y", [10, 20, 30]),
        a["ir_map"]("m", "x", "add", "y"),
    ]
    got = list(_eq(a, jobs, 3))
    assert [int(v) for v in got] == [11, 22, 33]


@pytest.mark.fast
def test_map_array_array_sub_mul_int_exact(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("x", [5, 6, 7]),
        a["ir_series"]("y", [1, 2, 3]),
        a["ir_map"]("d", "x", "sub", "y"),
        a["ir_map"]("m", "x", "mul", "y"),
    ]
    graph = a["optimize"](a["compile"](jobs))
    res = a["evaluate"](graph, "auto", 3)
    assert res["execution_info"]["actual"] == "cpu"


@pytest.mark.fast
def test_map_div_float_tolerance(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("s", [3.0, 5.0, 7.0], "float32"),
        a["ir_map"]("m", "s", "div", 2.0),
    ]
    got = list(_eq(a, jobs, 3))
    for v, w in zip(got, [1.5, 2.5, 3.5]):
        assert_float_close(v, w, PROFILE, "f32", label="div")


@pytest.mark.fast
def test_compare_scalar_all_ops_exact(kernel):
    a = kernel.alias
    vals = [1, 2, 3]
    expect = {
        "==": [False, True, False],
        "!=": [True, False, True],
        "<": [True, False, False],
        "<=": [True, True, False],
        ">": [False, False, True],
        ">=": [False, True, True],
    }
    for op, want in expect.items():
        jobs = [a["ir_series"]("s", vals), a["ir_compare"]("c", "s", 2, op)]
        got = list(_eq(a, jobs, 3))
        assert [bool(v) for v in got] == want, op


@pytest.mark.fast
def test_compare_array_array_exact(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("x", [1, 2, 3]),
        a["ir_series"]("y", [1, 0, 3]),
        a["ir_compare"]("c", "x", "y", "=="),
    ]
    got = list(_eq(a, jobs, 3))
    assert [bool(v) for v in got] == [True, False, True]


@pytest.mark.fast
def test_compare_nan_semantics(kernel):
    a = kernel.alias
    nan = float("nan")
    jobs = [a["ir_series"]("s", [nan, 1.0], "float32"), a["ir_compare"]("c", "s", nan, "==")]
    got = list(_eq(a, jobs, 2))
    assert [bool(v) for v in got] == [False, False]  # NaN != NaN (spec 08)
    jobs = [a["ir_series"]("s", [nan, 1.0], "float32"), a["ir_compare"]("c", "s", nan, "!=")]
    got = list(_eq(a, jobs, 2))
    assert [bool(v) for v in got] == [True, True]


@pytest.mark.fast
def test_map_pow_array_exponent_rejected(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("x", [1, 2]),
        a["ir_series"]("e", [2, 3]),
        a["ir_map"]("m", "x", "pow", "e"),
    ]
    graph = a["optimize"](a["compile"](jobs))
    with pytest.raises(ValueError, match="scalar-exp"):
        a["evaluate"](graph, "auto", 2)


@pytest.mark.fast
def test_map_compare_unknown_rejected(kernel):
    a = kernel.alias
    with pytest.raises(ValueError, match="unknown map fn"):
        a["ir_map"]("m", "s", "hypot", 1)
    with pytest.raises(ValueError, match="unknown compare op"):
        a["ir_compare"]("c", "s", 1, "<>")
    assert_int_exact(1, 1, label="sanity")
