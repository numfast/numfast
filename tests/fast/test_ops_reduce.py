# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: reduce full set (sum/count/mean/min/max skipna + var/std ddof, spec 08).

int — exact; float — tolerance/ULP из conformance-profile.toml.
NaN без skipna -> NaN + warning; со skipna -> исключение NaN.
"""

import math
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


def _run(a, jobs, n):
    s = time.perf_counter()
    graph = a["optimize"](a["compile"](jobs))
    t_compile = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    res = a["evaluate"](graph, "auto", n)
    t_exec = (time.perf_counter() - s) * 1000
    print(f"\nstages ms: compile+optimize={t_compile:.3f} execute={t_exec:.3f}")
    return res


@pytest.mark.fast
def test_reduce_var_std_ddof_exact(kernel):
    a = kernel.alias
    vals = [1, 2, 3, 4, 5]  # var ddof=0 -> 2.0, ddof=1 -> 2.5 (hand-derived)
    for op, ddof, want in (("var", 0, 2.0), ("var", 1, 2.5), ("std", 0, 2.0**0.5), ("std", 1, 2.5**0.5)):
        jobs = [a["ir_series"]("s", vals, "float32"), a["ir_reduce"]("r", "s", op, ddof=ddof)]
        got = _run(a, jobs, 5)["result"]
        assert_float_close(got, want, PROFILE, "f64", label=f"{op} ddof={ddof}")


@pytest.mark.fast
def test_reduce_var_int_input_tolerance(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [2, 4, 4, 4, 5, 5, 7, 9]), a["ir_reduce"]("r", "s", "var")]
    got = _run(a, jobs, 8)["result"]
    assert_float_close(got, 4.0, PROFILE, "f64", label="var int")  # hand: mean 5, var 4


@pytest.mark.fast
def test_reduce_nan_propagates_with_warning(kernel):
    a = kernel.alias
    nan = float("nan")
    jobs = [a["ir_series"]("s", [1.0, nan, 3.0], "float32"), a["ir_reduce"]("r", "s", "sum")]
    with pytest.warns(UserWarning, match="skipna=False"):
        got = _run(a, jobs, 3)["result"]
    assert math.isnan(float(got))


@pytest.mark.fast
def test_reduce_skipna_sum_mean(kernel):
    a = kernel.alias
    nan = float("nan")
    jobs = [
        a["ir_series"]("s", [1.0, nan, 3.0], "float32"),
        a["ir_reduce"]("rs", "s", "sum", skipna=True),
        a["ir_reduce"]("rm", "s", "mean", skipna=True),
    ]
    graph = a["optimize"](a["compile"](jobs))
    bufs = a["cpu_execute"](graph["nodes"])
    assert_float_close(bufs["rs"], 4.0, PROFILE, "f32", label="nansum")
    assert_float_close(bufs["rm"], 2.0, PROFILE, "f32", label="nanmean")


@pytest.mark.fast
def test_reduce_skipna_min_max_count(kernel):
    a = kernel.alias
    nan = float("nan")
    jobs = [
        a["ir_series"]("s", [1.0, nan, 3.0], "float32"),
        a["ir_reduce"]("mn", "s", "min", skipna=True),
        a["ir_reduce"]("mx", "s", "max", skipna=True),
        a["ir_reduce"]("c1", "s", "count", skipna=True),
        a["ir_reduce"]("c0", "s", "count"),
    ]
    graph = a["optimize"](a["compile"](jobs))
    bufs = a["cpu_execute"](graph["nodes"])
    assert_int_exact(bufs["mn"], 1, label="nanmin")
    assert_int_exact(bufs["mx"], 3, label="nanmax")
    assert_int_exact(bufs["c1"], 2, label="count skipna")
    assert_int_exact(bufs["c0"], 3, label="count size")


@pytest.mark.fast
def test_reduce_var_std_skipna(kernel):
    a = kernel.alias
    nan = float("nan")
    jobs = [
        a["ir_series"]("s", [1.0, nan, 3.0], "float32"),
        a["ir_reduce"]("v", "s", "var", skipna=True),
        a["ir_reduce"]("d", "s", "std", skipna=True),
    ]
    graph = a["optimize"](a["compile"](jobs))
    bufs = a["cpu_execute"](graph["nodes"])
    assert_float_close(bufs["v"], 1.0, PROFILE, "f64", label="nanvar")  # var([1,3]) ddof=0
    assert_float_close(bufs["d"], 1.0, PROFILE, "f64", label="nanstd")


@pytest.mark.fast
def test_reduce_unknown_rejected(kernel):
    a = kernel.alias
    with pytest.raises(ValueError, match="unknown reduce op"):
        a["ir_reduce"]("r", "s", "median")
