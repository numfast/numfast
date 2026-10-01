# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: groupby sum/count/mean, mean = sum+count composition (spec 02/04).

int — exact; float — tolerance/ULP из conformance-profile.toml.
chunkable=false v0.2 (spec 04); GPU dense path covered separately in
test_ops_gpu_groupby.py (DELTA-6).
"""

import time
from pathlib import Path

import pytest

from harness import assert_float_close, load_profile

APP_DIR = str(Path(__file__).resolve().parents[2])
PROFILE = load_profile()


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


def _bufs(a, jobs):
    s = time.perf_counter()
    graph = a["optimize"](a["compile"](jobs))
    t_compile = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    bufs = a["cpu_execute"](graph["nodes"])
    t_exec = (time.perf_counter() - s) * 1000
    print(f"\nstages ms: compile+optimize={t_compile:.3f} execute={t_exec:.3f}")
    return bufs


@pytest.mark.fast
def test_groupby_sum_int_exact(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("v", [10, 20, 30, 40]),
        a["ir_series"]("k", [0, 1, 0, 1]),
        a["ir_groupby"]("g", "v", "k", "sum"),
    ]
    assert _bufs(a, jobs)["g"] == {0: 40, 1: 60}


@pytest.mark.fast
def test_groupby_count_int_exact(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("v", [10, 20, 30, 40]),
        a["ir_series"]("k", [0, 1, 0, 1]),
        a["ir_groupby"]("g", "v", "k", "count"),
    ]
    assert _bufs(a, jobs)["g"] == {0: 2, 1: 2}


@pytest.mark.fast
def test_groupby_mean_is_sum_over_count(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("v", [1, 2, 3, 4], "float32"),
        a["ir_series"]("k", [0, 1, 0, 1]),
        a["ir_groupby"]("m", "v", "k", "mean"),
        a["ir_groupby"]("s", "v", "k", "sum"),
        a["ir_groupby"]("c", "v", "k", "count"),
    ]
    bufs = _bufs(a, jobs)
    assert bufs["m"].keys() == bufs["s"].keys() == bufs["c"].keys()
    for key in bufs["m"]:
        assert_float_close(bufs["m"][key], bufs["s"][key] / bufs["c"][key], PROFILE, "f32", label=f"mean[{key}]")
    for key, want in ((0, 2.0), (1, 3.0)):
        assert_float_close(bufs["m"][key], want, PROFILE, "f32", label=f"mean=={want}")


@pytest.mark.fast
def test_groupby_unsorted_keys_sorted_output(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("v", [1, 10, 3, 30, 100]),
        a["ir_series"]("k", [2, 0, 2, 0, 1]),
        a["ir_groupby"]("g", "v", "k", "sum"),
    ]
    got = _bufs(a, jobs)["g"]
    assert got == {0: 40, 1: 100, 2: 4}
    assert list(got) == [0, 1, 2]


@pytest.mark.fast
def test_groupby_single_group(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("v", [5, 5, 5]),
        a["ir_series"]("k", [7, 7, 7]),
        a["ir_groupby"]("g", "v", "k", "mean"),
    ]
    got = _bufs(a, jobs)["g"]
    assert_float_close(got[7], 5.0, PROFILE, "f32", label="single-group mean")


@pytest.mark.fast
def test_groupby_size_mismatch_rejected(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("v", [1, 2, 3]),
        a["ir_series"]("k", [0, 1]),
        a["ir_groupby"]("g", "v", "k", "sum"),
    ]
    graph = a["optimize"](a["compile"](jobs))
    with pytest.raises(ValueError, match="size mismatch"):
        a["cpu_execute"](graph["nodes"])


@pytest.mark.fast
def test_groupby_unknown_op_rejected(kernel):
    a = kernel.alias
    with pytest.raises(ValueError, match="unknown groupby op"):
        a["ir_groupby"]("g", "v", "k", "median")
