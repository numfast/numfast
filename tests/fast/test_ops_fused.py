# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: fused multi-aggregate groupby_multi (DELTA-1).

One scatter, state (sum,count[,mean]); mean derived from same state.
int — exact; float — tolerance/ULP. Microbench: Q5 <= 1.3x Q1 (seed 42).
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
def test_groupby_multi_int_exact(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("v", [10, 20, 30, 40]),
        a["ir_series"]("k", [0, 1, 0, 1]),
        a["ir_groupby_multi"]("g", "v", "k", ("sum", "count", "mean")),
    ]
    got = _bufs(a, jobs)["g"]
    assert got[0] == {"sum": 40, "count": 2, "mean": 20.0}
    assert got[1] == {"sum": 60, "count": 2, "mean": 30.0}
    assert list(got) == [0, 1]


@pytest.mark.fast
def test_groupby_multi_subset_ops(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("v", [1, 2, 3]),
        a["ir_series"]("k", [5, 5, 6]),
        a["ir_groupby_multi"]("g", "v", "k", ("count",)),
    ]
    assert _bufs(a, jobs)["g"] == {5: {"count": 2}, 6: {"count": 1}}


@pytest.mark.fast
def test_groupby_multi_float_tolerance(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("v", [1.5, 2.5, 3.0, 4.0], "float32"),
        a["ir_series"]("k", [0, 0, 1, 1]),
        a["ir_groupby_multi"]("g", "v", "k", ("sum", "count", "mean")),
    ]
    got = _bufs(a, jobs)["g"]
    assert_float_close(got[0]["sum"], 4.0, PROFILE, "f32", label="fused sum")
    assert_int_exact(got[0]["count"], 2, label="fused count")
    assert_float_close(got[0]["mean"], 2.0, PROFILE, "f32", label="fused mean")
    assert_float_close(got[1]["mean"], 3.5, PROFILE, "f32", label="fused mean g1")


@pytest.mark.fast
def test_groupby_multi_equals_separate_ops(kernel):
    a = kernel.alias
    vals = [7, 1, 9, 3, 5, 5]
    keys = [2, 0, 2, 0, 1, 1]
    fused = _bufs(a, [
        a["ir_series"]("v", vals), a["ir_series"]("k", keys),
        a["ir_groupby_multi"]("g", "v", "k", ("sum", "count", "mean")),
    ])["g"]
    sep = _bufs(a, [
        a["ir_series"]("v", vals), a["ir_series"]("k", keys),
        a["ir_groupby"]("s", "v", "k", "sum"),
        a["ir_groupby"]("c", "v", "k", "count"),
        a["ir_groupby"]("m", "v", "k", "mean"),
    ])
    assert fused.keys() == sep["s"].keys()
    for key in fused:
        assert_int_exact(fused[key]["sum"], sep["s"][key], label=f"sum[{key}]")
        assert_int_exact(fused[key]["count"], sep["c"][key], label=f"count[{key}]")
        assert_float_close(fused[key]["mean"], sep["m"][key], PROFILE, "f64", label=f"mean[{key}]")


@pytest.mark.fast
def test_groupby_multi_bad_ops_rejected(kernel):
    a = kernel.alias
    with pytest.raises(ValueError, match="non-empty subset"):
        a["ir_groupby_multi"]("g", "v", "k", ("median",))
    with pytest.raises(ValueError, match="non-empty subset"):
        a["ir_groupby_multi"]("g", "v", "k", ())


@pytest.mark.fast
def test_microbench_fused_q5_vs_q1(kernel):
    """Q5 (fused sum+count+mean) <= 1.3x Q1 (single sum). Seed 42."""
    import numpy as np

    a = kernel.alias
    rng = np.random.default_rng(42)
    n, ngroups = 200_000, 1_000
    vals = rng.integers(0, 100, size=n).tolist()
    keys = rng.integers(0, ngroups, size=n).tolist()

    def build(extra):
        jobs = [a["ir_series"]("v", vals), a["ir_series"]("k", keys)]
        jobs += extra
        return a["optimize"](a["compile"](jobs))["nodes"]

    n_q1 = build([a["ir_groupby"]("g", "v", "k", "sum")])
    n_sep = build([a["ir_groupby"]("s", "v", "k", "sum"),
                   a["ir_groupby"]("c", "v", "k", "count"),
                   a["ir_groupby"]("m", "v", "k", "mean")])
    n_fused = build([a["ir_groupby_multi"]("g", "v", "k", ("sum", "count", "mean"))])

    def best(nodes, reps=5):
        ts = []
        for _ in range(reps):
            s = time.perf_counter()
            a["cpu_execute"](nodes)
            ts.append((time.perf_counter() - s) * 1000)
        return min(ts)

    t_q1, t_sep, t_fused = best(n_q1), best(n_sep), best(n_fused)
    print(f"\nmicrobench ms: Q1={t_q1:.2f} SEP3={t_sep:.2f} Q5-fused={t_fused:.2f} "
          f"ratio-fused/Q1={t_fused / t_q1:.3f}")
    assert t_fused <= 1.3 * t_q1, f"Q5 {t_fused:.2f}ms > 1.3x Q1 {t_q1:.2f}ms"
