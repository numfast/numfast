# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""GPU groupby golden (DELTA-6): CPU==GPU on deterministic fixtures (seed 42).

Path: Schema -> IR -> Planner.compile -> gpu_execute/evaluate(gpu) -> WGPU.
int sum/count exact; mean = same sum/count formula as ColumnCarry (exact for
int inputs); float32 within conformance-profile.toml tolerance.
Chunk-equivalence via NUMFAST_GPU_GROUPBY_CHUNK (int bit-exact).
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest

from harness import assert_float_close, load_profile

APP_DIR = str(Path(__file__).resolve().parents[2])
PROFILE = load_profile()


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


def _gpu(a, jobs):
    s = time.perf_counter()
    g = a["compile"](jobs)
    t_compile = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    bufs = a["gpu_execute"](g["nodes"])
    t_exec = (time.perf_counter() - s) * 1000
    print(f"\ngpu-groupby stages ms: compile={t_compile:.3f} execute={t_exec:.3f}")
    return bufs


def _cpu(a, jobs):
    return a["cpu_execute"](a["compile"](jobs)["nodes"])


@pytest.mark.fast
def test_gpu_groupby_sum_count_mean_int_exact(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    for N, M in ((1, 1), (5, 3), (1000, 7), (5000, 100)):
        K = rng.integers(0, M, N, dtype=np.int32)
        V = rng.integers(-(2 ** 31), 2 ** 31 - 1, N,
                         dtype=np.int64).astype(np.int32)
        for op in ("sum", "count", "mean"):
            jobs = [a["ir_series"]("v", V), a["ir_series"]("k", K),
                    a["ir_groupby"]("g", "v", "k", op)]
            want = _cpu(a, jobs)["g"]
            got = _gpu(a, jobs)["g"]
            assert list(got) == sorted(got), "GPU keys sorted"
            assert list(got) == list(want), f"N={N} M={M} op={op} keys"
            for kk in want:
                assert got[kk] == want[kk], f"N={N} M={M} op={op} key={kk}"


@pytest.mark.fast
def test_gpu_groupby_large_m_global_path_exact(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    N, M = 20000, 5000  # M > tiled cap 2048 -> global atomics path
    K = rng.integers(0, M, N, dtype=np.int32)
    V = rng.integers(-100000, 100000, N, dtype=np.int32)
    jobs = [a["ir_series"]("v", V), a["ir_series"]("k", K),
            a["ir_groupby"]("g", "v", "k", "sum")]
    assert _gpu(a, jobs)["g"] == _cpu(a, jobs)["g"]


@pytest.mark.fast
def test_gpu_groupby_float32_tolerance(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    Vf = rng.normal(0, 50, 3000).astype(np.float32)
    K = rng.integers(0, 17, 3000, dtype=np.int32)
    for op in ("sum", "mean"):
        jobs = [a["ir_series"]("v", Vf, "float32"),
                a["ir_series"]("k", K),
                a["ir_groupby"]("g", "v", "k", op)]
        want = _cpu(a, jobs)["g"]
        got = _gpu(a, jobs)["g"]
        assert list(got) == list(want)
        for kk in want:
            assert_float_close(got[kk], want[kk], PROFILE, "f32",
                               label=f"f32 groupby {op}[{kk}]")


@pytest.mark.fast
def test_gpu_groupby_multi_exact(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    N = 2000
    V1 = rng.integers(0, 100, N, dtype=np.int32)
    V2 = rng.normal(0, 5, N).astype(np.float32)
    K = rng.integers(0, 11, N, dtype=np.int32)
    jobs = [a["ir_series"]("v1", V1),
            a["ir_series"]("v2", V2, "float32"),
            a["ir_series"]("k", K),
            a["ir_groupby_multi"]("g", ["v1", "v2"], "k",
                                  {"v1": ("sum", "count", "mean"),
                                   "v2": ("sum", "mean")})]
    want = _cpu(a, jobs)["g"]
    got = _gpu(a, jobs)["g"]
    assert list(got) == list(want)
    for kk in want:
        for col in want[kk]:
            for op, x in want[kk][col].items():
                y = got[kk][col][op]
                if isinstance(x, int):
                    assert y == x, (kk, col, op)
                else:
                    assert_float_close(y, x, PROFILE, "f32",
                                       label=f"multi[{kk}][{col}][{op}]")


@pytest.mark.fast
def test_gpu_groupby_single_group_and_empty(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("v", [5, 5, 5]),
            a["ir_series"]("k", [7, 7, 7]),
            a["ir_groupby"]("g", "v", "k", "sum")]
    assert _gpu(a, jobs)["g"] == {7: 15}
    jobs = [a["ir_series"]("v", np.zeros(0, dtype=np.int32)),
            a["ir_series"]("k", np.zeros(0, dtype=np.int32)),
            a["ir_groupby"]("g", "v", "k", "sum")]
    assert _gpu(a, jobs)["g"] == {}
    assert _cpu(a, jobs)["g"] == {}


@pytest.mark.fast
def test_gpu_groupby_chunk_equivalence(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    N = 1000
    K = rng.integers(0, 37, N, dtype=np.int32)
    V = rng.integers(-(2 ** 30), 2 ** 30, N,
                     dtype=np.int64).astype(np.int32)
    jobs = [a["ir_series"]("v", V), a["ir_series"]("k", K),
            a["ir_groupby"]("g", "v", "k", "sum")]
    os.environ["NUMFAST_GPU_GROUPBY_CHUNK"] = "100"
    try:
        chunked = _gpu(a, jobs)["g"]
    finally:
        del os.environ["NUMFAST_GPU_GROUPBY_CHUNK"]
    whole = _gpu(a, jobs)["g"]
    assert chunked == whole, "op-internal chunking must be associative"


@pytest.mark.fast
def test_gpu_groupby_evaluate_runtime_path(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("v", [10, 20, 30, 40]),
            a["ir_series"]("k", [0, 1, 0, 1]),
            a["ir_groupby"]("g", "v", "k", "sum")]
    graph = a["compile"](jobs)
    res = a["evaluate"](graph, "gpu", 4)
    assert res["result"] == {0: 40, 1: 60}
    assert res["execution_info"]["actual"] == "gpu"
    assert res["execution_info"]["requested"] == "gpu"


@pytest.mark.fast
def test_gpu_groupby_explicit_errors(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    # negative keys: dense-only -> explicit CPU-only
    jobs = [a["ir_series"]("v", [1, 2, 3]),
            a["ir_series"]("k", [0, -1, 2]),
            a["ir_groupby"]("g", "v", "k", "sum")]
    with pytest.raises(ValueError, match="CPU-only|dense"):
        a["gpu_execute"](a["compile"](jobs)["nodes"])
    # float64 values: MapF64 guard
    jobs = [a["ir_series"]("v", rng.normal(0, 1, 16), "float64"),
            a["ir_series"]("k", rng.integers(0, 3, 16, dtype=np.int32)),
            a["ir_groupby"]("g", "v", "k", "sum")]
    with pytest.raises(ValueError, match="float64|CPU|float32"):
        a["gpu_execute"](a["compile"](jobs)["nodes"])
    # result='carry': CPU-only
    jobs = [a["ir_series"]("v", [1, 2, 3]),
            a["ir_series"]("k", [0, 0, 1]),
            a["ir_groupby"]("g", "v", "k", "sum", result="carry")]
    with pytest.raises(ValueError, match="CPU-only|carry"):
        a["gpu_execute"](a["compile"](jobs)["nodes"])
    # explicit gpu on a sort graph: sort is GPU-eligible (pow2 N sorts on
    # GPU; non-pow2 raises an explicit driver error, never silent fallback)
    jobs = [a["ir_series"]("v", [3, 1, 2, 0]), a["ir_sort"]("s", "v")]
    sel = a["select_backend"](a["compile"](jobs), 4)
    assert sel["gpu_eligible"] is True and sel["gpu_blockers"] == []
    jobs = [a["ir_series"]("v", [3, 1, 2]), a["ir_sort"]("s", "v")]
    with pytest.raises(ValueError, match="power-of-two|backend='cpu'"):
        a["evaluate"](a["compile"](jobs), "gpu", 3)
