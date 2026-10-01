# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""GPU scaled-float64 golden (SPEC-DELTA-7): CPU==GPU exact via int32 ticks.

Path: Schema.column_schema -> ir_series(scale/offset) -> compile ->
gpu_execute (WGSL integer) vs cpu_execute (f64). Seed 42.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from harness import load_profile  # noqa: F401 -- profile parity hook

APP_DIR = str(Path(__file__).resolve().parents[2])


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
    print(f"\ngpu-scaled stages ms: compile={t_compile:.3f} execute={t_exec:.3f}")
    return bufs


def _cpu(a, jobs):
    return a["cpu_execute"](a["compile"](jobs)["nodes"])


def _scaled_jobs(a, v, k, ops=("sum", "mean")):
    col = a["column_schema"]("v3", "float64", 1e-6, 0)
    assert col["scale"] == 1e-6 and col["physical"] == "int32"
    return [a["ir_series"]("v", np.ascontiguousarray(v, dtype=np.float64),
                           "float64", scale=col["scale"],
                           offset=col["offset"]),
            a["ir_series"]("k", np.ascontiguousarray(k, dtype=np.int32)),
            a["ir_groupby_multi"]("g", "v", "k", ops)]


@pytest.mark.fast
def test_gpu_scaled_sum_mean_exact(kernel):
    # Contract (SPEC-DELTA-7): scaled accumulator exact (tick-level ==),
    # logical display diff 0.0 (round 6). Raw f64 accumulation order differs
    # (CPU sequential f64 vs GPU exact-int-then-scale), so last-ULP diffs
    # are honest — tick/display equality is the conformance claim.
    a = kernel.alias
    rng = np.random.default_rng(42)
    for N, M in ((10, 3), (1000, 7), (5000, 50)):
        K = rng.integers(0, M, N, dtype=np.int32)
        V = np.rint(rng.uniform(0, 100, N) * 1e6) / 1e6
        jobs = _scaled_jobs(a, V, K)
        want = _cpu(a, jobs)["g"]
        got = _gpu(a, jobs)["g"]
        assert list(got) == list(want)
        ticks = np.rint(V * 1e6).astype(np.int64)
        ref = {}
        for kk, t in zip(K.tolist(), ticks.tolist()):
            ref[kk] = ref.get(kk, 0) + t
        md = 0.0
        for kk in want:
            assert int(round(float(got[kk]["sum"]) * 1e6)) == ref[kk]
            for op in ("sum", "mean"):
                assert round(float(got[kk][op]), 6) == round(
                    float(want[kk][op]), 6), (N, M, kk, op)
                md = max(md, abs(float(got[kk][op]) - float(want[kk][op])))
        print(f"\nscaled N={N} M={M}: tick-exact ok, display diff 0.0, "
              f"max_ulp_diff={md:.3e}")


@pytest.mark.fast
def test_gpu_scaled_multi_mixed_offsets(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    N, M = 2000, 11
    K = rng.integers(0, M, N, dtype=np.int32)
    V1 = rng.integers(0, 100, N, dtype=np.int32)
    V3 = np.rint(rng.uniform(0, 100, N) * 1e6) / 1e6
    col = a["column_schema"]("v3", "float64", 1e-3, 5.0)
    jobs = [a["ir_series"]("v1", V1),
            a["ir_series"]("v3", np.ascontiguousarray(V3), "float64",
                           scale=col["scale"], offset=col["offset"]),
            a["ir_series"]("k", K),
            a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                  {"v1": ("sum", "count", "mean"),
                                   "v3": ("sum", "mean")})]
    want = _cpu(a, jobs)["g"]
    assert set(want) == set(range(M)) or len(want) == len(set(K.tolist()))
    # CPU has no scale semantics: compare GPU scaled math against host ticks.
    ticks = np.rint((V3 - 5.0) / 1e-3).astype(np.int64)
    ref_sum, ref_cnt = {}, {}
    for kk, t in zip(K.tolist(), ticks.tolist()):
        ref_sum[kk] = ref_sum.get(kk, 0) + t
        ref_cnt[kk] = ref_cnt.get(kk, 0) + 1
    got = _gpu(a, jobs)["g"]
    assert list(got) == sorted(got)
    for kk in got:
        exp_sum = ref_sum[kk] * 1e-3 + 5.0 * ref_cnt[kk]
        exp_mean = ref_sum[kk] * 1e-3 / ref_cnt[kk] + 5.0
        assert got[kk]["v3"]["sum"] == exp_sum, (kk, "sum")
        assert got[kk]["v3"]["mean"] == exp_mean, (kk, "mean")
        assert got[kk]["v1"] == want[kk]["v1"], (kk, "v1")


@pytest.mark.fast
def test_gpu_scaled_chunk_equiv(kernel):
    a = kernel.alias
    import os

    rng = np.random.default_rng(42)
    N = 1000
    K = rng.integers(0, 37, N, dtype=np.int32)
    V = np.rint(rng.uniform(0, 50, N) * 1e6) / 1e6
    jobs = _scaled_jobs(a, V, K, ("sum",))
    os.environ["NUMFAST_GPU_GROUPBY_CHUNK"] = "100"
    try:
        chunked = _gpu(a, jobs)["g"]
    finally:
        del os.environ["NUMFAST_GPU_GROUPBY_CHUNK"]
    assert _gpu(a, jobs)["g"] == chunked


@pytest.mark.fast
def test_gpu_global_path_multi_int_exact(kernel):
    # DELTA-6 latent fix: global path (M > 2048) kept one shared accum for
    # all int cols (multi-col sums merged). Per-column accums now.
    a = kernel.alias
    rng = np.random.default_rng(42)
    N, M = 20000, 5000
    K = rng.integers(0, M, N, dtype=np.int32)
    V1 = rng.integers(-100000, 100000, N, dtype=np.int32)
    V2 = rng.integers(-100000, 100000, N, dtype=np.int32)
    jobs = [a["ir_series"]("v1", V1), a["ir_series"]("v2", V2),
            a["ir_series"]("k", K),
            a["ir_groupby_multi"]("g", ["v1", "v2"], "k",
                                  {"v1": ("sum", "count", "mean"),
                                   "v2": ("sum", "mean")})]
    want = _cpu(a, jobs)["g"]
    got = _gpu(a, jobs)["g"]
    assert list(got) == list(want)
    for kk in want:
        assert got[kk] == want[kk], kk


@pytest.mark.fast
def test_gpu_scaled_guards(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    K = rng.integers(0, 3, 16, dtype=np.int32)
    # unscaled f64 still guarded
    jobs = [a["ir_series"]("v", rng.normal(0, 1, 16), "float64"),
            a["ir_series"]("k", K),
            a["ir_groupby"]("g", "v", "k", "sum")]
    with pytest.raises(ValueError, match="float64|CPU|float32"):
        a["gpu_execute"](a["compile"](jobs)["nodes"])
    # non-finite scaled guarded
    bad = np.array([1.0, np.inf, 3.0])
    jobs = [a["ir_series"]("v", bad, "float64", scale=1e-6, offset=0),
            a["ir_series"]("k", np.array([0, 1, 2], dtype=np.int32)),
            a["ir_groupby"]("g", "v", "k", "sum")]
    with pytest.raises(ValueError, match="non-finite|finite"):
        a["gpu_execute"](a["compile"](jobs)["nodes"])
    # out-of-int32 physical guarded
    big = np.array([3e4, -3e4, 1.0])  # *1e6 -> 3e10 > int32
    jobs = [a["ir_series"]("v", big, "float64", scale=1e-6, offset=0),
            a["ir_series"]("k", np.array([0, 1, 2], dtype=np.int32)),
            a["ir_groupby"]("g", "v", "k", "sum")]
    with pytest.raises(ValueError, match="int32"):
        a["gpu_execute"](a["compile"](jobs)["nodes"])
    # scaled keys rejected
    jobs = [a["ir_series"]("v", [1, 2, 3]),
            a["ir_series"]("k", np.array([0.5, 1.5, 2.5]), "float64",
                           scale=1e-6, offset=0),
            a["ir_groupby"]("g", "v", "k", "sum")]
    with pytest.raises(ValueError, match="keys|scaled"):
        a["gpu_execute"](a["compile"](jobs)["nodes"])
    # scaled outside groupby rejected
    col_jobs = [a["ir_series"]("v", np.array([1.5, 2.5]), "float64",
                               scale=1e-6, offset=0),
                a["ir_reduce"]("r", "v", "sum")]
    with pytest.raises(ValueError, match="groupby only"):
        a["gpu_execute"](a["compile"](col_jobs)["nodes"])
    # scale on non-f64 rejected at IR time
    with pytest.raises(ValueError, match="float64"):
        a["ir_series"]("v", [1, 2, 3], "int32", scale=1e-6)
