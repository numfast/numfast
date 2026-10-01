# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""GPU golden: CPU==GPU exact on deterministic fixtures (seed 42).

Path: Schema -> IR -> Planner.compile -> gpu_execute -> WGPU -> WGSL.
Stage breakdown (ms) printed per run. filter hybrid (CPU-prefix+GPU-gather),
reduce hybrid (GPU-partials+CPU-int64/float64-final); groupby/sort/encode
raise explicit CPU-only errors (spec 06, no silent fallback).
"""

import operator as opm
import time
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])
OPS = ["==", "!=", "<", "<=", ">", ">="]
FN = {"==": opm.eq, "!=": opm.ne, "<": opm.lt,
      "<=": opm.le, ">": opm.gt, ">=": opm.ge}


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


def _stages(a, jobs):
    t = {}
    s = time.perf_counter()
    g = a["compile"](jobs)
    t["compile"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    bufs = a["gpu_execute"](g["nodes"])
    t["execute"] = (time.perf_counter() - s) * 1000
    print("\ngpu stages ms: " + " ".join(f"{k}={v:.3f}" for k, v in t.items()))
    return bufs


@pytest.mark.fast
def test_gpu_pack_exact(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    k1 = rng.integers(0, 50, 1024, dtype=np.int32)
    k2 = rng.integers(0, 100, 1024, dtype=np.int32)
    jobs = [a["ir_series"]("k1", k1), a["ir_series"]("k2", k2),
            {"op": "pack_keys", "inputs": ["k1", "k2"],
             "params": {"mode": "pack"}, "out": "c"}]
    bufs = _stages(a, jobs)
    m2 = int(k2.max()) + 1
    ref = k1.astype(np.int64) * m2 + k2.astype(np.int64)
    assert (np.asarray(bufs["c"], dtype=np.int64) == ref).all()


@pytest.mark.fast
def test_gpu_compare_exact(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    A = rng.integers(-5, 6, 1025, dtype=np.int32)
    B = rng.integers(-5, 6, 1025, dtype=np.int32)
    for o in OPS:
        jobs = [a["ir_series"]("x", A), a["ir_series"]("y", B),
                a["ir_compare"]("m", "x", "y", o)]
        got = np.asarray(_stages(a, jobs)["m"], dtype=np.int32)
        assert (got == FN[o](A, B).astype(np.int32)).all(), o
        jobs = [a["ir_series"]("x", A), a["ir_compare"]("m", "x", 2, o)]
        got = np.asarray(_stages(a, jobs)["m"], dtype=np.int32)
        assert (got == FN[o](A, 2).astype(np.int32)).all(), o
    Af = rng.normal(0, 5, 1025).astype(np.float32)
    jobs = [a["ir_series"]("x", Af, "float32"),
            a["ir_compare"]("m", "x", 0.5, ">")]
    got = np.asarray(_stages(a, jobs)["m"], dtype=np.int32)
    assert (got == (Af > 0.5).astype(np.int32)).all()


@pytest.mark.fast
def test_gpu_mask_exact(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    A = rng.integers(0, 2, 1025).astype(bool)
    B = rng.integers(0, 2, 1025).astype(bool)
    for o, ref in (("and", A & B), ("or", A | B)):
        jobs = [a["ir_series"]("x", A, "bool"), a["ir_series"]("y", B, "bool"),
                a["ir_mask"]("m", "x", "y", o)]
        got = np.asarray(_stages(a, jobs)["m"], dtype=np.int32)
        assert (got == ref.astype(np.int32)).all(), o
    jobs = [a["ir_series"]("x", A, "bool"), a["ir_mask"]("m", "x", op="not")]
    got = np.asarray(_stages(a, jobs)["m"], dtype=np.int32)
    assert (got == (~A).astype(np.int32)).all()


@pytest.mark.fast
def test_gpu_filter_gather_exact(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    V = rng.integers(0, 100, 1025, dtype=np.int32)
    jobs = [a["ir_series"]("v", V), a["ir_compare"]("m", "v", 50, ">"),
            a["ir_filter"]("f", "v", "m")]
    got = np.asarray(_stages(a, jobs)["f"])
    assert (got == V[V > 50]).all()
    jobs = [a["ir_series"]("v", V), a["ir_compare"]("m", "v", -1, ">"),
            a["ir_filter"]("f", "v", "m")]
    assert np.asarray(_stages(a, jobs)["f"]).size == V.size
    Ix = np.array([9, 0, 5], dtype=np.int32)
    jobs = [a["ir_series"]("v", np.arange(10, dtype=np.int32)),
            a["ir_series"]("ix", Ix), a["ir_gather"]("o", "v", "ix")]
    assert (np.asarray(_stages(a, jobs)["o"]) == np.arange(10)[Ix]).all()


@pytest.mark.fast
def test_gpu_sum_count_mean_exact(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    V = rng.integers(0, 100, 1025, dtype=np.int32)
    jobs = [a["ir_series"]("v", V), a["ir_reduce"]("s", "v", "sum")]
    assert int(_stages(a, jobs)["s"]) == int(V.astype(np.int64).sum())
    jobs = [a["ir_series"]("v", V), a["ir_reduce"]("c", "v", "count")]
    assert int(_stages(a, jobs)["c"]) == V.size
    Vf = rng.normal(0, 5, 1025).astype(np.float32)
    jobs = [a["ir_series"]("v", Vf, "float32"),
            a["ir_reduce"]("m", "v", "mean")]
    got = float(_stages(a, jobs)["m"])
    assert abs(got - float(Vf.astype(np.float64).mean())) < 1e-4


@pytest.mark.fast
def test_gpu_cpu_only_raise_explicit(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    # sort is GPU-supported for single int32/finite-f32 keys at power-of-two N
    # (see test_ops_gpu_sort.py); non-pow2/NaN stay CPU-only with explicit
    # errors. encode_pattern stays CPU-only.
    V = rng.integers(0, 10, 100, dtype=np.int32)
    jobs = [a["ir_series"]("v", V), a["ir_sort"]("s", "v")]
    with pytest.raises(ValueError, match="CPU-only|cpu"):
        a["gpu_execute"](a["compile"](jobs)["nodes"])
    jobs = [a["ir_series"]("v", V[:64]), a["ir_sort"]("s", "v")]
    got = np.asarray(a["gpu_execute"](a["compile"](jobs)["nodes"])["s"])
    assert (got == np.argsort(V[:64], kind="stable")).all()
    Vf64 = rng.normal(0, 1, 64)
    jobs = [a["ir_series"]("v", Vf64, "float64"),
            a["ir_reduce"]("s", "v", "sum")]
    with pytest.raises(ValueError, match="float64|CPU"):
        a["gpu_execute"](a["compile"](jobs)["nodes"])
