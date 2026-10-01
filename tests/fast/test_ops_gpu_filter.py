# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""GPU filter/mask-scan golden (SPEC-DELTA-8): block-partials compact.

Predicate eval reuses compare (no new predicate code); compact = GPU
per-workgroup counts + host W-prefix + workgroup-local scan scatter.
Gates: exact vs numpy boolean-mask (1000 random, seed 42), edges
(empty/all-true/all-false), selectivity ladder 0/1/10/50/90/100%,
dtypes int32/float32, 3VL validity, Planner plan_filter + gpu_eligible.
"""

import time
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


def _gpu(a, jobs):
    s = time.perf_counter()
    bufs = a["gpu_execute"](a["compile"](jobs)["nodes"])
    dt = (time.perf_counter() - s) * 1000
    print(f"\ngpu filter stages ms: execute={dt:.3f}")
    return bufs


@pytest.mark.fast
def test_gpu_filter_matches_numpy_ladder(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    V = rng.integers(0, 1000, 4096, dtype=np.int32)
    for sel, thr in ((0, 1001), (1, 999), (10, 900), (50, 500),
                     (90, 100), (100, -1)):
        jobs = [a["ir_series"]("v", V), a["ir_compare"]("m", "v", thr, ">"),
                a["ir_filter"]("f", "v", "m")]
        got = np.asarray(_gpu(a, jobs)["f"])
        ref = V[V > thr]
        assert got.shape == ref.shape and (got == ref).all(), (sel, thr)


@pytest.mark.fast
def test_gpu_filter_1000_random_exact(kernel):
    """1000 random cases vs numpy boolean-mask (seed 42, small N)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "nfgpu", str(Path(APP_DIR) / "src" / "Drivers" / "GPU" / "_lib"
                     / "gpu.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    rng = np.random.default_rng(42)
    bad = 0
    for t in range(1000):
        n = int(rng.integers(0, 600))
        kind = t % 3
        if kind == 0:
            V = rng.integers(-50, 50, n, dtype=np.int32)
            dt = "int32"
        elif kind == 1:
            V = rng.normal(0, 5, n).astype(np.float32)
            dt = "float32"
        else:
            V = rng.integers(0, 2, n).astype(np.int32)
            dt = "int32"
        m = rng.integers(0, 2, n).astype(np.int32)
        # force edges across the sweep
        if t % 100 == 0:
            m[:] = 0
        elif t % 100 == 1:
            m[:] = 1
        got = G.filter_compact(V, m, dt)
        ref = np.asarray(V)[m.astype(bool)]
        if got.shape != ref.shape or not (got == ref).all():
            bad += 1
    assert bad == 0, f"{bad}/1000 mismatch"


@pytest.mark.fast
def test_gpu_filter_edges(kernel):
    a = kernel.alias
    V = np.arange(8, dtype=np.int32)
    all_true = [a["ir_series"]("v", V), a["ir_compare"]("m", "v", -1, ">"),
                a["ir_filter"]("f", "v", "m")]
    assert list(_gpu(a, all_true)["f"]) == list(range(8))
    all_false = [a["ir_series"]("v", V), a["ir_compare"]("m", "v", 99, ">"),
                 a["ir_filter"]("f", "v", "m")]
    assert list(_gpu(a, all_false)["f"]) == []
    empty = [a["ir_series"]("v", [], "int32"),
             a["ir_series"]("m", [], "int32"),
             a["ir_compare"]("c", "m", "m", "=="),
             a["ir_filter"]("f", "v", "c")]
    assert list(_gpu(a, empty)["f"]) == []


@pytest.mark.fast
def test_gpu_filter_3vl_excludes(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("v", [10, 20, 30], "int32", [1, 0, 1]),
            a["ir_compare"]("m", "v", 5, ">"),
            a["ir_filter"]("f", "v", "m")]
    bufs = _gpu(a, jobs)
    assert list(bufs["f"]) == [10, 30]
    assert list(bufs["f#validity"]) == [True, True]


@pytest.mark.fast
def test_gpu_filter_size_mismatch_errors(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("v", [1, 2, 3]),
            a["ir_series"]("w", [1, 2]),
            a["ir_compare"]("m", "w", 0, ">"),
            a["ir_filter"]("f", "v", "m")]
    with pytest.raises(ValueError, match="filter"):
        _gpu(a, jobs)


@pytest.mark.fast
def test_gpu_filter_indices_stream(kernel):
    """filter_indices feeds gather == filter (internal stream, no IR)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "nfgpu2", str(Path(APP_DIR) / "src" / "Drivers" / "GPU" / "_lib"
                      / "gpu.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    rng = np.random.default_rng(42)
    for n in (0, 7, 3000, 100000):
        V = rng.integers(0, 100, n, dtype=np.int32)
        m = (V > 50).astype(np.int32)
        ix = G.filter_indices(m)
        assert (ix == np.flatnonzero(V > 50)).all()
        assert (V[ix] == V[V > 50]).all()


@pytest.mark.fast
def test_plan_filter_and_eligible(kernel):
    a = kernel.alias
    assert a["plan_filter"](100)["strategy"] == "host"
    assert a["plan_filter"](10_000_000)["strategy"] == "gpu_blocks"
    jobs = [a["ir_series"]("v", [1, 2, 3]),
            a["ir_compare"]("m", "v", 1, ">"),
            a["ir_filter"]("f", "v", "m")]
    sel = a["select_backend"](a["compile"](jobs), 3)
    assert sel["gpu_eligible"], sel


@pytest.mark.fast
def test_resident_chain_matches_cpu(kernel):
    """Consumer: Filter->GroupBy resident == CPU dict (seed 42)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "nfgpu3", str(Path(APP_DIR) / "src" / "Drivers" / "GPU" / "_lib"
                      / "gpu.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    rng = np.random.default_rng(42)
    N = 50000
    vals = rng.integers(0, 1000, N, dtype=np.int32)
    keys = rng.integers(0, 40, N, dtype=np.int32)
    mk = (vals > 500).astype(np.int32)
    uk, cc, ss = G.filter_groupby_resident(keys, vals, mk, "sum")
    sel = vals > 500
    ref = {}
    cnt = {}
    for kk, vv in zip(keys[sel].tolist(), vals[sel].tolist()):
        ref[kk] = ref.get(kk, 0) + vv
        cnt[kk] = cnt.get(kk, 0) + 1
    assert {int(k): int(v) for k, v in zip(uk.tolist(), ss.tolist())} == ref
    assert [int(c) for c in cc.tolist()] == [cnt[int(k)] for k in uk.tolist()]
