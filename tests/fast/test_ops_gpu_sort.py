# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""GPU sort golden: bitonic stable permutation vs CPU driver (seed 42).

Contract: single int32/finite-float32 key, power-of-two N -> int32
permutation (canonical resident; sorted keys = gather(keys, perm)
composition, never dict/list). Non-pow2/NaN/multi-key/int64 raise explicit
CPU-only errors (spec 06, no silent fallback). Stability/validity-last match
the CPU driver exactly. Stage breakdown (ms) printed per run.
"""

import time
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])
I32MIN, I32MAX = -(2 ** 31), 2 ** 31 - 1


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
    print(f"\ngpu-sort stages ms: compile={t_compile:.3f} execute={t_exec:.3f}")
    return bufs


def _cpu(a, jobs):
    return a["cpu_execute"](a["compile"](jobs)["nodes"])


def _stable_desc_ref(V):
    n = len(V)
    asc = np.argsort(V, kind="stable")
    sk = V[asc]
    ch = np.empty(n, dtype=bool)
    ch[0] = True
    if n > 1:
        ch[1:] = sk[1:] != sk[:-1]
    grp = np.cumsum(ch)
    return asc[np.argsort(grp.max(initial=0) - grp, kind="stable")]


@pytest.mark.fast
def test_gpu_sort_int32_exact_vs_cpu(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    for n in (4, 64, 1024):
        V = rng.integers(I32MIN, I32MAX + 1, n).astype(np.int32)
        for desc in (False, True):
            jobs = [a["ir_series"]("v", V),
                    a["ir_sort"]("p", "v", descending=desc)]
            got = np.asarray(_gpu(a, jobs)["p"])
            want = np.asarray(_cpu(a, jobs)["p"])
            assert got.dtype == np.dtype(np.int32)
            assert (got == want).all(), f"n={n} desc={desc}"


@pytest.mark.fast
def test_gpu_sort_stable_ties_and_int32min(kernel):
    a = kernel.alias
    V = np.array([I32MIN, 5, I32MIN, 0, 5, I32MAX, 0, 5], dtype=np.int32)
    for desc in (False, True):
        jobs = [a["ir_series"]("v", V),
                a["ir_sort"]("p", "v", descending=desc)]
        got = [int(i) for i in _gpu(a, jobs)["p"]]
        want = [int(i) for i in _cpu(a, jobs)["p"]]
        assert got == want, desc
    assert [int(v) for v in V[np.asarray(_gpu(a, [
        a["ir_series"]("v", V),
        a["ir_sort"]("p", "v", descending=True)])["p"])]] == [
        I32MAX, 5, 5, 5, 0, 0, I32MIN, I32MIN]


@pytest.mark.fast
def test_gpu_sort_float32_finite_exact(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    Vf = rng.normal(0, 5, 1024).astype(np.float32)
    Vf[::7] = np.inf
    Vf[1::11] = -np.inf
    for desc in (False, True):
        jobs = [a["ir_series"]("v", Vf, "float32"),
                a["ir_sort"]("p", "v", descending=desc)]
        got = np.asarray(_gpu(a, jobs)["p"])
        want = np.asarray(_cpu(a, jobs)["p"])
        assert (got == want).all(), desc


@pytest.mark.fast
def test_gpu_sort_invalid_rows_last(kernel):
    a = kernel.alias
    # Contract: popcount(valid) must itself be power-of-two (GPU sorts the
    # valid subset); otherwise explicit CPU error, never silent.
    vals = [3, 1, 2, 0, 5, 4, 7, 6]
    validity = [1, 0, 1, 0, 1, 1, 0, 0]
    for desc in (False, True):
        jobs = [a["ir_series"]("v", vals, validity=validity),
                a["ir_sort"]("p", "v", descending=desc)]
        got = [int(i) for i in _gpu(a, jobs)["p"]]
        want = [int(i) for i in _cpu(a, jobs)["p"]]
        assert got == want, desc


@pytest.mark.fast
def test_gpu_sort_validity_nonpow2_raises(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("v", [3, 1, 2, 0, 5, 4, 7, 6],
                           validity=[1, 0, 1, 0, 1, 1, 0, 1]),
            a["ir_sort"]("p", "v")]
    with pytest.raises(ValueError, match="power-of-two|backend='cpu'"):
        _gpu(a, jobs)


@pytest.mark.fast
def test_gpu_sort_explicit_cpu_errors(kernel):
    a = kernel.alias
    V6 = np.arange(6, dtype=np.int32)
    with pytest.raises(ValueError, match="power-of-two|backend='cpu'"):
        _gpu(a, [a["ir_series"]("v", V6), a["ir_sort"]("p", "v")])
    Vn = np.array([1.0, np.nan, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                  dtype=np.float32)
    with pytest.raises(ValueError, match="NaN|backend='cpu'"):
        _gpu(a, [a["ir_series"]("v", Vn, "float32"),
                 a["ir_sort"]("p", "v")])
    with pytest.raises(ValueError, match="pack_keys|CPU-only"):
        _gpu(a, [a["ir_series"]("x", np.arange(8, dtype=np.int32)),
                 a["ir_series"]("y", np.arange(8, dtype=np.int32)),
                 a["ir_sort"]("p", "x", "y")])


@pytest.mark.fast
def test_gpu_sort_planner_eligible_and_not_chunkable(kernel):
    a = kernel.alias
    cap = a["gpu_capability"]()
    assert "sort" in cap["ops"]
    assert cap["chunkable_hints"]["sort"] is False
    graph = a["compile"]([a["ir_series"]("v", [2, 1, 4, 3]),
                          a["ir_sort"]("p", "v")])
    sel = a["select_backend"](graph, 4)
    assert sel["gpu_eligible"] is True
    assert sel["gpu_blockers"] == []
    assert a["chunk_plan"]("sort", 10, backend="gpu")["num_chunks"] == 1


@pytest.mark.fast
def test_gpu_topk_rank_consumers_resident(kernel):
    a = kernel.alias
    vals = [5, 1, 4, 1, 5, 9, 2, 6]
    bufs = _gpu(a, [a["ir_series"]("v", vals),
                    a["ir_sort"]("p", "v", descending=True),
                    a["ir_slice"]("t", "p", limit=3),
                    a["ir_gather"]("g", "v", "t")])
    assert [int(v) for v in bufs["g"]] == [9, 6, 5]
    bufs = _gpu(a, [a["ir_series"]("v", [30, 10, 20, 10]),
                    a["ir_sort"]("p", "v"),
                    a["ir_sort"]("r", "p")])
    assert [int(i) for i in bufs["p"]] == [1, 3, 2, 0]
    assert [int(i) for i in bufs["r"]] == [3, 0, 2, 1]
