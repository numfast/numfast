# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: universal gather (positional take) + ClickBench chains (spec 02).

Contract under test (unified, CPU == GPU):
- indices int32 canonical (int64 accepted); non-integer -> explicit error.
- OOB (i < 0 or i >= n) -> explicit error, never wrap; empty -> empty.
- order = indices order; dtype preserved; values validity follows the take.
- chains: sort->gather (ORDER BY), sort->slice->gather (top-k),
  lookup positions[hit] -> gather(payload) (lookup->gather).
Seed 42; stage breakdown printed per run.
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


def _cpu(a, jobs):
    s = time.perf_counter()
    graph = a["optimize"](a["compile"](jobs))
    t_compile = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    bufs = a["cpu_execute"](graph["nodes"])
    t_exec = (time.perf_counter() - s) * 1000
    print(f"\ncpu stages ms: compile+optimize={t_compile:.3f} execute={t_exec:.3f}")
    return bufs


def _gpu(a, jobs):
    s = time.perf_counter()
    g = a["compile"](jobs)
    t_compile = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    bufs = a["gpu_execute"](g["nodes"])
    t_exec = (time.perf_counter() - s) * 1000
    print(f"\ngpu stages ms: compile={t_compile:.3f} execute={t_exec:.3f}")
    return bufs


def _jobs(a, values, indices, dtype="int32", validity=None):
    kw = {"values": values, "dtype": dtype}
    if validity is not None:
        kw["validity"] = validity
    return [a["ir_series"]("v", **kw), a["ir_series"]("i", indices),
            a["ir_gather"]("g", "v", "i")]


@pytest.mark.fast
def test_gather_order_and_parity(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    vals = rng.integers(0, 1000, size=1000).astype(np.int32)
    perm = rng.permutation(1000).astype(np.int32)
    jobs = _jobs(a, vals, perm)
    c, g = _cpu(a, jobs)["g"], _gpu(a, jobs)["g"]
    assert (c == vals[perm]).all()
    assert (g == c).all()


@pytest.mark.fast
def test_gather_f32_parity(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    vals = rng.normal(size=512).astype(np.float32)
    idx = rng.permutation(512).astype(np.int32)
    c, g = _cpu(a, _jobs(a, vals, idx, dtype="float32"))["g"], \
        _gpu(a, _jobs(a, vals, idx, dtype="float32"))["g"]
    assert (c == vals[idx]).all()
    assert (g == c).all()


@pytest.mark.fast
def test_gather_validity_parity(kernel):
    a = kernel.alias
    jobs = _jobs(a, [10, 20, 30, 40], [3, 0, 2], validity=[1, 0, 1, 1])
    c, g = _cpu(a, jobs), _gpu(a, jobs)
    assert [bool(v) for v in c["g#validity"]] == [True, True, True]
    assert [bool(v) for v in g["g#validity"]] == [True, True, True]


@pytest.mark.fast
def test_gather_empty(kernel):
    a = kernel.alias
    jobs = _jobs(a, [1, 2, 3], np.zeros(0, dtype=np.int32))
    assert _cpu(a, jobs)["g"].size == 0
    assert _gpu(a, jobs)["g"].size == 0


@pytest.mark.fast
def test_gather_oob_raises_both_backends(kernel):
    a = kernel.alias
    for bad in ([5], [-1]):
        jobs = _jobs(a, [1, 2, 3], bad)
        with pytest.raises(ValueError, match="out of range"):
            _cpu(a, jobs)
        with pytest.raises(ValueError, match="out of range"):
            _gpu(a, jobs)


@pytest.mark.fast
def test_gather_non_integer_indices_raise(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("v", [1, 2, 3]),
            a["ir_series"]("i", [0.5, 1.5], dtype="float64"),
            a["ir_gather"]("g", "v", "i")]
    with pytest.raises(ValueError, match="integer indices"):
        _cpu(a, jobs)
    # GPU rejects float64 already at series ingest (MapF64 guard) --
    # float indices never reach the gather kernel on either backend.
    with pytest.raises(ValueError):
        _gpu(a, jobs)


@pytest.mark.fast
def test_gather_int64_indices_accepted(kernel):
    a = kernel.alias
    jobs = _jobs(a, [7, 8, 9], np.array([2, 0], dtype=np.int64))
    assert [int(v) for v in _cpu(a, jobs)["g"]] == [9, 7]
    assert [int(v) for v in _gpu(a, jobs)["g"]] == [9, 7]


@pytest.mark.fast
def test_chain_sort_gather_parity(kernel):
    """ClickBench ORDER BY: sort perm -> gather, CPU == GPU."""
    a = kernel.alias
    rng = np.random.default_rng(42)
    vals = rng.integers(0, 10_000, size=2048).astype(np.int32)
    jobs = [a["ir_series"]("v", vals), a["ir_sort"]("p", "v"),
            a["ir_gather"]("g", "v", "p")]
    c, g = _cpu(a, jobs)["g"], _gpu(a, jobs)["g"]
    assert (c == np.sort(vals, kind="stable")).all()
    assert (g == c).all()


@pytest.mark.fast
def test_chain_sort_slice_gather_topk(kernel):
    """ClickBench top-k: sort -> slice(limit) -> gather."""
    a = kernel.alias
    rng = np.random.default_rng(42)
    vals = rng.integers(0, 10_000, size=2048).astype(np.int32)
    jobs = [a["ir_series"]("v", vals), a["ir_sort"]("p", "v"),
            a["ir_slice"]("t", "p", limit=10, offset=0),
            a["ir_gather"]("g", "v", "t")]
    c = _cpu(a, jobs)["g"]
    assert (c == np.sort(vals, kind="stable")[:10]).all()
    g = _gpu(a, jobs)["g"]
    assert (g == c).all()


@pytest.mark.fast
def test_chain_lookup_gather(kernel):
    """lookup positions[hit] -> gather(payload): composition golden."""
    a = kernel.alias
    build = np.array([10, 20, 30], dtype=np.int32)
    probe = np.array([30, 10, 99], dtype=np.int32)
    payload = np.array([1, 2, 3], dtype=np.int32)
    lj = [a["ir_series"]("b", build), a["ir_series"]("q", probe),
          a["ir_lookup"]("lk", "b", "q")]
    lb = _cpu(a, lj)
    hit = np.asarray(lb["lk#hit"], dtype=bool)
    pos = np.asarray(lb["lk"]).astype(np.int32)[hit]
    gj = [a["ir_series"]("pl", payload), a["ir_series"]("pp", pos),
          a["ir_gather"]("g", "pl", "pp")]
    got = _cpu(a, gj)["g"]
    assert [int(v) for v in got] == [3, 1]
    got_gpu = _gpu(a, gj)["g"]
    assert (got_gpu == got).all()


@pytest.mark.fast
def test_gather_benchmark(kernel):
    """Benchmark: 100k sort->gather + gather take (seed 42, ms + rows/s)."""
    a = kernel.alias
    rng = np.random.default_rng(42)
    n = 100_000
    vals = rng.integers(0, 1_000_000, size=n).astype(np.int32)
    s = time.perf_counter()
    graph = a["optimize"](a["compile"]([a["ir_series"]("v", vals),
                                        a["ir_sort"]("p", "v"),
                                        a["ir_gather"]("g", "v", "p")]))
    t_compile = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    bufs = a["cpu_execute"](graph["nodes"])
    t_exec = (time.perf_counter() - s) * 1000
    assert bufs["g"].size == n
    print(f"\nbench sort->gather n={n}: compile+optimize={t_compile:.3f}ms "
          f"execute={t_exec:.3f}ms throughput={n / max(t_exec, 1e-9) * 1000:.0f} rows/s")
    idx = rng.permutation(n).astype(np.int32)
    s = time.perf_counter()
    bufs = a["cpu_execute"](a["optimize"](a["compile"](
        [a["ir_series"]("v", vals), a["ir_series"]("i", idx),
         a["ir_gather"]("g", "v", "i")]))["nodes"])
    t_take = (time.perf_counter() - s) * 1000
    assert (bufs["g"] == vals[idx]).all()
    print(f"bench gather take n={n}: execute={t_take:.3f}ms "
          f"throughput={n / max(t_take, 1e-9) * 1000:.0f} rows/s")
