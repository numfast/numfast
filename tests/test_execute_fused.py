"""execute_fused: юнит-тесты fused-исполнения цепочки пакетов.

Покрытие:
  1. Цепочка Map -> MapBinary -> Shift: fused == sequential, промежуточные
     выходы НЕ readback'ятся (остаются нулями-заглушками).
  2. Одиночный пакет (Map): fused == sequential.
  3. ExecutionScheduler fusion: Runtime.execute сам запускает
     execute_fused() для цепочек >1 узла на драйвере с этой возможностью.
  4. CpuDriver (без execute_fused) — обычный путь не сломан.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "math")))

from Runtime._lib.runtime import Runtime
from Compute import register_all as register_compute
from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Runtime._lib.Drivers.CPU._lib.cpu_driver import CpuDriver
from Runtime._lib.planner import planner
from Runtime._lib.optimizer import optimize_graph
from Runtime._lib.builder import builder

N = 1000


def _make_gpu_runtime():
    rt = Runtime(driver=WebGpuDriver())
    register_compute(rt)
    return rt


def _make_data():
    return np.random.default_rng(42).standard_normal(N).astype(np.float64)


def _chain_jobs():
    """Map(square) -> MapBinary(mul, оба входа = выход Map) -> Shift(offset=1)."""
    return [
        {"op": "Map", "inputs": ["in0"], "params": {"func": 7}},
        {"op": "MapBinary", "inputs": ["map_7", "map_7"], "params": {"op": 2}},
        {"op": "Shift", "inputs": ["map_binary_2"], "params": {"offset": 1}},
    ]


def _numpy_chain_ref(data):
    """Numpy-эталон: Map(square) -> MapBinary(mul) -> Shift(offset=1)."""
    x = np.asarray(data, dtype=np.float64)
    sq = x * x
    mul = sq * sq
    out = np.zeros_like(mul)
    out[1:] = mul[:-1]
    return out


def _fused_execute(rt, jobs, source_data):
    """Fused путь: planner + optimizer + builder + driver.execute_fused."""
    rt.driver.kernel_table = rt.kernel_table
    tasks = rt.compile(jobs)
    graph = planner(tasks, rt.kernel_table)
    graph = optimize_graph(graph, level=rt.optimizer_level)
    packets = builder(graph, source_data, rt.driver, rt.kernel_table)
    rt.driver.execute_fused(packets)
    return packets


def _sequential_chain(rt, data):
    """Sequential путь: 3 отдельных compile + execute + resolve."""
    rt.execute(rt.compile([{"op": "Map", "inputs": ["in0"], "params": {"func": 7}}]),
               {"in0": data})
    m = rt.driver.resolve_output("map_7")
    rt.execute(rt.compile([{"op": "MapBinary", "inputs": ["a", "b"], "params": {"op": 2}}]),
               {"a": m, "b": m})
    mb = rt.driver.resolve_output("map_binary_2")
    rt.execute(rt.compile([{"op": "Shift", "inputs": ["s"], "params": {"offset": 1}}]),
               {"s": mb})
    return rt.driver.resolve_output("shift_1")


def _median_ms(fn, runs=3):
    """Median wall time (ms) over `runs` fresh-Runtime executions."""
    samples = []
    for _ in range(runs):
        rt = _make_gpu_runtime()
        try:
            t0 = time.perf_counter()
            fn(rt)
            samples.append((time.perf_counter() - t0) * 1e3)
        finally:
            rt.driver.release()
    return float(np.median(samples))


def test_fused_chain_matches_sequential():
    data = _make_data()

    rt_fused = _make_gpu_runtime()
    try:
        packets = _fused_execute(rt_fused, _chain_jobs(), {"in0": data})

        out_fused = rt_fused.driver.resolve_output("shift_1")
        assert out_fused is not None, "shift_1 should exist"
        assert out_fused.shape == (N,), f"expected shape {(N,)}, got {out_fused.shape}"

        # Промежуточные выходы НЕ readback'ались — остаются нулями-заглушками
        mid1 = rt_fused.driver.resolve_output("map_7")
        mid2 = rt_fused.driver.resolve_output("map_binary_2")
        assert mid1 is not None, "map_7 should exist"
        assert mid2 is not None, "map_binary_2 should exist"
        assert np.all(mid1 == 0), "map_7 must stay zeros (no readback)"
        assert np.all(mid2 == 0), "map_binary_2 must stay zeros (no readback)"

        # Профиль: readback_ns == 0 для промежуточных, > 0 для финального
        assert packets[0].profile["readback_ns"] == 0, "map_7 readback must be 0"
        assert packets[1].profile["readback_ns"] == 0, "map_binary_2 readback must be 0"
        assert packets[2].profile["readback_ns"] > 0, "shift_1 must be read back"
    finally:
        rt_fused.driver.release()

    rt_seq = _make_gpu_runtime()
    try:
        out_seq = _sequential_chain(rt_seq, data)
        assert out_seq is not None, "sequential shift_1 should exist"
    finally:
        rt_seq.driver.release()

    assert np.allclose(out_fused, out_seq, rtol=1e-4, atol=1e-4), (
        f"fused vs sequential max diff "
        f"{float(np.abs(out_fused - out_seq).max())}"
    )

    fused_ms = _median_ms(lambda rt: _fused_execute(rt, _chain_jobs(), {"in0": data}))
    sequential_ms = _median_ms(lambda rt: _sequential_chain(rt, data))
    print(f"fused_ms={fused_ms:.3f} sequential_ms={sequential_ms:.3f}")


def test_scheduler_fuses_chain_via_runtime():
    """ExecutionScheduler fusion: rt.execute сам вызывает execute_fused.

    Цепочка Map -> MapBinary -> Shift (3 узла) на WebGpuDriver: финальный
    выход корректен, промежуточные остаются нулями (нет readback —
    fusion через scheduler сработал).
    """
    data = _make_data()
    rt = _make_gpu_runtime()
    try:
        t0 = time.perf_counter()
        rt.execute(rt.compile(_chain_jobs()), {"in0": data})
        exec_ms = (time.perf_counter() - t0) * 1e3

        out = rt.driver.resolve_output("shift_1")
        assert out is not None, "shift_1 should exist"
        assert out.shape == (N,), f"expected shape {(N,)}, got {out.shape}"

        ref = _numpy_chain_ref(data)
        assert np.allclose(out, ref, rtol=1e-4, atol=1e-4), (
            f"scheduler-fused vs numpy max diff "
            f"{float(np.abs(out - ref).max())}"
        )

        # Промежуточные выходы НЕ readback'ились — остаются нулями
        mid1 = rt.driver.resolve_output("map_7")
        mid2 = rt.driver.resolve_output("map_binary_2")
        assert mid1 is not None, "map_7 should exist"
        assert mid2 is not None, "map_binary_2 should exist"
        assert np.all(mid1 == 0), "map_7 must stay zeros (no readback)"
        assert np.all(mid2 == 0), "map_binary_2 must stay zeros (no readback)"
        print(f"scheduler_fused_exec_ms={exec_ms:.3f}")
    finally:
        rt.driver.release()


def test_scheduler_fusion_preserves_sequential_equivalence():
    """Fusion через scheduler == поштучное исполнение (rt_seq)."""
    data = _make_data()

    rt_fused = _make_gpu_runtime()
    try:
        rt_fused.execute(rt_fused.compile(_chain_jobs()), {"in0": data})
        out_fused = rt_fused.driver.resolve_output("shift_1")
        assert out_fused is not None, "fused shift_1 should exist"
    finally:
        rt_fused.driver.release()

    rt_seq = _make_gpu_runtime()
    try:
        out_seq = _sequential_chain(rt_seq, data)
        assert out_seq is not None, "sequential shift_1 should exist"
    finally:
        rt_seq.driver.release()

    assert np.allclose(out_fused, out_seq, rtol=1e-4, atol=1e-4), (
        f"fused vs sequential max diff "
        f"{float(np.abs(out_fused - out_seq).max())}"
    )


def test_cpu_driver_unchanged():
    """CpuDriver без execute_fused: обычный путь, поведение не сломано."""
    rt = Runtime(driver=CpuDriver())
    register_compute(rt)
    try:
        data = _make_data()
        rt.execute(rt.compile(_chain_jobs()), {"in0": data})

        out = rt.driver.resolve_output("shift_1")
        assert out is not None, "shift_1 should exist"
        assert out.shape == (N,), f"expected shape {(N,)}, got {out.shape}"

        ref = _numpy_chain_ref(data)
        assert np.allclose(out, ref), (
            f"cpu chain vs numpy max diff {float(np.abs(out - ref).max())}"
        )
    finally:
        rt.driver.release()


def test_fused_single_packet():
    data = _make_data()
    job = [{"op": "Map", "inputs": ["in0"], "params": {"func": 7}}]

    rt_fused = _make_gpu_runtime()
    try:
        _fused_execute(rt_fused, job, {"in0": data})
        out_fused = rt_fused.driver.resolve_output("map_7")
        assert out_fused is not None, "map_7 should exist"
    finally:
        rt_fused.driver.release()

    rt_seq = _make_gpu_runtime()
    try:
        rt_seq.execute(rt_seq.compile(job), {"in0": data})
        out_seq = rt_seq.driver.resolve_output("map_7")
        assert out_seq is not None, "sequential map_7 should exist"
    finally:
        rt_seq.driver.release()

    assert np.allclose(out_fused, out_seq, rtol=1e-4, atol=1e-4), (
        f"fused vs sequential max diff "
        f"{float(np.abs(out_fused - out_seq).max())}"
    )
