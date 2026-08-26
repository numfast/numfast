"""ExecutionScheduler: юнит-тесты чанкования больших elementwise-графов.

Покрытие:
  1. Map на 10M элементов — чанкование (раньше GPUValidationError).
  2. MapBinary (add) на 10M — два входа, чанкование.
  3. Map на 100K — малый размер, обычный путь без чанкования.
  4. Принудительное чанкование (scheduler_max_elements) == обычный путь
     бит-в-бит.
  5. Неполный хвост: 4,194,240 + 12,345 элементов (два чанка).
  6. Reduce (workspace-kernel) — НЕ чанкуется, обычный путь корректен.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "math")))

from Runtime._lib.runtime import Runtime
from Compute import register_all as register_compute
from Runtime._lib.Drivers.WebGPU import WebGpuDriver

RANDOM_STATE = 42
TOL_SCALE = 1e-3


def _make_data(n, count=1):
    """Синтетические данные: randn(n) * 5.0 + 30.0, фиксированный seed 42."""
    rng = np.random.RandomState(RANDOM_STATE)
    if count == 1:
        return rng.randn(n) * 5.0 + 30.0
    return [rng.randn(n) * 5.0 + 30.0 for _ in range(count)]


def _make_gpu_runtime(**kwargs):
    """Runtime на WebGpuDriver со всеми Compute-ядрами."""
    rt = Runtime(driver=WebGpuDriver(), **kwargs)
    register_compute(rt)
    return rt


def _tol_for(ref):
    """Допуск: 1e-3 * max(1, max|ref|)."""
    return TOL_SCALE * max(1.0, float(np.abs(ref).max()))


def test_map_10m_split():
    """Map(sin) на 10M элементов: чанкование сработало, результат корректен."""
    n = 10_000_000
    rt = _make_gpu_runtime()
    try:
        x = _make_data(n)
        tasks = rt.compile([{"op": "Map", "inputs": ["x"], "params": {"func": 0}}])
        rt.execute(tasks, {"x": x})

        result = rt.driver.resolve_output("map_0")
        assert result is not None, "map_0 should exist"
        assert result.shape == (n,), f"expected shape {(n,)}, got {result.shape}"

        ref = np.sin(x.astype(np.float32))
        err = float(np.abs(result - ref).max())
        assert err <= _tol_for(ref), f"max err {err} > tol {_tol_for(ref)}"
    finally:
        rt.driver.release()


def test_mapbinary_10m_split():
    """MapBinary(add) на 10M элементов: два входа, чанкование."""
    n = 10_000_000
    rt = _make_gpu_runtime()
    try:
        a, b = _make_data(n, count=2)
        tasks = rt.compile([
            {"op": "MapBinary", "inputs": ["a", "b"], "params": {"op": 0}},
        ])
        rt.execute(tasks, {"a": a, "b": b})

        result = rt.driver.resolve_output("map_binary_0")
        assert result is not None, "map_binary_0 should exist"
        assert result.shape == (n,), f"expected shape {(n,)}, got {result.shape}"

        ref = a.astype(np.float32) + b.astype(np.float32)
        err = float(np.abs(result - ref).max())
        assert err <= _tol_for(ref), f"max err {err} > tol {_tol_for(ref)}"
    finally:
        rt.driver.release()


def test_small_unchanged():
    """Map(sin) на 100K: чанкование не срабатывает, обычный путь."""
    n = 100_000
    rt = _make_gpu_runtime()
    try:
        x = _make_data(n)
        tasks = rt.compile([{"op": "Map", "inputs": ["x"], "params": {"func": 0}}])
        rt.execute(tasks, {"x": x})

        result = rt.driver.resolve_output("map_0")
        assert result is not None, "map_0 should exist"
        assert result.shape == (n,), f"expected shape {(n,)}, got {result.shape}"

        ref = np.sin(x.astype(np.float32))
        err = float(np.abs(result - ref).max())
        assert err <= _tol_for(ref), f"max err {err} > tol {_tol_for(ref)}"
    finally:
        rt.driver.release()


def test_force_chunk_consistency():
    """Принудительное чанкование == обычный путь бит-в-бит."""
    n = 1_000_000
    chunked_rt = _make_gpu_runtime(scheduler_max_elements=100_000)
    normal_rt = _make_gpu_runtime()
    try:
        x = _make_data(n)

        chunked_rt.execute(
            chunked_rt.compile([{"op": "Map", "inputs": ["x"], "params": {"func": 0}}]),
            {"x": x},
        )
        normal_rt.execute(
            normal_rt.compile([{"op": "Map", "inputs": ["x"], "params": {"func": 0}}]),
            {"x": x},
        )

        res_chunked = chunked_rt.driver.resolve_output("map_0")
        res_normal = normal_rt.driver.resolve_output("map_0")
        assert res_chunked is not None, "chunked map_0 should exist"
        assert res_normal is not None, "normal map_0 should exist"
        assert res_chunked.shape == (n,), f"expected shape {(n,)}, got {res_chunked.shape}"

        max_diff = float(np.abs(res_chunked - res_normal).max())
        assert max_diff == 0.0, f"chunked vs normal max diff {max_diff} != 0"
    finally:
        chunked_rt.driver.release()
        normal_rt.driver.release()


def test_odd_tail():
    """Map(sin) на 4,194,240 + 12,345: неполный последний чанк."""
    n = 4_194_240 + 12_345
    rt = _make_gpu_runtime()
    try:
        x = _make_data(n)
        tasks = rt.compile([{"op": "Map", "inputs": ["x"], "params": {"func": 0}}])
        rt.execute(tasks, {"x": x})

        result = rt.driver.resolve_output("map_0")
        assert result is not None, "map_0 should exist"
        assert result.shape == (n,), f"expected shape {(n,)}, got {result.shape}"

        ref = np.sin(x.astype(np.float32))
        err = float(np.abs(result - ref).max())
        assert err <= _tol_for(ref), f"max err {err} > tol {_tol_for(ref)}"
    finally:
        rt.driver.release()


def test_workspace_kernel_not_split():
    """Reduce (workspace-kernel) на 1M: НЕ чанкуется, обычный путь корректен."""
    n = 1_000_000
    rt = _make_gpu_runtime()
    try:
        x = _make_data(n)
        tasks = rt.compile([
            {"op": "Reduce", "inputs": ["x"], "params": {}, "out": "y"},
        ])

        # Прямая проверка: scheduler возвращает None для workspace-kernels
        from Runtime._lib.planner import planner
        from Runtime._lib.optimizer import optimize_graph
        from Runtime._lib.execution_scheduler import ExecutionScheduler

        graph = planner(tasks, rt.kernel_table)
        graph = optimize_graph(graph, level=rt.optimizer_level)
        sch = ExecutionScheduler()
        assert sch.execute(graph, {"x": x}, rt.driver, rt.kernel_table) is None

        # Обычный путь через Runtime.execute
        rt.execute(tasks, {"x": x})

        result = rt.driver.resolve_output("y")
        assert result is not None, "y should exist"
        total = float(np.sum(result))
        ref_total = float(np.sum(x))
        rel_err = abs(total - ref_total) / max(1.0, abs(ref_total))
        assert rel_err <= 1e-2, f"rel err {rel_err} > 1e-2 (total={total}, ref={ref_total})"
    finally:
        rt.driver.release()


def test_driver_max_dispatch_elements():
    """Лимит dispatch: WebGPU = 4,194,240; CPU/base = None (ADR-012 amendment)."""
    from Runtime._lib.Drivers.base import Driver
    from Runtime._lib.Drivers.CPU._lib.cpu_driver import CpuDriver

    class _MinDriver(Driver):
        def execute(self, packet):
            pass

    gpu = WebGpuDriver()
    cpu = CpuDriver()
    try:
        assert gpu.max_dispatch_elements() == 4_194_240, \
            f"WebGPU limit, got {gpu.max_dispatch_elements()}"
        assert _MinDriver().max_dispatch_elements() is None
        assert cpu.max_dispatch_elements() is None
    finally:
        gpu.release()
        cpu.release()


def test_capabilities_chunkable():
    """capabilities: Map/MapBinary chunkable=True; остальные — нет."""
    rt = _make_gpu_runtime()
    try:
        assert rt.kernel_table["Map"]["capabilities"]["chunkable"] is True
        assert rt.kernel_table["MapBinary"]["capabilities"]["chunkable"] is True
        for alias in ("Reduce", "ScanLocal", "Histogram", "StateKernel_Single"):
            caps = rt.kernel_table[alias]["capabilities"]
            assert caps.get("chunkable", False) is False, \
                f"{alias} must not be chunkable"
    finally:
        rt.driver.release()


def test_scheduler_has_no_magic_numbers():
    """Acceptance: в планировщике нет магических чисел и SAFE_KERNELS."""
    import Runtime._lib.execution_scheduler as module
    source = open(module.__file__, encoding="utf-8").read()
    assert "4194240" not in source
    assert "65535" not in source
    assert "SAFE_KERNELS" not in source
