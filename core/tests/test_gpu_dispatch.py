"""Phase 21 — GPU dispatch: SMA, ROC, MIN/MAX, Pointwise.

Проверяет, что GPU-диспетчеризация даёт бит-в-бит тот же результат,
что и CPU.

Запуск:
    python -m pytest numfast/core/tests/test_gpu_dispatch.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from numfast.core.nseries import NumericSeries
from numfast.core.driver import CpuDriver
from numfast.core.gpu_driver import WebGpuDriver
from numfast.core.executor import CpuKernel, CoreAPI


class TestGpuSma:
    """SMA: CPU vs GPU (WebGPU) — bit-exact."""

    @pytest.fixture
    def webgpu_driver(self):
        try:
            d = WebGpuDriver()
            d.barrier()
            return d
        except Exception as e:
            pytest.skip(f"WebGPU not available: {e}")

    @pytest.mark.parametrize("w", [3, 5, 14, 30])
    def test_sma_cpu_vs_gpu(self, w, webgpu_driver):
        n = 500
        np.random.seed(42)
        prices = (10000 + np.random.randn(n).cumsum()).astype(np.int32)

        # CPU SMA
        cpu_series = NumericSeries(prices.copy(), offset=0, scale=1)
        cpu_kernel = CpuKernel(CpuDriver())
        cpu_api = CoreAPI(kernel=cpu_kernel)
        cpu_result = cpu_api.sma(cpu_series, [w])[0]

        # GPU SMA
        gpu_series = NumericSeries(prices.copy(), offset=0, scale=1)
        gpu_series2 = webgpu_driver.allocate(n, offset=0, scale=1, valid_from=0)
        for i in range(n):
            gpu_series2.write(i, int(prices[i]))

        gpu_kernel = CpuKernel(webgpu_driver)
        gpu_api = CoreAPI(kernel=gpu_kernel)
        gpu_result = gpu_api.sma(gpu_series2, [w])[0]

        max_diff = 0
        for i in range(n):
            cpu_val = cpu_result.read(i)
            gpu_val = gpu_result.read(i)
            diff = abs(cpu_val - gpu_val)
            if diff > max_diff:
                max_diff = diff

        assert max_diff == 0, f"SMA({w}): CPU vs GPU max_diff={max_diff}"

    @pytest.mark.parametrize("w", [3, 5, 14])
    def test_sma_gpu_reuse(self, w, webgpu_driver):
        """Multiple SMA calls reuse cached WGSL."""
        n = 100
        prices = np.array([100 + i for i in range(n)], dtype=np.int32)

        series = webgpu_driver.allocate(n, offset=0, scale=1, valid_from=0)
        for i in range(n):
            series.write(i, int(prices[i]))

        kernel = CpuKernel(webgpu_driver)
        api = CoreAPI(kernel=kernel)

        r1 = api.sma(series, [w])[0]
        r2 = api.sma(series, [w])[0]

        for i in range(n):
            assert r1.read(i) == r2.read(i), f"Mismatch at {i}"


class TestGpuRoc:
    """ROC: CPU vs GPU (WebGPU) — bit-exact."""

    @pytest.fixture
    def webgpu_driver(self):
        try:
            d = WebGpuDriver()
            d.barrier()
            return d
        except Exception as e:
            pytest.skip(f"WebGPU not available: {e}")

    @pytest.mark.parametrize("period", [1, 5, 14])
    def test_roc_cpu_vs_gpu(self, period, webgpu_driver):
        n = 500
        np.random.seed(42)
        prices = (10000 + np.random.randn(n).cumsum()).astype(np.int32)

        cpu_series = NumericSeries(prices.copy(), offset=0, scale=1)
        cpu_kernel = CpuKernel(CpuDriver())
        cpu_api = CoreAPI(kernel=cpu_kernel)
        cpu_result = cpu_api.roc(cpu_series, period)

        gpu_series = webgpu_driver.allocate(n, offset=0, scale=1, valid_from=0)
        for i in range(n):
            gpu_series.write(i, int(prices[i]))

        gpu_kernel = CpuKernel(webgpu_driver)
        gpu_api = CoreAPI(kernel=gpu_kernel)
        gpu_result = gpu_api.roc(gpu_series, period)

        max_diff = 0
        for i in range(period, n):
            diff = abs(cpu_result.read(i) - gpu_result.read(i))
            if diff > max_diff:
                max_diff = diff

        assert max_diff == 0, f"ROC({period}): CPU vs GPU max_diff={max_diff}"


class TestGpuMinMax:
    """Rolling MIN/MAX: CPU vs GPU — bit-exact."""

    @pytest.fixture
    def webgpu_driver(self):
        try:
            d = WebGpuDriver()
            d.barrier()
            return d
        except Exception as e:
            pytest.skip(f"WebGPU not available: {e}")

    @pytest.mark.parametrize("w", [5, 14, 30])
    @pytest.mark.parametrize("op", ["min", "max"])
    def test_minmax_cpu_vs_gpu(self, w, op, webgpu_driver):
        n = 500
        np.random.seed(42)
        prices = (10000 + np.random.randn(n).cumsum()).astype(np.int32)

        cpu_series = NumericSeries(prices.copy(), offset=0, scale=1)
        cpu_kernel = CpuKernel(CpuDriver())
        cpu_api = CoreAPI(kernel=cpu_kernel)
        if op == "min":
            cpu_result = cpu_api.rolling_min(cpu_series, w)
        else:
            cpu_result = cpu_api.rolling_max(cpu_series, w)

        gpu_series = webgpu_driver.allocate(n, offset=0, scale=1, valid_from=0)
        for i in range(n):
            gpu_series.write(i, int(prices[i]))

        gpu_kernel = CpuKernel(webgpu_driver)
        gpu_api = CoreAPI(kernel=gpu_kernel)
        if op == "min":
            gpu_result = gpu_api.rolling_min(gpu_series, w)
        else:
            gpu_result = gpu_api.rolling_max(gpu_series, w)

        max_diff = 0
        for i in range(w - 1, n):
            diff = abs(cpu_result.read(i) - gpu_result.read(i))
            if diff > max_diff:
                max_diff = diff

        assert max_diff == 0, f"{op.upper()}({w}): CPU vs GPU max_diff={max_diff}"


class TestGpuPointwise:
    """Pointwise (add/sub/mul/div): CPU vs GPU — bit-exact."""

    @pytest.fixture
    def webgpu_driver(self):
        try:
            d = WebGpuDriver()
            d.barrier()
            return d
        except Exception as e:
            pytest.skip(f"WebGPU not available: {e}")

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_pointwise_cpu_vs_gpu(self, op, webgpu_driver):
        n = 500
        np.random.seed(42)
        a = (10000 + np.random.randn(n).cumsum()).astype(np.int32)
        b = (5000 + np.random.randn(n).cumsum()).astype(np.int32)

        op_map = {
            "add": lambda x, y: x + y,
            "sub": lambda x, y: x - y,
            "mul": lambda x, y: x * y,
            "div": lambda x, y: x // y,
        }

        cpu_a = NumericSeries(a.copy(), offset=0, scale=1)
        cpu_b = NumericSeries(b.copy(), offset=0, scale=1)
        cpu_result = op_map[op](cpu_a.read(0), cpu_b.read(0))

        gpu_a = webgpu_driver.allocate(n, offset=0, scale=1, valid_from=0)
        gpu_b = webgpu_driver.allocate(n, offset=0, scale=1, valid_from=0)
        for i in range(n):
            gpu_a.write(i, int(a[i]))
            gpu_b.write(i, int(b[i]))

        # Direct dispatch — pointwise не через CoreAPI (нет публичного API)
        gpu_kernel = CpuKernel(webgpu_driver)
        out = webgpu_driver.allocate(n, offset=0, scale=1, valid_from=0)
        job = gpu_kernel._ops.get("pointwise")
        if job is None:
            pytest.skip("pointwise op not registered")

        # Создаём Job вручную
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from numfast.core.executor import Job
        j = Job(op_id="pointwise", inputs=[gpu_a, gpu_b], params={"op": op})
        j.outputs = [out]
        gpu_kernel.run(j)

        max_diff = 0
        for i in range(n):
            cpu_val = op_map[op](int(a[i]), int(b[i]))
            gpu_val = out.read(i)
            diff = abs(cpu_val - gpu_val)
            if diff > max_diff:
                max_diff = diff

        assert max_diff == 0, f"Pointwise({op}): CPU vs GPU max_diff={max_diff}"
