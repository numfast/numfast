"""Driver Conformance — Phase 14.1.

Доказывает, что CpuDriver и WebGpuDriver (Mock) взаимозаменяемы.
Все операции проходят одинаковый набор тестов.

Ни одно изменение в Core (Planner, Kernel, NumericSeries, CoreAPI)
не требуется.

Запуск:
    python -m pytest numfast/core/tests/test_gpu_conformance.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import numpy as np

from numfast.core.nseries import NumericSeries
from numfast.core.driver import CpuDriver
from numfast.core.gpu_driver import MockGpuDriver, WebGpuDriver
from numfast.core.executor import CpuKernel, CoreAPI


# ── Фикстуры ────────────────────────────────────────────────────

# Оба драйвера должны проходить одинаковые тесты
_DRIVER_LIST = [
    pytest.param(CpuDriver(), id="cpu"),
    pytest.param(MockGpuDriver(), id="mock_gpu"),
]

# WebGpuDriver — только если wgpu доступен
try:
    d = WebGpuDriver()
    d.barrier()
    _DRIVER_LIST.append(pytest.param(d, id="webgpu"))
except Exception:
    pass


@pytest.fixture(params=_DRIVER_LIST)
def driver(request):
    return request.param


# ── Conformance Suite ──────────────────────────────────────────────

class TestDriverConformance:
    """Оба Driver проходят одинаковый набор тестов."""

    def _api(self, driver):
        """Создать CoreAPI с указанным driver."""
        kernel = CpuKernel(driver)
        return CoreAPI(kernel=kernel)

    # ── SMA ────────────────────────────────────────────────

    def test_sma_basic(self, driver):
        """SMA(3) — базовый."""
        api = self._api(driver)
        s = NumericSeries([1, 2, 3, 4, 5])
        result = api.sma(s, [3])[0]
        assert result.valid_from == 2
        assert result.read(2) == 2  # (1+2+3)//3

    def test_sma_multi_window(self, driver):
        """SMA(3) + SMA(5) — merge."""
        api = self._api(driver)
        s = NumericSeries([1, 2, 3, 4, 5, 6, 7])
        results = api.sma(s, [3, 5])
        assert len(results) == 2
        assert results[0].valid_from == 2  # 3-1
        assert results[1].valid_from == 4  # 5-1

    # ── Rolling MIN/MAX ────────────────────────────────────

    def test_rolling_min(self, driver):
        """Rolling MIN(3)."""
        api = self._api(driver)
        s = NumericSeries([3, 1, 4, 1, 5, 9, 2])
        result = api.rolling_min(s, 3)
        assert result.valid_from == 2
        # window [3,1,4] → min=1, [1,4,1] → min=1, [4,1,5] → min=1
        assert result.read(2) == min(3, 1, 4)  # 1
        assert result.read(3) == min(1, 4, 1)  # 1

    def test_rolling_max(self, driver):
        """Rolling MAX(3)."""
        api = self._api(driver)
        s = NumericSeries([3, 1, 4, 1, 5, 9, 2])
        result = api.rolling_max(s, 3)
        assert result.valid_from == 2
        assert result.read(2) == max(3, 1, 4)  # 4
        assert result.read(4) == max(4, 1, 5)  # window [4,1,5] → 5

    # ── STDDEV ─────────────────────────────────────────────

    def test_stddev(self, driver):
        """STDDEV(3) — σ = isqrt((w·sum2 — sum²) / w²)."""
        api = self._api(driver)
        s = NumericSeries([1, 4, 7, 10, 13])
        result = api.rolling_stddev(s, 3)
        assert result.valid_from == 2
        # Window [1,4,7]: sum=12, sum2=66, numerator=54, variance=6, σ=2
        assert result.read(2) == 2  # isqrt(6) = 2

    # ── ROC ────────────────────────────────────────────────

    def test_roc(self, driver):
        """ROC(3): (close - close[-3]) / close[-3] * SCALE."""
        api = self._api(driver)
        s = NumericSeries([2, 3, 4, 5, 6, 7])
        result = api.roc(s, 3)
        assert result.valid_from == 3
        # ROC[3] = (5-2)*ROC_SCALE/2 = 3*100000/2 = 150000
        assert result.read(3) == 150000

    # ── ATR (multi-input) ──────────────────────────────────

    def test_atr(self, driver):
        """ATR(3) — три входные серии."""
        api = self._api(driver)
        high = NumericSeries([3, 5, 4, 6, 7])
        low  = NumericSeries([1, 2, 2, 3, 4])
        close = NumericSeries([2, 3, 3, 5, 6])
        result = api.atr(high, low, close, 3)
        assert result.valid_from == 2
        # TR[0]=2, TR[1]=3, TR[2]=2 → ATR[2]=(2+3+2)//3 = 2
        assert result.read(2) == 2

    # ── EMA (stateful) ─────────────────────────────────────

    def test_ema(self, driver):
        """EMA(3) — рекурсия."""
        api = self._api(driver)
        s = NumericSeries([100, 102, 104, 106, 108])
        result = api.ema(s, 3)
        assert result.valid_from == 0
        # EMA[0]=100, EMA[1]=101, EMA[2]=102, EMA[3]=104
        assert result.read(0) == 100
        assert result.read(1) == 101

    # ── Инвариант I2: Driver скрывает физический формат ──

    def test_driver_hides_storage(self, driver):
        """Kernel не должен знать, какой Driver внутри."""
        kernel = CpuKernel(driver)
        assert hasattr(kernel.driver, 'read')
        assert hasattr(kernel.driver, 'write')
        assert hasattr(kernel.driver, 'allocate')
        assert hasattr(kernel.driver, 'dispatch')
        assert hasattr(kernel.driver, 'barrier')

    # ── Большой тест ───────────────────────────────────────

    def test_chain_ema_roc(self, driver):
        """Close → EMA(14) → ROC(5) — цепочка зависимостей."""
        api = self._api(driver)
        np.random.seed(42)
        n = 200
        values = list(np.random.randn(n).cumsum() + 10000)
        close = NumericSeries(values)
        ema14 = api.ema(close, 14)
        roc5 = api.roc(ema14, 5)
        assert isinstance(roc5, NumericSeries)
        assert roc5.read(roc5.n - 1) is not None
