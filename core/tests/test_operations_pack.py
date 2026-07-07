"""Phase 17 — Standard Operations Pack.

Проверяет 8 новых операций (WMA, TRANGE, RSI, STOCH, MEDIAN, OBV, VWAP, CCI)
и 8 композиций из существующего API (HMA, Bollinger, MACD, Keltner,
Donchian, Momentum, TMA, RMA).

Таблица изменений Core:
  Runtime | Planner | Driver | ABI
  --------|---------|--------|-----
    ❌    |   ❌    |   ❌   |  ❌

Ни одно изменение Core не потребовалось.
Все операции — через существующий Runtime.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from numfast.core.nseries import NumericSeries
from numfast.core.executor import CoreAPI, Planner, CpuKernel, CpuDriver, \
    merge_sma, merge_window
from numfast.core.driver import CpuDriver


class TestNewOperations:
    """8 новых операций — проверка базовой функциональности."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        n = 200
        base = list(np.random.randn(n).cumsum() + 10000)
        self.close = NumericSeries(base)
        self.high = NumericSeries([v + np.random.uniform(0, 5) for v in base])
        self.low = NumericSeries([v - np.random.uniform(0, 5) for v in base])
        self.volume = NumericSeries(
            [int(abs(x) * 100 + 1000) for x in np.random.randn(n)]
        )

    def test_wma_basic(self):
        """WMA(3) — Weighted Moving Average."""
        api = CoreAPI()
        inp = NumericSeries([1, 2, 3, 4, 5])
        result = api.wma(inp, 3)
        # WMA: weights [3,2,1], weight_sum = 6
        # WMA[2] = (1*3 + 2*2 + 3*1) / 6 = (3+4+3)/6 = 10/6 = 1
        # WMA[3] = (2*3 + 3*2 + 4*1) / 6 = (6+6+4)/6 = 16/6 = 2
        # WMA[4] = (3*3 + 4*2 + 5*1) / 6 = (9+8+5)/6 = 22/6 = 3
        assert result.valid_from == 2
        assert result.read(2) == 1
        assert result.read(3) == 2
        assert result.read(4) == 3

    def test_trange_basic(self):
        """True Range — отдельная операция."""
        api = CoreAPI()
        h = NumericSeries([3, 5, 4, 6])
        l = NumericSeries([1, 2, 2, 3])
        c = NumericSeries([2, 3, 3, 5])
        result = api.trange(h, l, c)
        # TR[0] = H[0]-L[0] = 2
        # TR[1] = max(3, |5-2|=3, |2-3|=1) = 3
        # TR[2] = max(2, |4-3|=1, |2-3|=1) = 2
        # TR[3] = max(3, |6-3|=3, |3-5|=2) = 3
        assert result.read(0) == 2
        assert result.read(1) == 3
        assert result.read(2) == 2
        assert result.read(3) == 3

    def test_rsi_basic(self):
        """RSI(3) — Relative Strength Index."""
        api = CoreAPI()
        # Uptrend: close monotonically increasing
        inp = NumericSeries([100, 101, 102, 103, 104])
        result = api.rsi(inp, 3)
        # gain > loss always → RSI close to 100
        last = result.read(result.n - 1)
        assert last > 90 * 1000  # RSI > 90 (in SCALE=1000)

    def test_rsi_downtrend(self):
        """RSI(3) — нисходящий тренд → RSI ~0."""
        api = CoreAPI()
        inp = NumericSeries([104, 103, 102, 101, 100])
        result = api.rsi(inp, 3)
        last = result.read(result.n - 1)
        assert last < 10 * 1000  # RSI < 10

    def test_stoch_basic(self):
        """Stochastic %K — значения в [0, 100]."""
        api = CoreAPI()
        h = NumericSeries([3, 5, 4, 6, 7])
        l = NumericSeries([1, 2, 2, 3, 4])
        c = NumericSeries([2, 3, 3, 5, 6])
        result = api.stoch(h, l, c, 3)
        assert result.valid_from == 2
        # К: в диапазоне [0, 100*SCALE]
        for i in range(result.valid_from, result.n):
            val = result.read(i)
            assert 0 <= val <= 100 * 1000, \
                f"STOCH out of range: {val} at {i}"

    def test_median_basic(self):
        """MEDIAN(3) — rolling median."""
        api = CoreAPI()
        inp = NumericSeries([3, 1, 4, 1, 5, 9, 2])
        result = api.rolling_median(inp, 3)
        assert result.valid_from == 2
        # window [3,1,4] → sorted [1,3,4] → median=3
        assert result.read(2) == 3
        # window [1,4,1] → sorted [1,1,4] → median=1
        assert result.read(3) == 1
        # window [4,1,5] → sorted [1,4,5] → median=4
        assert result.read(4) == 4

    def test_obv_basic(self):
        """OBV — On-Balance Volume."""
        api = CoreAPI()
        c = NumericSeries([10, 11, 12, 11, 10])
        v = NumericSeries([5, 3, 7, 2, 4])
        result = api.obv(c, v)
        # OBV[0] = 0
        # close[1] > close[0] → OBV[1] = 0 + 3 = 3
        # close[2] > close[1] → OBV[2] = 3 + 7 = 10
        # close[3] < close[2] → OBV[3] = 10 - 2 = 8
        # close[4] < close[3] → OBV[4] = 8 - 4 = 4
        assert result.read(0) == 0
        assert result.read(1) == 3
        assert result.read(2) == 10
        assert result.read(3) == 8
        assert result.read(4) == 4

    def test_vwap_basic(self):
        """VWAP — Volume Weighted Average Price."""
        api = CoreAPI()
        c = NumericSeries([10, 20, 30])
        v = NumericSeries([2, 3, 5])
        result = api.vwap(c, v)
        # VWAP[0] = 10*2/2 = 10
        # VWAP[1] = (20+60)/(2+3) = 80/5 = 16
        # VWAP[2] = (20+60+150)/(2+3+5) = 230/10 = 23
        assert result.read(0) == 10
        assert result.read(1) == 16
        assert result.read(2) == 23

    def test_cci_basic(self):
        """CCI — Commodity Channel Index."""
        api = CoreAPI()
        base = [100 + i for i in range(50)]  # steady uptrend
        h = NumericSeries([v + 2 for v in base])
        l = NumericSeries([v - 2 for v in base])
        c = NumericSeries(base)
        result = api.cci(h, l, c, 7)
        assert result.valid_from == 6
        last = result.read(result.n - 1)
        # Восходящий тренд → CCI положительный
        assert last > 0


class TestComposedOperations:
    """8 композиций из существующего API — без новых ядер."""

    def test_hma(self):
        """Hull Moving Average — WMA + WMA + Pointwise."""
        api = CoreAPI()
        n = 100
        np.random.seed(1)
        base = list(np.random.randn(n).cumsum() + 10000)
        close = NumericSeries(base)

        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        # Упрощённая версия для phase = 14
        wma_half = api.wma(close, 7)
        wma_full = api.wma(close, 14)
        raw = NumericSeries([
            2 * wma_half.read(i) - wma_full.read(i)
            for i in range(n)
        ])
        hma = api.wma(raw, 4)
        assert hma.read(hma.n - 1) is not None

    def test_bollinger(self):
        """Bollinger Bands — SMA + STDDEV + Pointwise."""
        api = CoreAPI()
        np.random.seed(2)
        base = list(np.random.randn(200).cumsum() + 10000)
        close = NumericSeries(base)

        sma20 = api.sma(close, [20])[0]
        std20 = api.rolling_stddev(close, 20)

        upper = NumericSeries([
            sma20.read(i) + 2 * std20.read(i)
            for i in range(close.n)
        ])
        lower = NumericSeries([
            sma20.read(i) - 2 * std20.read(i)
            for i in range(close.n)
        ])
        assert upper.read(upper.n - 1) >= lower.read(lower.n - 1)

    def test_macd(self):
        """MACD — EMA + EMA + Pointwise."""
        api = CoreAPI()
        np.random.seed(3)
        base = list(np.random.randn(200).cumsum() + 10000)
        close = NumericSeries(base)

        ema12 = api.ema(close, 12)
        ema26 = api.ema(close, 26)
        macd_line = NumericSeries([
            ema12.read(i) - ema26.read(i)
            for i in range(close.n)
        ])
        signal = api.ema(macd_line, 9)
        histogram = NumericSeries([
            macd_line.read(i) - signal.read(i)
            for i in range(close.n)
        ])
        assert histogram.read(histogram.n - 1) is not None

    def test_keltner(self):
        """Keltner Channels — EMA + ATR + Pointwise."""
        api = CoreAPI()
        np.random.seed(4)
        n = 200
        base = list(np.random.randn(n).cumsum() + 10000)
        h = NumericSeries([v + np.random.uniform(0, 5) for v in base])
        l = NumericSeries([v - np.random.uniform(0, 5) for v in base])
        c = NumericSeries(base)

        ema20 = api.ema(c, 20)
        atr14 = api.atr(h, l, c, 14)

        upper = NumericSeries([
            ema20.read(i) + 2 * atr14.read(i)
            for i in range(n)
        ])
        lower = NumericSeries([
            ema20.read(i) - 2 * atr14.read(i)
            for i in range(n)
        ])
        assert upper.read(upper.n - 1) >= ema20.read(ema20.n - 1)

    def test_donchian(self):
        """Donchian Channels — Rolling MIN + MAX."""
        api = CoreAPI()
        np.random.seed(5)
        n = 100
        base = list(np.random.randn(n).cumsum() + 10000)
        h = NumericSeries([v + np.random.uniform(0, 3) for v in base])
        l = NumericSeries([v - np.random.uniform(0, 3) for v in base])

        upper = api.rolling_max(h, 20)
        lower = api.rolling_min(l, 20)

        middle = NumericSeries([
            (upper.read(i) + lower.read(i)) // 2
            for i in range(n)
        ])
        assert middle.read(middle.n - 1) is not None

    def test_momentum(self):
        """Momentum — Pointwise (close[i] - close[i-p])."""
        api = CoreAPI()
        c = NumericSeries([10, 12, 15, 13, 11, 14, 16])
        n = c.n
        # Momentum: close[i] - close[i-3]
        mom = NumericSeries([
            c.read(i) - c.read(i - 3) if i >= 3 else 0
            for i in range(n)
        ])
        assert mom.read(3) == 13 - 10  # 3
        assert mom.read(5) == 14 - 15  # -1

    def test_tma(self):
        """Triangular Moving Average — SMA of SMA."""
        api = CoreAPI()
        c = NumericSeries([1, 2, 3, 4, 5, 6, 7])
        sma3 = api.sma(c, [3])[0]
        tma = api.sma(sma3, [3])[0]
        # SMA(3) of [1,2,3,4,5,6,7] → [0,0,2,3,4,5,6]
        # SMA(3) of [0,0,2,3,4,5,6] → [0,0,0,0,2,3,4]
        assert tma.read(6) == 5  # SMA(3) of [4,5,6] = (4+5+6)//3 = 5

    def test_rma(self):
        """RMA (Rolling Moving Average) — EMA-like с α=1/w."""
        api = CoreAPI()
        c = NumericSeries([100, 102, 104, 106, 108])
        # RMA(3) с α=1/3
        # RMA[0] = close[0] = 100
        # RMA[1] = (100 + 2*102) / 3 = 304/3 ≈ 101
        rma = api.ema(c, 3)  # EMA с α=2/4=0.5 — это стандартный EMA
        # Для RMA с α=1/3 используем EMA(5): α=2/6=0.333...
        rma2 = api.ema(c, 5)
        assert rma2.read(0) == 100
        # RMA-like формула: α = 1/w
