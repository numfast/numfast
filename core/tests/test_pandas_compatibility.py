"""Phase 19 — Pandas Compatibility Suite.

Проверяет совместимость NumFast с pandas:
  Каждая операция NumFast даёт результат, математически эквивалентный
  float64-версии той же формулы, с погрешностью в пределах int32.

  NumFast:  int32  × MULTIPLIER  (цены)
  Pandas:   float64

  Сравнение: float_result = int32_result / MULTIPLIER

  Для RSI/STOCH: внутренний SCALE = 1000 (0..100000 = 0.0..100.0)
  Для ROC: ROC_SCALE = 100000 (100000 = 100%)
  Для CCI: CCI_SCALE = 1000
  Для EMA: EMA_SCALE = 1_000_000 (для alpha)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
import numpy as np
import pandas as pd
import pytest

from numfast.core.nseries import NumericSeries
from numfast.core.executor import CoreAPI


# ── Константы ──────────────────────────────────────────────

MULTIPLIER = 1000      # 3 десятичных знака
RSI_SCALE = 1000       # RSI/STOCH: 0..100000 = 0.0..100.0
ROC_SCALE = 100000     # ROC: 100000 = 100%
EMA_SCALE = 1_000_000  # EMA alpha precision
CCI_SCALE = 1000       # CCI values: -300000..+300000 typically

np.random.seed(42)


# ── Вспомогательные функции ────────────────────────────────

def to_int(series_float: np.ndarray) -> np.ndarray:
    """Convert float prices to int32 × MULTIPLIER."""
    return np.round(series_float * MULTIPLIER).astype(np.int32)


def to_float(series_int: np.ndarray) -> np.ndarray:
    """Convert int32 × MULTIPLIER back to float."""
    return series_int.astype(np.float64) / MULTIPLIER


def make_series(data_float: np.ndarray) -> NumericSeries:
    """Create NumericSeries from float data with proper int32 conversion."""
    return NumericSeries(to_int(data_float), offset=0, scale=1)


# ── Reference реализации (float64, точно повторяют NumFast kernels) ──

def ref_sma(prices: np.ndarray, w: int) -> np.ndarray:
    """SMA: prefix sum, same formula as _run_sma"""
    n = len(prices)
    result = np.zeros(n, dtype=np.float64)
    running = 0.0
    for i in range(n):
        running += prices[i]
        if i >= w:
            running -= prices[i - w]
        if i >= w - 1:
            result[i] = running / w  # float division (not floor)
    return result


def ref_ema(prices: np.ndarray, w: int) -> np.ndarray:
    """EMA: same formula as _run_ema"""
    n = len(prices)
    result = np.zeros(n, dtype=np.float64)
    alpha_f = 2.0 / (w + 1)
    result[0] = prices[0]
    for i in range(1, n):
        result[i] = alpha_f * prices[i] + (1 - alpha_f) * result[i - 1]
    return result


def ref_rsi(prices: np.ndarray, w: int) -> np.ndarray:
    """RSI: same formula as _run_rsi (Wilder's smoothing, α=1/w)"""
    n = len(prices)
    result = np.zeros(n, dtype=np.float64)
    alpha = 1.0 / w  # Wilder's
    
    if n < 2:
        return result
    
    gain = max(prices[1] - prices[0], 0)
    loss = max(prices[0] - prices[1], 0)
    avg_gain = gain
    avg_loss = loss
    denom = avg_gain + avg_loss
    result[0] = 50.0 if denom == 0 else 100.0 * avg_gain / denom
    
    for i in range(1, n):
        diff = prices[i] - prices[i-1]
        g = max(diff, 0)
        l_val = max(-diff, 0)
        avg_gain = alpha * g + (1 - alpha) * avg_gain
        avg_loss = alpha * l_val + (1 - alpha) * avg_loss
        denom = avg_gain + avg_loss
        result[i] = 50.0 if denom == 0 else 100.0 * avg_gain / denom
    
    return result


def ref_stoch(high, low, close, w: int) -> np.ndarray:
    """Stochastic %K: same formula as _run_stoch (clamped to [0,100])"""
    n = len(close)
    result = np.zeros(n, dtype=np.float64)
    
    for i in range(w - 1, n):
        hh = np.max(high[i - w + 1 : i + 1])
        ll = np.min(low[i - w + 1 : i + 1])
        denom = hh - ll
        if denom > 0:
            k = 100.0 * (close[i] - ll) / denom
            k = max(0, min(100.0, k))
        else:
            k = 50.0
        result[i] = k
    
    return result


def ref_median(prices: np.ndarray, w: int) -> np.ndarray:
    """Rolling median: lower middle for even w (matching _run_median)"""
    n = len(prices)
    result = np.zeros(n, dtype=np.float64)
    for i in range(w - 1, n):
        window = sorted(prices[i - w + 1 : i + 1])
        result[i] = window[w // 2]  # lower median
    return result


def ref_cci(high, low, close, w: int) -> np.ndarray:
    """CCI: same formula as _run_cci"""
    n = len(close)
    result = np.zeros(n, dtype=np.float64)
    
    for i in range(w - 1, n):
        # Typical Price
        tp = (high[i] + low[i] + close[i]) / 3.0
        
        # SMA of TP
        tp_window = (high[i - w + 1 : i + 1] + low[i - w + 1 : i + 1] + close[i - w + 1 : i + 1]) / 3.0
        sma_tp = np.mean(tp_window)
        
        # Mean Deviation
        md = np.mean(np.abs(tp_window - sma_tp))
        
        if md > 0:
            diff = tp - sma_tp
            result[i] = diff / (0.015 * md)
    
    return result


def ref_obv(close, volume) -> np.ndarray:
    """OBV: same formula as _run_obv (uses already-scaled close/volume)"""
    n = len(close)
    result = np.zeros(n, dtype=np.float64)
    obv = 0.0
    result[0] = obv
    for i in range(1, n):
        vol = int(max(volume[i], 1.0))
        if close[i] > close[i-1]:
            obv += vol
        elif close[i] < close[i-1]:
            obv -= vol
        result[i] = obv
    return result


def ref_vwap(close, volume) -> np.ndarray:
    """VWAP: same formula as _run_vwap"""
    n = len(close)
    result = np.zeros(n, dtype=np.float64)
    cum_pv = 0.0
    cum_vol = 0.0
    for i in range(n):
        pv = close[i] * max(volume[i], 1.0)
        cum_pv += pv
        cum_vol += max(volume[i], 1.0)
        result[i] = cum_pv / cum_vol
    return result


def ref_roc(prices: np.ndarray, period: int) -> np.ndarray:
    """ROC: same formula as _run_roc: (curr-prev)*100/prev"""
    n = len(prices)
    result = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        prev = prices[i - period]
        if prev > 0:
            result[i] = (prices[i] - prev) * 100.0 / prev
    return result


def ref_wma(prices: np.ndarray, w: int) -> np.ndarray:
    """WMA: same formula as _run_wma"""
    n = len(prices)
    result = np.zeros(n, dtype=np.float64)
    weight_sum = w * (w + 1) // 2
    for i in range(w - 1, n):
        total = 0.0
        for j in range(w):
            idx = i - w + 1 + j
            total += prices[idx] * (w - j)
        result[i] = total / weight_sum
    return result


def ref_trange(high, low, close) -> np.ndarray:
    """True Range: same formula as _run_trange"""
    n = len(high)
    result = np.zeros(n, dtype=np.float64)
    result[0] = max(high[0] - low[0], 0)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        result[i] = max(hl, hc, lc)
    return result


def ref_atr(high, low, close, w: int) -> np.ndarray:
    """ATR: TR + SMA, matching _run_atr formula"""
    tr = ref_trange(high, low, close)
    return ref_sma(tr, w)


def ref_stddev(prices: np.ndarray, w: int) -> np.ndarray:
    """STDDEV: population std, matching _run_rolling_stddev"""
    n = len(prices)
    result = np.zeros(n, dtype=np.float64)
    for i in range(w - 1, n):
        window = prices[i - w + 1 : i + 1]
        mean = np.mean(window)
        variance = np.mean((window - mean) ** 2)  # population variance
        result[i] = math.sqrt(variance)
    return result


# ── Вспомогательная функция сравнения ──────────────────────

def check_compatibility(nf_series: NumericSeries, ref_float: np.ndarray,
                        eps: float, label: str, from_idx: int = None):
    """Сравнить NumFast int32 результат с float64 reference.
    
    Args:
        nf_series: результат из CoreAPI (int32)
        ref_float: float64 reference массив
        eps: максимальный допуск
        label: для сообщения об ошибке
        from_idx: с какого индекса сравнивать (None = valid_from)
    """
    if from_idx is None:
        from_idx = nf_series.valid_from
    
    n = min(nf_series.n, len(ref_float))
    nf_vals = np.array([nf_series.read(i) for i in range(from_idx, n)])
    ref_slice = ref_float[from_idx:n]
    
    # Convert NumFast result to float
    nf_float = nf_vals.astype(np.float64) / MULTIPLIER
    
    diff = np.max(np.abs(nf_float - ref_slice))
    assert diff < eps, \
        f"{label}: max_diff={diff:.6f} > eps={eps}, " \
        f"nf[{from_idx}]={nf_float[0]:.4f}, ref[{from_idx}]={ref_slice[0]:.4f}"


# ── Тесты ──────────────────────────────────────────────────

class TestPandasSMA:
    @pytest.mark.parametrize("w", [3, 5, 14, 20, 50])
    def test_sma(self, w):
        n = 500
        prices = 100 + np.random.randn(n).cumsum()
        s = make_series(prices)
        api = CoreAPI()
        
        nf = api.sma(s, [w])[0]
        ref = ref_sma(prices, w)
        
        check_compatibility(nf, ref, eps=2.0/MULTIPLIER, label=f"SMA({w})")


class TestPandasEMA:
    @pytest.mark.parametrize("w", [5, 12, 14, 26, 50])
    def test_ema(self, w):
        n = 500
        prices = 100 + np.random.randn(n).cumsum()
        s = make_series(prices)
        api = CoreAPI()
        
        nf = api.ema(s, [w])[0]
        ref = ref_ema(prices, w)
        
        # EMA error compounds with recursion → larger epsilon
        check_compatibility(nf, ref, eps=15.0/MULTIPLIER, label=f"EMA({w})")


class TestPandasRSI:
    @pytest.mark.parametrize("w", [7, 14, 21])
    def test_rsi(self, w):
        n = 500
        prices = 100 + np.random.randn(n).cumsum()
        s = make_series(prices)
        api = CoreAPI()
        
        nf = api.rsi(s, [w])[0]
        ref = ref_rsi(prices, w)
        
        # RSI: NumFast stores 0..100000 (RSI_SCALE=1000), convert to 0..100
        from_idx = nf.valid_from
        n_comp = min(nf.n, len(ref))
        nf_vals = np.array([nf.read(i) for i in range(from_idx, n_comp)], dtype=np.float64)
        nf_float = nf_vals / RSI_SCALE  # RSI_SCALE = 1000
        ref_slice = ref[from_idx:n_comp]
        
        diff = np.max(np.abs(nf_float - ref_slice))
        # Allow 1.5 RSI points of error (int32 rounding)
        assert diff < 1.5, f"RSI({w}): max_diff={diff:.4f} RSI points"


class TestPandasWMA:
    @pytest.mark.parametrize("w", [3, 5, 14])
    def test_wma(self, w):
        n = 500
        prices = 100 + np.random.randn(n).cumsum()
        s = make_series(prices)
        api = CoreAPI()
        
        nf = api.wma(s, [w])[0]
        ref = ref_wma(prices, w)
        
        check_compatibility(nf, ref, eps=2.0/MULTIPLIER, label=f"WMA({w})")


class TestPandasSTDDEV:
    @pytest.mark.parametrize("w", [5, 14, 50])
    def test_stddev(self, w):
        n = 500
        prices = 100 + np.random.randn(n).cumsum()
        s = make_series(prices)
        api = CoreAPI()
        
        nf = api.rolling_stddev(s, w)
        ref = ref_stddev(prices, w)
        
        # STDDEV has sqrt + floor errors
        check_compatibility(nf, ref, eps=20.0/MULTIPLIER, label=f"STDDEV({w})")


class TestPandasMINMAX:
    @pytest.mark.parametrize("w", [5, 14, 50])
    @pytest.mark.parametrize("op", ["min", "max"])
    def test_minmax(self, w, op):
        n = 500
        prices = 100 + np.random.randn(n).cumsum()
        s = make_series(prices)
        api = CoreAPI()
        
        if op == "min":
            nf = api.rolling_min(s, w)
            ref = pd.Series(prices).rolling(w).min().values
        else:
            nf = api.rolling_max(s, w)
            ref = pd.Series(prices).rolling(w).max().values
        
        check_compatibility(nf, ref, eps=1.0/MULTIPLIER, label=f"{op.upper()}({w})")


class TestPandasROC:
    @pytest.mark.parametrize("p", [1, 5, 14])
    def test_roc(self, p):
        n = 500
        prices = 100 + np.random.randn(n).cumsum()
        s = make_series(prices)
        api = CoreAPI()
        
        nf = api.roc(s, p)
        ref = ref_roc(prices, p)
        
        # ROC: NumFast stores diff*ROC_SCALE//prev → equivalent to %
        from_idx = nf.valid_from
        n_comp = min(nf.n, len(ref))
        nf_vals = np.array([nf.read(i) for i in range(from_idx, n_comp)], dtype=np.float64)
        nf_pct = nf_vals / ROC_SCALE * 100  # ROC_SCALE=100000, convert to %
        ref_slice = ref[from_idx:n_comp]
        
        diff = np.max(np.abs(nf_pct - ref_slice))
        assert diff < 0.05, f"ROC({p}): max_diff={diff:.6f}%"


class TestPandasMEDIAN:
    @pytest.mark.parametrize("w", [5, 14, 50])
    def test_median(self, w):
        n = 500
        prices = 100 + np.random.randn(n).cumsum()
        s = make_series(prices)
        api = CoreAPI()
        
        nf = api.rolling_median(s, w)
        ref = ref_median(prices, w)
        
        check_compatibility(nf, ref, eps=2.0/MULTIPLIER, label=f"MEDIAN({w})")


class TestPandasTRANGE:
    @pytest.mark.parametrize("n_len", [100, 500])
    def test_trange(self, n_len):
        base = 100 + np.random.randn(n_len).cumsum()
        high = base + np.random.uniform(0, 5, n_len)
        low = base - np.random.uniform(0, 5, n_len)
        close = base
        
        h = make_series(high)
        l = make_series(low)
        c = make_series(close)
        api = CoreAPI()
        
        nf = api.trange(h, l, c)
        ref = ref_trange(high, low, close)
        
        check_compatibility(nf, ref, eps=2.0/MULTIPLIER, label=f"TRANGE(n={n_len})")


class TestPandasATR:
    @pytest.mark.parametrize("w", [7, 14, 21])
    def test_atr(self, w):
        n = 500
        base = 100 + np.random.randn(n).cumsum()
        high = base + np.random.uniform(0, 5, n)
        low = base - np.random.uniform(0, 5, n)
        close = base
        
        h = make_series(high)
        l = make_series(low)
        c = make_series(close)
        api = CoreAPI()
        
        nf = api.atr(h, l, c, [w])[0]
        ref = ref_atr(high, low, close, w)
        
        # ATR = SMA(TR) → compound error
        check_compatibility(nf, ref, eps=5.0/MULTIPLIER, label=f"ATR({w})")


class TestPandasVWAP:
    @pytest.mark.parametrize("n_len", [100, 500])
    def test_vwap(self, n_len):
        prices = 100 + np.random.randn(n_len).cumsum()
        volumes = np.abs(np.random.randn(n_len) * 1000 + 10000)
        
        c = make_series(prices)
        v = NumericSeries(np.round(volumes).astype(np.int32))
        api = CoreAPI()
        
        nf = api.vwap(c, v)
        ref = ref_vwap(prices, volumes)
        
        check_compatibility(nf, ref, eps=2.0/MULTIPLIER, label=f"VWAP(n={n_len})")


class TestPandasSTOCH:
    @pytest.mark.parametrize("w", [7, 14, 21])
    def test_stoch(self, w):
        n = 500
        base = 100 + np.random.randn(n).cumsum()
        high = base + np.random.uniform(0, 5, n)
        low = base - np.random.uniform(0, 5, n)
        close = base
        
        h = make_series(high)
        l = make_series(low)
        c = make_series(close)
        api = CoreAPI()
        
        nf = api.stoch(h, l, c, [w])[0]
        ref = ref_stoch(high, low, close, w)
        
        # STOCH: same scale as RSI
        from_idx = nf.valid_from
        n_comp = min(nf.n, len(ref))
        nf_vals = np.array([nf.read(i) for i in range(from_idx, n_comp)], dtype=np.float64)
        nf_float = nf_vals / RSI_SCALE
        ref_slice = ref[from_idx:n_comp]
        
        diff = np.max(np.abs(nf_float - ref_slice))
        assert diff < 1.5, f"STOCH({w}): max_diff={diff:.4f}%"


class TestPandasCCI:
    @pytest.mark.parametrize("w", [10, 14, 20])
    def test_cci(self, w):
        n = 500
        base = 100 + np.random.randn(n).cumsum()
        high = base + np.random.uniform(0, 5, n)
        low = base - np.random.uniform(0, 5, n)
        close = base
        
        h = make_series(high)
        l = make_series(low)
        c = make_series(close)
        api = CoreAPI()
        
        nf = api.cci(h, l, c, [w])[0]
        ref = ref_cci(high, low, close, w)
        
        # CCI: NumFast stores diff*SCALE*200//(3*md) where SCALE=1000
        from_idx = nf.valid_from
        n_comp = min(nf.n, len(ref))
        nf_vals = np.array([nf.read(i) for i in range(from_idx, n_comp)], dtype=np.float64)
        # NumFast CCI: int32 = CCI * CCI_SCALE where CCI_SCALE=1000
        nf_float = nf_vals / CCI_SCALE
        ref_slice = ref[from_idx:n_comp]
        
        mask = ~(np.isnan(ref_slice) | np.isinf(ref_slice))
        if mask.any():
            diff = np.max(np.abs(nf_float[mask] - ref_slice[mask]))
            assert diff < 5.0, f"CCI({w}): max_diff={diff:.4f}"


class TestPandasOBV:
    @pytest.mark.parametrize("n_len", [100, 500])
    def test_obv(self, n_len):
        prices = 100 + np.random.randn(n_len).cumsum()
        volumes = np.abs(np.random.randn(n_len) * 1000 + 10000)
        
        # Use same int32 data as NumFast
        close_i32 = to_int(prices)
        vol_i32 = np.round(volumes).astype(np.int32)
        
        c = make_series(prices)
        v = NumericSeries(vol_i32)
        api = CoreAPI()
        
        nf = api.obv(c, v)
        # Reference uses same int32 inputs for exact direction match
        ref = ref_obv(close_i32.astype(np.float64), vol_i32.astype(np.float64))
        
        # OBV values are raw volume counts, not scaled by MULTIPLIER
        # Compare int32 directly
        from_idx = 0
        n_comp = min(nf.n, len(ref))
        nf_vals = np.array([nf.read(i) for i in range(from_idx, n_comp)], dtype=np.float64)
        ref_slice = ref[from_idx:n_comp]
        
        diff = np.max(np.abs(nf_vals - ref_slice))
        # Allow rounding errors in volume × 1 count
        assert diff < n_len * 0.6, f"OBV(n={n_len}): max_diff={diff:.2f}"


class TestPandasBollinger:
    """Bollinger Bands: composed from SMA + STDDEV."""
    
    @pytest.mark.parametrize("w", [10, 20, 50])
    def test_bollinger(self, w):
        n = 500
        prices = 100 + np.random.randn(n).cumsum()
        s = make_series(prices)
        api = CoreAPI()
        
        # NumFast composition
        nf_sma = api.sma(s, [w])[0]
        nf_std = api.rolling_stddev(s, w)
        vf = max(nf_sma.valid_from, nf_std.valid_from)
        upper_vals = np.array([nf_sma.read(i) + 2 * nf_std.read(i) for i in range(n)])
        lower_vals = np.array([nf_sma.read(i) - 2 * nf_std.read(i) for i in range(n)])
        
        # Reference
        ref_sma_vals = ref_sma(prices, w)
        ref_std_vals = ref_stddev(prices, w)
        ref_upper = ref_sma_vals + 2 * ref_std_vals
        ref_lower = ref_sma_vals - 2 * ref_std_vals
        
        for label, nf_arr, ref_arr in [
            ("upper", upper_vals, ref_upper),
            ("lower", lower_vals, ref_lower),
        ]:
            nf_float = nf_arr[vf:].astype(np.float64) / MULTIPLIER
            ref_slice = ref_arr[vf:]
            diff = np.max(np.abs(nf_float - ref_slice))
            assert diff < 30.0/MULTIPLIER, \
                f"Bollinger({w}) {label}: max_diff={diff:.6f}"
