"""Synthetic OHLC data generator for benchmarks."""

import numpy as np

_RNG = np.random.RandomState(42)

def generate_ohlc(n_bars: int, base_price: float = 50000.0, volatility: float = 0.02) -> dict:
    """Generate synthetic OHLC data.

    Args:
        n_bars: Number of bars to generate
        base_price: Starting price level
        volatility: Daily volatility

    Returns:
        dict with 'open', 'high', 'low', 'close' arrays (float64)
    """
    # Random walk close prices with momentum
    returns = _RNG.randn(n_bars) * volatility
    log_prices = np.zeros(n_bars)
    log_prices[0] = np.log(base_price)
    for i in range(1, n_bars):
        log_prices[i] = log_prices[i-1] + returns[i]

    close = np.exp(log_prices)

    # Generate OHLC from close
    daily_ranges = np.abs(_RNG.randn(n_bars)) * volatility * close

    high = close + daily_ranges * _RNG.uniform(0.3, 0.7, n_bars)
    low = close - daily_ranges * _RNG.uniform(0.3, 0.7, n_bars)
    open_ = np.zeros(n_bars)
    open_[0] = close[0]
    for i in range(1, n_bars):
        # Open is close of previous bar +/- noise
        open_[i] = close[i-1] * (1 + _RNG.randn() * volatility * 0.3)

    # Ensure high >= max(open, close) and low <= min(open, close)
    for i in range(n_bars):
        high[i] = max(high[i], open_[i], close[i])
        low[i] = min(low[i], open_[i], close[i])

    return {
        'open': open_.astype(np.float64),
        'high': high.astype(np.float64),
        'low': low.astype(np.float64),
        'close': close.astype(np.float64),
    }


def generate_expression_set(seed: int = 42, count: int = 100) -> list:
    """Generate a set of random expressions for benchmarking.

    Uses simple expressions of varying complexity:
    - Level 1: SMA(close, N), EMA(close, N)
    - Level 2: SMA + EMA, close - open
    - Level 3: Nested: SMA(EMA(close,N), M)
    - Level 4: Mixed: SMA*K + EMA, ATR*K + SMA

    Returns list of expression strings.
    """
    rng = np.random.RandomState(seed)
    exprs = []
    periods = [5, 7, 10, 14, 20, 21, 30, 50]
    consts = [1.0, 1.5, 2.0, 2.5]

    templates = []

    # Level 1: simple functions
    for p in periods[:4]:
        templates.append(f"SMA(close, {p})")
        templates.append(f"EMA(close, {p})")
    templates.append("ATR(high, low, close, 14)")

    # Level 2: binary
    for p in periods[:3]:
        templates.append(f"SMA(close, {p}) + EMA(close, {p+5})")
        templates.append(f"SMA(close, {p}) - SMA(high, {p})")
    templates.append("close - open")
    templates.append("high - low")

    # Level 3: nested
    for p in periods[:3]:
        for q in periods[1:4]:
            templates.append(f"SMA(EMA(close, {p}), {q})")
            templates.append(f"EMA(SMA(close, {p}), {q})")

    # Level 4: mixed with constants
    for p in periods[:3]:
        for c in consts[:2]:
            templates.append(f"SMA(close, {p}) * {c} + EMA(close, {p+5})")
            templates.append(f"ATR(high, low, close, 14) * {c} + SMA(close, {p})")

    # Level 5: compound
    templates.append("SMA(close, 10) + EMA(high, 20) - ATR(high, low, close, 14) * 1.5")
    templates.append("SMA(close, 5) * 2.0 - EMA(close, 20) / 1.5")
    templates.append("-(high - low) * SMA(close, 10)")
    templates.append("ATR(high, low, close, 7) * 1.5 + ATR(high, low, close, 14)")

    # Generate requested count by cycling templates and adding noise
    for i in range(count):
        expr = templates[i % len(templates)]
        exprs.append(expr)

    return exprs[:count]
