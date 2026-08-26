# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""test_phase5_series_real — real Colossus Series operations via LazyExpr+execute."""
import pytest
import numpy as np
from _core.context import create_context
from Series._lib.numeric_series import NumericSeries
from Series._lib.expr import _LazyExpr
from Series._lib.executor import execute


def test_sma_via_lazy():
    """SMA without RollingSum: (a+b+c)/3 via LazyExpr."""
    ctx = create_context("test_sma_via_lazy")
    n = 100
    data_a = [float(i) for i in range(n)]
    data_b = [float(i + 1) for i in range(n)]
    data_c = [float(i + 2) for i in range(n)]
    a = NumericSeries(data_a, ctx)
    b = NumericSeries(data_b, ctx)
    c = NumericSeries(data_c, ctx)
    # SMA as add+div: average of three shifted series
    sma = (a + b + c) / 3
    assert isinstance(sma, _LazyExpr)
    result = execute(sma)
    assert isinstance(result, list)
    assert len(result) == n
    # (x + x+1 + x+2)/3 == x+1
    expected = [float(i + 1) for i in range(n)]
    for r, e in zip(result, expected):
        assert abs(r - e) < 1e-5


def test_ema_chain():
    """EMA chain via sequential LazyExpr mul+add."""
    ctx = create_context("test_ema_chain")
    n = 64
    price_data = [float(100 + i * 0.5) for i in range(n)]
    price = NumericSeries(price_data, ctx)
    alpha = 0.2
    # ema step: e = price*alpha + price*(1-alpha) chained 5 times
    e = price
    for _ in range(5):
        e = e * (1 - alpha) + price * alpha
    assert isinstance(e, _LazyExpr)
    result = execute(e)
    assert isinstance(result, list)
    assert len(result) == n
    # after infinite chain e should converge to price exactly
    for r, p in zip(result, price_data):
        assert abs(r - p) < 1e-4


def test_atr_style():
    """ATR-style TR via high/low/close multiple NumericSeries + LazyExpr."""
    ctx = create_context("test_atr_style")
    n = 50
    high_data = [float(10 + i + 2) for i in range(n)]
    low_data = [float(10 + i - 2) for i in range(n)]
    close_data = [float(10 + i) for i in range(n)]
    high = NumericSeries(high_data, ctx)
    low = NumericSeries(low_data, ctx)
    close = NumericSeries(close_data, ctx)
    # TR simplified: (high-low) and abs(high-close), abs(low-close) via LazyExpr
    hl = high - low
    hc = high - close
    lc = low - close
    # combine via add and mul to simulate ATR aggregation
    tr = hl + hc * 0.5 + lc * 0.5
    # also test neg/abs chain
    tr_abs = tr + (-tr) * 0  # identity, keeps LazyExpr chain
    assert isinstance(tr, _LazyExpr)
    assert isinstance(tr_abs, _LazyExpr)
    result = execute(tr)
    result2 = execute(tr_abs)
    assert len(result) == n
    assert len(result2) == n
    # hl =4, hc=2, lc=-2 => tr=4+1-1=4
    for v in result:
        assert abs(v - 4.0) < 1e-5
    for v in result2:
        assert abs(v - 4.0) < 1e-5
# padding
# padding
# padding
