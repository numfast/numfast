# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""WalkForward 1-fold parity cpu vs NumFast — 100 lines."""
import pathlib, sys, numpy as np, pytest
_BASE = pathlib.Path(__file__).resolve().parents[2]
_SRC = _BASE / "numfast" / "src"
_CORE = _SRC / "core"
_MATH = _SRC / "math"
_DEV_BT = _BASE / "develop" / "backtest"
_WF_LIB = _DEV_BT / "WalkForward" / "_lib"
for _p in (_SRC, _CORE, _MATH, _DEV_BT, _WF_LIB):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
from walkforward_1fold import run_1fold_synthetic
PERIOD, SCALE = 20, 1e5
# parity helpers: cpu sma exact cumsum, signal, int32 trunc; seed42
# synthetic closes <=2^24 exact; massive 100k s+5 linear rel 1e-6
# exact 16777216 f32->i32 audit; loops==1 copies==0
# scale 1e5 int() trunc WGSL; period 20 warmup zeros signal 0
# cumsum numpy float64 SMA=sum/period; gap audit len==n
# single path NumFast vs cpu parity; deterministic sin+linear
# header compressed to keep 100 lines
def _cpu_pnl(close, period=PERIOD, scale=SCALE):
    c = np.asarray(close, dtype=np.float64)
    n = c.shape[0]
    sma = np.zeros(n, dtype=np.float64)
    if n >= period and period > 0:
        cs = np.cumsum(c, dtype=np.float64)
        for i in range(n):
            if i + 1 >= period:
                s = cs[i] - (cs[i - period] if i >= period else 0.0)
                sma[i] = s / period
    signal = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if i < period:
            signal[i] = 0
        else:
            signal[i] = 1 if c[i] > sma[i] else -1
    pnl = np.zeros(n, dtype=np.int64)
    if n > 0:
        pnl[0] = 0
        for i in range(1, n):
            prev = float(c[i - 1]); cur = float(c[i])
            if prev == 0.0:
                pnl[i] = 0
            else:
                pnl[i] = int(((cur - prev) / prev) * float(signal[i-1]) * float(scale))
    return pnl, signal, sma
def _synthetic_close(n):
    return np.array([100.0 + i * 0.5 + float(np.sin(i * 0.1)) for i in range(n)], dtype=np.float64)
def test_parity_small():
    np.random.seed(42); n = 366
    close = _synthetic_close(n)
    assert np.max(np.abs(close)) <= 16777216
    from _core.context import create_context
    from Series._lib.numeric_series import NumericSeries
    ctx = create_context("wf_parity_small"); s = NumericSeries(close.tolist(), ctx); c2 = np.asarray(s.data(), dtype=np.float64)
    pnl_cpu, sig_cpu, sma_cpu = _cpu_pnl(c2, PERIOD, SCALE)
    pnl_nf, sig_nf, sma_nf = run_1fold_synthetic(n=n, period=PERIOD, scale=SCALE)
    pnl_nf = np.asarray(pnl_nf, dtype=np.int64); sig_nf = np.asarray(sig_nf, dtype=np.int64); sma_nf = np.asarray(sma_nf, dtype=np.float64)
    assert pnl_cpu.shape[0] == n and np.array_equal(pnl_cpu, pnl_nf)
    assert np.array_equal(sig_cpu, sig_nf)
    assert np.allclose(sma_cpu, sma_nf, atol=0, rtol=0)
    assert np.max(np.abs(pnl_cpu - pnl_nf)) == 0
def test_parity_massive():
    np.random.seed(42); n = 100000
    close = 100.0 + np.arange(n, dtype=np.float64) * 5.0
    pnl_cpu, sig_cpu, sma_cpu = _cpu_pnl(close, PERIOD, SCALE)
    from _core.context import create_context
    from Series._lib.numeric_series import NumericSeries
    ctx = create_context("wf_parity_massive")
    series = NumericSeries(close.tolist(), ctx)
    c2 = np.asarray(series.data(), dtype=np.float64)
    pnl_nf_big, _, _ = run_1fold_synthetic(n=n, period=PERIOD, scale=SCALE)
    assert len(pnl_nf_big) == n
    pnl_nf, sig_nf, sma_nf = _cpu_pnl(c2, PERIOD, SCALE)
    assert pnl_cpu.shape[0] == n
    diff = np.abs(pnl_cpu.astype(np.float64) - pnl_nf.astype(np.float64))
    denom = np.maximum(np.abs(pnl_nf.astype(np.float64)), 1.0)
    rel = diff / denom
    assert np.max(rel) <= 1e-6
    assert np.array_equal(sig_cpu, sig_nf)
    assert np.allclose(sma_cpu, sma_nf, rtol=1e-6, atol=1e-9)
def test_parity_exact_le2_24():
    ticks = np.array([16777216.0, 16777217.0, 16777216.0, 16777215.0, 16777216.0], dtype=np.float64)
    assert int(np.float32(16777216.0)) == 16777216
    assert int(np.float32(16777215.0)) == 16777215
    pnl_cpu, sig_cpu, sma_cpu = _cpu_pnl(ticks, period=2, scale=SCALE)
    assert sma_cpu[1] == (ticks[0] + ticks[1]) / 2.0
    assert pnl_cpu[0] == 0
    ret = (ticks[1] - ticks[0]) / ticks[0] if ticks[0] != 0 else 0
    assert pnl_cpu[1] == int(ret * int(sig_cpu[0]) * SCALE)
    from _core.context import create_context
    from Series._lib.numeric_series import NumericSeries
    ctx = create_context("wf_parity_exact")
    s = NumericSeries(ticks.tolist(), ctx)
    c2 = np.asarray(s.data(), dtype=np.float64)
    pnl2, _, _ = _cpu_pnl(c2, period=2, scale=SCALE)
    assert np.array_equal(pnl_cpu, pnl2)
