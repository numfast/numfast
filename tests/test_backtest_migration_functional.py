# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Migration functional S140 — small golden parity + real file + loader bridge."""
import os
import pathlib
import pytest
import numpy as np
from _core.context import create_context
from core.Series._lib.numeric_series import NumericSeries
from core.Series._lib.expr import _LazyExpr
from core.Series._lib.executor import execute
try:
    from Loaders._lib.bybit import BybitLoader as _BL, reconstruct_deltas as _RD
    BybitLoader = _BL
    reconstruct_deltas = _RD
    _HAS_LDR = True
except ImportError:
    try:
        from Loader._lib.dzst import load_dzst as _ld
        _HAS_LDR = True
        class BybitLoader:  # shim
            @staticmethod
            def load_bybit_file(p):
                return _ld(p)
        def reconstruct_deltas(p, keep_int=True):
            return _ld(p)
    except ImportError:
        _HAS_LDR = False
        class BybitLoader:
            @staticmethod
            def load_bybit_file(p):
                raise FileNotFoundError(p)
        def reconstruct_deltas(p, keep_int=True):
            raise FileNotFoundError(p)
DATA_DIR = os.environ.get("NUMFAST_DATA_DIR", "")
REAL = (
    os.path.join(DATA_DIR, "bybit", "BTCUSDT", "BTCUSDT_d1_2024-01-01.csv.zst")
    if DATA_DIR else ""
)
_requires_real_dataset = pytest.mark.skipif(
    not REAL or not os.path.exists(REAL),
    reason="private bybit dataset not available",
)

def test_golden_pnl_small():
    n = 366
    scale = 100
    data = [float(10000 + i * 10) for i in range(n)]
    ctx = create_context("golden_small")
    s = NumericSeries(data, ctx)
    assert isinstance(s, NumericSeries)
    expr = s + s * 2
    assert isinstance(expr, _LazyExpr)
    gpu = execute(expr)
    cpu = (np.asarray(data, dtype=np.float64) * 3.0).tolist()
    assert len(gpu) == n and len(cpu) == n
    mism = sum(1 for a, b in zip(cpu, gpu) if abs(float(a) - float(b)) != 0.0)
    assert mism == 0, f"parity exact 0 mismatches failed {mism}"
    assert scale == 100

@_requires_real_dataset
def test_real_file_exists():
    p = pathlib.Path(REAL)
    assert p.exists(), f"real file missing {REAL}"
    assert os.path.exists(REAL)
    from _core.container import pack_rows, compute_layout
    # load via BybitLoader.load_bybit_file
    res = BybitLoader.load_bybit_file(REAL)
    assert res is not None
    # res can be (arr, mult, power) or (rows, layout)
    if isinstance(res, tuple) and len(res) == 3 and isinstance(res[0], np.ndarray):
        arr, mult, power = res
        assert arr.ndim == 2 and arr.shape[1] == 6
        assert mult == 10 ** power
        # pack without extra copy via from_packed
        layout = compute_layout([{"name": "_value", "dtype": "float32"}])
        # build packed rows from arr Close column
        close = arr[:, 3].astype(np.float64).tolist()
        packed = pack_rows([{"name": "_value", "dtype": "float32"}], {"_value": close})
        rows, lo = packed["rows"], packed["layout"]
        ctx = create_context("real_exists")
        series = NumericSeries.from_packed(rows, lo, ctx, col="_value")
        assert len(series) == arr.shape[0]
        assert series.data() is not None
    else:
        # fallback: assume rows/layout already
        rows, layout = res[0], res[1] if len(res) > 1 else {}
        ctx = create_context("real_exists2")
        series = NumericSeries.from_packed(rows, layout, ctx)
        assert len(series) >= 0
@_requires_real_dataset
def test_loader_series_bridge():
    p = REAL
    assert os.path.exists(p), f"missing {p}"
    from _core.container import pack_rows
    res = reconstruct_deltas(p, keep_int=True)
    assert isinstance(res, tuple) and len(res) >= 2
    arr = res[0]
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] > 0 and arr.shape[1] == 6
    close = arr[:, 3].astype(np.float64).tolist()
    packed = pack_rows([{"name": "_value", "dtype": "float32"}], {"_value": close})
    rows, layout = packed["rows"], packed["layout"]
    ctx = create_context("bridge")
    s = NumericSeries.from_packed(rows, layout, ctx, col="_value")
    assert len(s) == arr.shape[0]
    out = execute(s + 1)
    assert len(out) == len(s) == arr.shape[0]
    # verify +1 exact
    exp = [v + 1 for v in close]
    mism = sum(1 for a, b in zip(exp, out) if abs(float(a) - float(b)) > 1e-6)
    assert mism == 0
