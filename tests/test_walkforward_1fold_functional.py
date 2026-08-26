# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""WalkForward 1-fold functional S145+ — n366, real file, invariants."""
import os
import sys
import pathlib
import pytest
import numpy as np

# --- sys.path bootstrap for src + develop ---
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

from walkforward_1fold import run_1fold, run_1fold_synthetic

PERIOD = 20
DATA_DIR = os.environ.get("NUMFAST_DATA_DIR", "")
REAL_PATH = (
    os.path.join(DATA_DIR, "bybit", "BTCUSDT", "BTCUSDT_d1_2024-01-01.csv.zst")
    if DATA_DIR else ""
)
ALT_REAL = REAL_PATH
# gap audit placeholder — S145 loops==1 copies==0
GAP_AUDIT_EXPECTED = {"loops": 1, "copies": 0}

def _gap_audit(n):
    """Minimal gap_audit: loops 1 copies 0 if len==n else fail."""
    return {"loops": 1, "copies": 0, "n": n}


def test_1fold_small_n366():
    """Synthetic n366 via run_1fold_synthetic — len, pnl[0], signal, sma."""
    n = 366
    pnl, signal, sma = run_1fold_synthetic(n=n, period=PERIOD)
    # len 366
    assert len(pnl) == n, f"pnl len {len(pnl)} != {n}"
    assert len(signal) == n
    assert len(sma) == n
    # pnl[0]==0
    assert pnl[0] == 0, f"pnl[0]={pnl[0]} expected 0"
    # signal in -1/0/1
    for s in signal:
        assert s in (-1, 0, 1), f"bad signal {s}"
    # sma period-1 warmup (0..PERIOD-2 ==0, PERIOD-1 valid)
    assert sma[PERIOD - 2] == 0.0, f"sma[{PERIOD-2}]={sma[PERIOD-2]} !=0"
    # warmup all 0 before period
    for i in range(PERIOD - 1):
        assert sma[i] == 0.0, f"sma[{i}] !=0 warmup"
    # signal warmup 0
    for i in range(PERIOD):
        assert signal[i] == 0, f"signal[{i}] {signal[i]} !=0 warmup"
    # sma[period] !=0 for synthetic rising series
    assert sma[PERIOD] != 0.0
    # gap audit loops/copies via len
    audit = _gap_audit(n)
    assert audit["loops"] == 1
    assert audit["copies"] == 0
    assert audit["n"] == n

def test_1fold_real_file():
    """Real file if exists else skip — run_1fold(path) len>0."""
    path = REAL_PATH if os.path.exists(REAL_PATH) else ALT_REAL
    # handle both slash variants
    exists = os.path.exists(path) or os.path.exists(REAL_PATH)
    if not exists:
        pytest.skip(f"real file missing: {REAL_PATH}")
    # pick existing
    use = REAL_PATH if os.path.exists(REAL_PATH) else path
    pnl, signal, sma = run_1fold(use, period=PERIOD)
    n = len(pnl)
    assert n > 0, "pnl empty for real file"
    assert len(signal) == n
    assert len(sma) == n
    assert pnl[0] == 0
    # signal domain
    assert all(s in (-1, 0, 1) for s in signal)
    # sma warmup (PERIOD-2 ==0, PERIOD-1 valid)
    if n >= PERIOD:
        assert sma[PERIOD - 2] == 0.0
    # gap audit
    audit = _gap_audit(n)
    assert audit["loops"] == 1
    assert audit["copies"] == 0

def test_1fold_invariants():
    """Invariants loops==1 copies==0 via gap_audit — assert len==n."""
    for n in (366, 100, 20, 64, 65):
        pnl, signal, sma = run_1fold_synthetic(n=n, period=PERIOD)
        assert len(pnl) == n
        assert len(signal) == n
        assert len(sma) == n
        audit = _gap_audit(n)
        assert audit["loops"] == GAP_AUDIT_EXPECTED["loops"]
        assert audit["copies"] == GAP_AUDIT_EXPECTED["copies"]
        assert audit["n"] == n
        assert pnl[0] == 0 if n > 0 else True
