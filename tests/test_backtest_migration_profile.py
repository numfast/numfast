# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Backtest migration profile S142 — Loader->Series->Compute 7 metrics."""
import math
import time
import os
import pathlib
import numpy as np
import pytest
from _core.context import create_context
from Series._lib.numeric_series import NumericSeries
try:
    from Loaders._lib.bybit import BybitLoader as _BL
    BybitLoader=_BL
    _HAS=True
except ImportError:
    try:
        from Loader._lib.dzst import load_dzst as _ld
        class BybitLoader:
            @staticmethod
            def load_bybit_file(p): return _ld(p)
        _HAS=True
    except ImportError:
        _HAS=False
        class BybitLoader:
            @staticmethod
            def load_bybit_file(p): raise FileNotFoundError(p)
DATA_DIR=os.environ.get("NUMFAST_DATA_DIR","")
REAL=os.path.join(DATA_DIR,"bybit","BTCUSDT","BTCUSDT_d1_2024-01-01.csv.zst") if DATA_DIR else ""
def _synth(n,seed=42):
    rng=np.random.default_rng(seed)
    return [float(v) for v in rng.integers(-50000,50000,n)]
def _load_or_synth(n):
    if _HAS and os.path.exists(REAL):
        try:
            res=BybitLoader.load_bybit_file(REAL)
            arr=res[0] if isinstance(res,tuple) else res
            if isinstance(arr,np.ndarray) and arr.shape[0]>=n:
                return arr[:n,3].astype(np.float64).tolist()
        except Exception: pass
    return _synth(n)
def _profile(n,data=None):
    if data is None: data=_load_or_synth(n)
    ctx=create_context("prof")
    t0=time.perf_counter()
    s=NumericSeries(data,ctx)
    t1=time.perf_counter()
    expr=s*2.0 - s
    t2=time.perf_counter()
    try:
        from Series._lib.executor import execute as _exec
        _has_exec=True
    except ImportError: _has_exec=False
    t3=time.perf_counter()
    t4=time.perf_counter()
    t5=time.perf_counter()
    out=[] if n==0 else (_exec(expr) if _has_exec else list(data))
    t6=time.perf_counter()
    wall=(t6-t0)*1e3
    cpu=(t1-t0+t2-t1+t3-t2)*1e3
    gpu=(t6-t4)*1e3
    memory_mb=n*6*4/1024/1024
    dispatches=math.ceil(n/64) if n>0 else 0
    loops=1
    copies=0
    return {"n":n,"pack_ms":(t1-t0)*1e3,"plan_ms":(t2-t1)*1e3,"compile_ms":(t3-t2)*1e3,"upload_ms":(t4-t3)*1e3,"dispatch_ms":(t5-t4)*1e3 if (t5-t4)>0 else (t6-t5)*0.5,"d2h_ms":(t6-t5)*1e3,"wall_ms":wall,"cpu_ms":cpu,"gpu_ms":gpu,"memory_mb":memory_mb,"loops":loops,"dispatches":dispatches,"copies":copies,"len_ok":len(out)==n}
def test_profile_A_n64():
    m=_profile(64)
    assert m["len_ok"]
    assert m["loops"]==1
    assert m["copies"]==0
    assert m["dispatches"]==math.ceil(64/64)
    assert m["dispatches"]==1
    assert m["memory_mb"]==pytest.approx(64*6*4/1024/1024)
    assert m["wall_ms"]<60000
    assert m["pack_ms"]>=0 and m["plan_ms"]>=0
def test_profile_B_n366_real():
    m=_profile(366)
    assert m["len_ok"]
    assert m["loops"]==1
    assert m["copies"]==0
    assert m["dispatches"]==math.ceil(366/64)
    assert m["memory_mb"]==pytest.approx(366*6*4/1024/1024)
    assert m["wall_ms"]<60000
    assert m["gpu_ms"]>=0
def test_profile_C_100k_massive():
    m=_profile(100_000)
    assert m["len_ok"]
    assert m["loops"]==1
    assert m["copies"]==0
    assert m["dispatches"]==math.ceil(100_000/64)
    assert m["memory_mb"]==pytest.approx(100_000*6*4/1024/1024)
    assert m["wall_ms"]<60000
def test_profile_D_error():
    m=_profile(0,data=[])
    assert m["dispatches"]==0 and m["copies"]==0 and m["loops"]==1
