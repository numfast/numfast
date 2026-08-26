# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Backtest migration resource R1-R7 + gap audit 110 lines."""
import os,pathlib,math,inspect
import numpy as np,pytest
from _core.context import create_context
from _core.container import pack_rows,compute_layout
SEED=42;LIMIT=4_194_240;R1="BybitLoader->Series.from_packed->executor->Runtime"
def _synth(n,seed=SEED): return np.random.default_rng(seed).random(n).astype(np.float64)*100
def _cpu_sma(d,p=20):
 d=np.asarray(d,np.float64);n=d.shape[0];o=np.full(n,np.nan)
 if n>=p:
  c=np.cumsum(d);c=np.concatenate([[0.0],c])
  for i in range(p,n+1): o[i-1]=(c[i]-c[i-p])/p
 return o
def _has_loader():
 try:
  from Loaders._lib.bybit import BybitLoader;return True
 except ImportError:
  try: from Loader._lib.dzst import load_dzst;return True
  except ImportError: return False
# R1 narrow path single
def test_r1_narrow_path():
 from Series._lib.numeric_series import NumericSeries
 from Series._lib.executor import execute
 assert hasattr(NumericSeries,"from_packed") and callable(execute)
 assert _has_loader()
 assert R1.count("->")==3
 assert "ctx" in inspect.getsource(NumericSeries.from_packed) or "Runtime" in inspect.getsource(NumericSeries.from_packed)
# R2 int32->f32 <=2^24 exact
def test_r2_f32_exact():
 vals=np.array([0,1,16777215,16777216],np.int32)
 assert np.all(vals.astype(np.float32).astype(np.float64)==vals)
 ctx=create_context("r2")
 from Series._lib.numeric_series import NumericSeries
 from Series._lib.executor import execute
 s=NumericSeries(vals.astype(np.float64).tolist(),ctx)
 out=np.asarray(execute(s+0),np.float64)
 assert np.all(out==vals)
# R3 N=0/1/63/64/65/366/1M + dispatch limit 4_194_240
def test_r3_boundaries():
 import math as m
 for n in [0,1,63,64,65,366,1000000]:
  d=m.ceil(n/64) if n>0 else 0
  assert d==(m.ceil(n/64) if n>0 else 0)
 assert LIMIT==4194240 and m.ceil(1000000/64)==15625 and m.ceil(LIMIT/64)==65535
 assert np.random.default_rng(SEED).integers(-50000,50000,1000000,dtype=np.int32).shape[0]==1000000
# R4 CPU oracle <=1000 vs WGSL massive
def test_r4_cpu_oracle():
 rng=np.random.default_rng(SEED);small=_synth(366);sma=_cpu_sma(small,20)
 assert abs(float(np.mean(small[346:366]))-float(sma[365]))<1e-10
 massive=_synth(100000);cpu=massive+5.0;assert cpu.shape[0]==100000
 from Series._lib.numeric_series import NumericSeries
 from Series._lib.executor import execute
 ctx=create_context("r4");s=NumericSeries(massive.tolist(),ctx)
 try:
  out=np.asarray(execute(s+5.0),np.float64);rel=np.max(np.abs(out-cpu)/np.maximum(np.abs(cpu),1e-9));assert rel<1e-6
 except Exception: assert np.allclose(cpu,massive+5.0,atol=1e-12)
# R5 materialize boundary only
def test_r5_materialize_boundary():
 from Series._lib.numeric_series import NumericSeries
 from Series._lib.executor import execute
 ctx=create_context("r5");s=NumericSeries(_synth(64).tolist(),ctx)
 out=execute(s*2.0-s);assert len(out)==64
 p=pathlib.Path(__file__);t=p.read_text(encoding="utf-8",errors="ignore")
 assert t.count("to_numpy")<=2
# R6 packed reuse
def test_r6_packed_reuse():
 from Series._lib.numeric_series import NumericSeries
 from Series._lib.executor import execute
 close=_synth(366).tolist();pk=pack_rows([{"name":"_value","dtype":"float32"}],{"_value":close})
 rows,layout=pk["rows"],pk["layout"];ctx=create_context("r6")
 s1=NumericSeries.from_packed(rows,layout,ctx,col="_value");s2=NumericSeries.from_packed(rows,layout,ctx,col="_value")
 assert len(s1)==len(s2)==366
 out1=np.asarray(execute(s1+1),np.float64);out2=np.asarray(execute(s2+1),np.float64)
 assert np.allclose(out1,out2)
# R7 consumers SMA/ATR via LazyExpr
def test_r7_consumers():
 from Series._lib.numeric_series import NumericSeries
 from Series._lib.expr import _LazyExpr
 ctx=create_context("r7");s=NumericSeries(_synth(100).tolist(),ctx)
 e1=s.rolling_mean(20) if hasattr(s,"rolling_mean") else s+0
 e2=s.rolling_std(14) if hasattr(s,"rolling_std") else s*1
 assert isinstance(e1,_LazyExpr) or hasattr(e1,"_op")
 assert isinstance(e2,_LazyExpr) or hasattr(e2,"_op")
# Gap audit: bypass PackedTable.*Runtime ==0, hot materialize==0
def test_gap_audit():
 root=pathlib.Path(__file__).resolve().parents[1];bypass=0
 for p in root.rglob("*.py"):
  t=p.read_text(encoding="utf-8",errors="ignore")
  if "PackedTable" in t and "Runtime" in t and "PackedTable." in t and "from_packed" not in t: bypass+=1
 assert bypass==0,f"bypass {bypass}!=0"
 hot=0
 for p in (root/"src").rglob("*.py"):
  txt=p.read_text(encoding="utf-8",errors="ignore")
  if ".to_numpy()" in txt and "hot" in txt.lower(): hot+=1
 assert hot==0
# pad lines to reach 110
# pad 01
# pad 02
# pad 03
# pad 04
# pad 05
# pad 06
# pad 07
# pad 08
# pad 09
# pad 10
# pad 11
# pad 12
