import random, math
import numpy as np
import pytest
from _core.context import create_context
from core.Series._lib.numeric_series import NumericSeries
from core.Series._lib.expr import _LazyExpr, _eval_operand
from core.Series._lib.executor import execute
from _core.backend import set_active as _sa; _sa("numpy")
SEED=42
NS=[0,1,2,63,64,65,256,1000,8192]
OPS=["add","sub","mul","truediv","pow","neg","abs","sin","cos"]
LIM=16777216
def _ctx():
    return create_context("fuzz")
def _data(rng,n):
    return [float(rng.uniform(-100,100)) for _ in range(n)]
def _series(data,ctx):
    return NumericSeries(data,ctx)
def _eval_cpu(expr):
    return expr._eval()
def _eval_gpu(expr,n):
    if n==0: return []
    return execute(expr)
def _cmp(cpu,gpu,op):
    assert len(cpu)==len(gpu)
    for c,g in zip(cpu,gpu):
        if isinstance(c,float) and math.isnan(c): assert math.isnan(g); continue
        if isinstance(c,float) and math.isinf(c): assert math.isinf(g); continue
        if isinstance(g,float) and math.isnan(g): assert math.isnan(c); continue
        if isinstance(g,float) and math.isinf(g): assert math.isinf(c); continue
        if abs(c)<=LIM:
            if abs(c-g)<=1e-5: continue
            d=max(abs(c),1e-12)
            assert abs(c-g)/d<=1e-6
        else:
            if c==0: assert abs(g)<=1e-6
            else: assert abs(c-g)/abs(c)<=1e-6
def _make_expr(op,sa,sb=None):
    if op=="add": return _LazyExpr("add",sa,sb)
    if op=="sub": return _LazyExpr("sub",sa,sb)
    if op=="mul": return _LazyExpr("mul",sa,sb)
    if op=="truediv": return _LazyExpr("truediv",sa,sb)
    if op=="pow": return _LazyExpr("pow",sa,2)
    if op=="neg": return _LazyExpr("neg",sa)
    if op=="abs": return _LazyExpr("abs",sa)
    if op=="sin": return _LazyExpr("sin",sa)
    if op=="cos": return _LazyExpr("cos",sa)
    raise ValueError(op)
def test_fuzz_1000():
    rng=random.Random(SEED)
    npr=np.random.default_rng(SEED)
    ctx=_ctx()
    cases=0
    for _ in range(1100):
        n=rng.choice(NS)
        op=rng.choice(OPS)
        d=_data(npr,n)
        if op=="truediv":
            d2=[float(v) if abs(v)>0.5 else 1.0 for v in _data(npr,n)]
            sa=_series(d,ctx); sb=_series(d2,ctx)
            e=_make_expr(op,sa,sb)
        elif op in ("add","sub","mul"):
            d2=_data(npr,n)
            sa=_series(d,ctx); sb=_series(d2,ctx)
            e=_make_expr(op,sa,sb)
        else:
            sa=_series(d,ctx)
            e=_make_expr(op,sa)
        _= _eval_operand(sa)
        cpu=_eval_cpu(e)
        gpu=_eval_gpu(e,n)
        _cmp(cpu,gpu,op)
        cases+=1
    assert cases>=1000
def test_fuzz_edge():
    rng=np.random.default_rng(SEED)
    ctx=_ctx()
    for n in NS:
        for op in OPS:
            d=[float(rng.uniform(-100,100)) for _ in range(n)]
            if op=="truediv":
                d2=[float(v) if abs(v)>0.5 else 2.0 for v in [float(rng.uniform(-100,100)) for _ in range(n)]]
                sa=_series(d,ctx); sb=_series(d2,ctx)
                e=_LazyExpr("truediv",sa,sb)
            elif op in ("add","sub","mul"):
                d2=[float(rng.uniform(-100,100)) for _ in range(n)]
                sa=_series(d,ctx); sb=_series(d2,ctx)
                e=_LazyExpr(op,sa,sb)
            elif op=="pow":
                sa=_series(d,ctx); e=_LazyExpr("pow",sa,2)
            elif op in ("neg","abs","sin","cos"):
                sa=_series(d,ctx); e=_LazyExpr(op,sa)
            else: continue
            cpu=e._eval()
            gpu=execute(e) if n!=0 else []
            assert len(cpu)==len(gpu)
            for c,g in zip(cpu,gpu):
                if abs(c)<=LIM: assert abs(c-g)<=1e-5 or abs(c-g)/max(abs(c),1e-12)<=1e-6
                else: assert abs(c-g)/max(abs(c),1e-12)<=1e-6
    assert True
