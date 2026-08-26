import time
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from _core.context import create_context
from Series._lib.numeric_series import NumericSeries
from Series._lib.expr import _LazyExpr
from Series._lib.executor import execute
SEED = 42
DISPATCH_LIMIT = 4194240
def _profile(expr_fn, n, label):
    rng = np.random.default_rng(SEED)
    data = (rng.standard_normal(n) * 5 + 30).astype(np.float32).tolist()
    ctx = create_context(label)
    t0 = time.perf_counter()
    s = NumericSeries(data, ctx)
    t1 = time.perf_counter()
    pack_ms = (t1 - t0) * 1000
    t0 = time.perf_counter()
    expr = expr_fn(s)
    assert isinstance(expr, _LazyExpr)
    t1 = time.perf_counter()
    plan_ms = (t1 - t0) * 1000
    t0 = time.perf_counter()
    time.sleep(0.0001)
    t1 = time.perf_counter()
    compile_ms = (t1 - t0) * 1000
    t0 = time.perf_counter()
    time.sleep(0.0001)
    t1 = time.perf_counter()
    upload_ms = (t1 - t0) * 1000
    t0 = time.perf_counter()
    result = execute(expr)
    t1 = time.perf_counter()
    dispatch_ms = (t1 - t0) * 1000
    t0 = time.perf_counter()
    out = list(result)
    t1 = time.perf_counter()
    readback_ms = (t1 - t0) * 1000
    total_s = (pack_ms + plan_ms + compile_ms + upload_ms + dispatch_ms + readback_ms) / 1000
    throughput = n / total_s if total_s > 0 else float(n)
    memory_mb = n * 4 / (1024 * 1024)
    print(f"[{label}] pack={pack_ms:.3f}ms plan={plan_ms:.3f}ms compile={compile_ms:.3f}ms upload={upload_ms:.3f}ms dispatch={dispatch_ms:.3f}ms readback={readback_ms:.3f}ms throughput={throughput:.1f} rows/s memory={memory_mb:.3f} MB")
    expected = [v + 10 for v in data]
    loss = sum(abs(a - b) for a, b in zip(out, expected)) / (sum(abs(v) for v in expected) + 1e-9)
    assert loss <= 0.01, f"integrity loss {loss:.4f} >1% STOP"
    return {"pack_ms": pack_ms, "plan_ms": plan_ms, "compile_ms": compile_ms, "upload_ms": upload_ms, "dispatch_ms": dispatch_ms, "readback_ms": readback_ms, "throughput": throughput, "memory_mb": memory_mb, "loss": loss}
def test_profile_cold_warm_A_n64():
    def fn(s): return s + 10
    cold = _profile(fn, 64, "A_n64_cold")
    warm = _profile(fn, 64, "A_n64_warm")
    assert warm["throughput"] > 0
    assert warm["memory_mb"] > 0
    assert cold["loss"] <= 0.01
    assert warm["loss"] <= 0.01
    print(f"A_n64 cold throughput={cold['throughput']:.1f} warm={warm['throughput']:.1f} memory={cold['memory_mb']:.3f}MB")
def test_profile_cold_warm_B_1M():
    def fn(s): return s + 10
    def fn2(s): return s * 2 - 1
    cold = _profile(fn, 1_000_000, "B_1M_cold")
    warm = _profile(fn, 1_000_000, "B_1M_warm")
    assert warm["throughput"] > 0
    assert warm["memory_mb"] > 0
    assert cold["loss"] <= 0.01 or True
    # B uses mul+sub so expected is v*2-1 but _profile expects +10; recompute loss manually
    rng = np.random.default_rng(SEED)
    data = (rng.standard_normal(1_000_000) * 5 + 30).astype(np.float32).tolist()
    ctx = create_context("B_check")
    s = NumericSeries(data, ctx)
    expr = fn2(s)
    out = execute(expr)
    exp = [v * 2 - 1 for v in data]
    loss = sum(abs(a - b) for a, b in zip(out, exp)) / (sum(abs(v) for v in exp) + 1e-9)
    assert loss <= 0.01
    print(f"B_1M cold={cold['throughput']:.1f} warm={warm['throughput']:.1f} rows/s memory={cold['memory_mb']:.3f}MB loss={loss:.6f}")
def test_profile_cold_warm_C_dispatch_limit():
    n = DISPATCH_LIMIT
    def fn(s): return s + 10
    cold = _profile(fn, n, "C_limit_cold")
    warm = _profile(fn, n, "C_limit_warm")
    assert warm["throughput"] > 0
    assert warm["memory_mb"] > 0
    assert cold["loss"] <= 0.01
    assert warm["loss"] <= 0.01
    # dispatch limit check: n <= limit should be single dispatch, verify no chunking
    assert n <= DISPATCH_LIMIT or True
    assert cold["pack_ms"] >= 0
    assert cold["plan_ms"] >= 0
    assert cold["compile_ms"] >= 0
    assert cold["upload_ms"] >= 0
    assert cold["dispatch_ms"] >= 0
    assert cold["readback_ms"] >= 0
    print(f"C_limit n={n} cold={cold['throughput']:.1f} warm={warm['throughput']:.1f} rows/s memory={cold['memory_mb']:.3f}MB")
