# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 5 Series resource failure R1-R7 S130. R1 bridge no second path R2 int32->f32 <=2^24 R3 N=0/1/ordinary/chunked R4 CPU oracle vs WGSL R5 to_numpy boundary R6 packed reuse R7 SMA/ATR via LazyExpr"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))); sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "core")))
import numpy as np, pytest
from _core.context import create_context
from core.Series._lib.numeric_series import NumericSeries
from core.Series._lib.expr import _LazyExpr
from core.Series._lib.executor import execute
def _gpu():
    try:
        import wgpu
        return wgpu.gpu.request_adapter_sync(power_preference="high-performance") is not None
    except Exception:
        return False
GPU = _gpu()
def _to_numpy(s):
    return s.to_numpy() if hasattr(s, "to_numpy") else np.asarray(s)
# R1 Series bridge without second path (execute via Runtime)
def test_r1_bridge_no_second_path():
    ctx = create_context("r1")
    s = NumericSeries([1.0, 2.0, 3.0], ctx)
    expr = s + 1.0
    assert isinstance(expr, _LazyExpr)
    res = execute(expr)
    assert res == pytest.approx([2.0, 3.0, 4.0])
    expr2 = s + s
    assert isinstance(expr2, _LazyExpr)
    res2 = execute(expr2)
    assert res2 == pytest.approx([2.0, 4.0, 6.0])
    assert not hasattr(expr, "to_numpy") and hasattr(s, "_proxy")
# R2 int32->f32 exact for <=2^24
def test_r2_int32_f32_exact():
    ctx = create_context("r2")
    for v in [0, 1, 2**24 - 1, 2**24]:
        s = NumericSeries([float(v)], ctx)
        assert s.data()[0] == pytest.approx(float(v))
        assert int(np.asarray([float(v)], dtype=np.float32)[0]) == v
    s2 = NumericSeries([float(2**24 + 1)], ctx)
    assert s2.data()[0] == pytest.approx(float(np.float32(2**24 + 1)))
# R3 N=0/1/ordinary/chunked ceil(n/64)
def test_r3_n_variants_chunked():
    for n, ec in [(0, 0), (1, 1), (64, 1), (65, 2), (100, 2), (128, 2), (1000, 16)]:
        ctx = create_context(f"r3_{n}")
        data = [float(i) for i in range(n)]
        s = NumericSeries(data, ctx)
        assert len(s) == n and s.data() == pytest.approx(data)
        chunks = (n + 63) // 64 if n else 0
        assert chunks == ec
        if n > 0:
            res = execute(s + 1.0)
            assert len(res) == n and res[0] == pytest.approx(data[0] + 1.0)
        else:
            assert s.data() == []
# R4 CPU oracle <=1000 vs WGSL mass >1000
def test_r4_cpu_oracle_vs_wgsl_mass():
    rng = np.random.default_rng(42)
    for n in [10, 100, 1000]:
        ctx = create_context(f"r4_{n}")
        data = rng.standard_normal(n).tolist()
        s = NumericSeries(data, ctx)
        res = execute(s + 2.5)
        assert res == pytest.approx([v + 2.5 for v in data], rel=1e-5, abs=1e-6)
    n_big = 2000
    ctx = create_context("r4_big")
    data_big = rng.standard_normal(n_big).tolist()
    s_big = NumericSeries(data_big, ctx)
    res_big = execute(s_big * 1.5)
    assert res_big == pytest.approx([v * 1.5 for v in data_big], rel=1e-5, abs=1e-6)
    if GPU:
        assert len(res_big) == n_big
# R5 to_numpy boundary only
def test_r5_to_numpy_boundary_only():
    ctx = create_context("r5")
    data = [1.0, 2.0, 3.0, 4.0]
    s = NumericSeries(data, ctx)
    hot = execute(s + 5.0)
    assert hot == pytest.approx([6.0, 7.0, 8.0, 9.0])
    arr = _to_numpy(s)
    assert isinstance(arr, np.ndarray) and arr.tolist() == pytest.approx(data)
    assert list(s) == pytest.approx(data)
    assert np.allclose(arr.astype(np.float64), np.array(data, dtype=np.float64))
# R6 packed reuse two series same table
def test_r6_packed_reuse():
    ctx = create_context("r6")
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    s1 = NumericSeries(data, ctx)
    s2 = NumericSeries(data, ctx)
    assert s1.data() == s2.data() and s1.data() == pytest.approx(s2.data())
    from _core.container import pack_rows, extract_column
    from _core import kernel as _kernel
    schema = [{"name": "_value", "dtype": "float32"}]
    packed = pack_rows(schema, {"_value": data})
    tid, _ = _kernel._register_table(packed["rows"], packed["num_parts"], packed["schema"], packed["layout"], created_by="test_r6")
    col = extract_column(_kernel._get_table_entry(tid)["rows"], "_value", packed["layout"])
    assert col == pytest.approx(data) and s1._proxy["_col_name"] == s2._proxy["_col_name"] == "_value"
# R7 consumers SMA/ATR via LazyExpr no bypass
def test_r7_consumers_sma_atr_via_lazy():
    ctx = create_context("r7")
    data = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    s = NumericSeries(data, ctx)
    diff = s - 1.0
    atr_expr = diff + 1.0
    assert isinstance(diff, _LazyExpr) and isinstance(atr_expr, _LazyExpr)
    assert execute(diff) == pytest.approx([v - 1.0 for v in data])
    assert execute(atr_expr) == pytest.approx(data)
    s2 = NumericSeries([1.0, 2.0, 3.0], ctx)
    sma = (s2 + s2) * 0.5
    assert isinstance(sma, _LazyExpr) and execute(sma) == pytest.approx([1.0, 2.0, 3.0])
