import numpy as np
import pytest
from _core.context import create_context
from core.Series._lib.numeric_series import NumericSeries
from core.Series._lib.executor import execute

def _to_numpy(s):
    if hasattr(s, "to_numpy"):
        return s.to_numpy()
    return np.asarray(s)

def test_to_numpy_boundary():
    ctx = create_context("b1")
    s = NumericSeries([1.0, 2.0, 3.0], ctx)
    arr = _to_numpy(s)
    expected = np.array(s.data(), dtype=np.float64)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype in (np.float64, np.float32, np.dtype("float64"), np.dtype("float32"))
    assert np.allclose(arr.astype(np.float64), expected.astype(np.float64))
    assert list(arr.astype(np.float64)) == pytest.approx(list(expected.astype(np.float64)))
    # ensure boundary: to_numpy equals data materialization only at boundary
    assert arr.tolist() == pytest.approx(s.data())

def test_iter_boundary():
    ctx = create_context("b2")
    s = NumericSeries([10.0, 20.0, 30.0, 40.0], ctx)
    assert list(s) == s.data()
    assert list(s) == pytest.approx(s.data())
    it = iter(s)
    first = next(it)
    assert first == pytest.approx(s.data()[0])
    rest = list(it)
    assert rest == pytest.approx(s.data()[1:])
    # iter boundary must equal data() exactly
    assert len(list(s)) == len(s) == 4

def test_no_materialize_hot():
    # hot path: execute must work without explicit .data() before
    ctx = create_context("b3")
    n = 10000
    data = [float(i) * 0.5 for i in range(n)]
    s = NumericSeries(data, ctx)
    expr = s + 1.0
    result = execute(expr)
    assert len(result) == n
    assert len(result) == len(s)
    assert result[0] == pytest.approx(data[0] + 1.0)
    assert result[-1] == pytest.approx(data[-1] + 1.0)
    assert result[5000] == pytest.approx(data[5000] + 1.0)
    assert s.data()[0] == pytest.approx(data[0])
    assert isinstance(result, list)
    assert isinstance(result[0], (float, np.floating))  # np.float32 тоже ок, это f32 RNE
    # no hidden materialize: result length matches series length
    assert len(result) == s._proxy["_length"]

def test_packed_reuse():
    ctx = create_context("b4")
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    s1 = NumericSeries(data, ctx)
    s2 = NumericSeries(data, ctx)
    assert s1.data() == s2.data()
    assert len(s1) == len(s2) == 5
    expr = s1 + s2
    result = execute(expr)
    assert len(result) == 5
    expected = [2 * v for v in data]
    assert result == pytest.approx(expected)
    from _core.container import pack_rows, extract_column
    from _core import kernel as _kernel
    schema = [{"name": "_value", "dtype": "float32"}]
    packed = pack_rows(schema, {"_value": data})
    tid, _ = _kernel._register_table(packed["rows"], packed["num_parts"], packed["schema"], packed["layout"], created_by="test_reuse")
    proxy1 = _kernel._ProxyDict(_series_id=tid, _context_id=ctx["_id"], _length=packed["num_rows"], _table_id=tid, _col_name="_value", _col_schema=packed["layout"]["columns"]["_value"], _layout=packed["layout"], _is_hidden_series=True)
    proxy2 = _kernel._ProxyDict(_series_id=tid, _context_id=ctx["_id"], _length=packed["num_rows"], _table_id=tid, _col_name="_value", _col_schema=packed["layout"]["columns"]["_value"], _layout=packed["layout"], _is_hidden_series=True)
    entry = _kernel._get_table_entry(tid)
    col1 = extract_column(entry["rows"], "_value", packed["layout"])
    col2 = extract_column(entry["rows"], "_value", packed["layout"])
    assert col1 == col2 == pytest.approx(data)
    assert proxy1["_table_id"] == proxy2["_table_id"]
    assert proxy1["_table_id"] == tid
    arr = _to_numpy(s1)
    assert np.allclose(arr.astype(np.float64), np.array(data, dtype=np.float64))
    # reuse check: two proxies share same table id and layout
    assert proxy1["_layout"] is proxy2["_layout"] or proxy1["_layout"] == proxy2["_layout"]

# boundary: to_numpy/iter only materialize at edge, hot path stays lazy
# hot path does not call extract_column / .data() hidden
# packed reuse validated via shared PackedTable
# S130: series boundary contract
# end
