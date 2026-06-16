# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from _core.backend import get_xp, set_active

xp = get_xp()

from Tables._lib.tables_lib import _to_dataframe, _merge, _aggregate


def test_to_dataframe_returns_dict_of_arrays():
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    result = _to_dataframe(data)
    assert isinstance(result, dict)
    assert "a" in result
    assert "b" in result
    assert isinstance(result["a"], xp.ndarray)
    assert result["a"].shape == (3,)


def test_merge_tables():
    left = {"id": [1, 2], "x": [10, 20]}
    right = {"id": [1, 2], "y": [100, 200]}
    l_arr = _to_dataframe(left)
    r_arr = _to_dataframe(right)
    result = _merge(l_arr, r_arr, on="id")
    assert "id" in result
    assert "x" in result
    assert "y" in result
    assert len(result["id"]) == 2


def test_aggregate_sum():
    data = {"cat": [1, 1, 2], "val": [10, 20, 30]}
    arr = _to_dataframe(data)
    result = _aggregate(arr, group_by="cat", agg="sum")
    assert "cat" in result
    assert "val" in result
    assert len(result["cat"]) == 2


def test_aggregate_mean():
    data = {"cat": [1, 1, 2], "val": [10, 20, 30]}
    arr = _to_dataframe(data)
    result = _aggregate(arr, group_by="cat", agg="mean")
    assert "cat" in result
    assert "val" in result
    mask1 = result["cat"] == 1
    mask2 = result["cat"] == 2
    assert xp.allclose(result["val"][mask1], xp.float64(15.0))
    assert xp.allclose(result["val"][mask2], xp.float64(30.0))


def test_aggregate_min_max():
    data = {"g": [1, 1, 2], "v": [5, 3, 9]}
    arr = _to_dataframe(data)
    r_min = _aggregate(arr, "g", "min")
    r_max = _aggregate(arr, "g", "max")
    assert len(r_min["v"]) == 2
    assert len(r_max["v"]) == 2


def test_aggregate_std():
    data = {"g": [1, 1, 1], "v": [1.0, 2.0, 3.0]}
    arr = _to_dataframe(data)
    result = _aggregate(arr, "g", "std")
    import math
    assert abs(result["v"][0] - math.sqrt(2.0 / 3.0)) < 1e-12


def test_aggregate_var():
    data = {"g": [1, 1], "v": [4.0, 6.0]}
    arr = _to_dataframe(data)
    result = _aggregate(arr, "g", "var")
    assert abs(result["v"][0] - 1.0) < 1e-12


def _wgpu_available() -> bool:
    try:
        set_active("wgpu")
        return True
    except Exception:
        return False


def test_aggregate_wgpu():
    if not _wgpu_available():
        import pytest
        pytest.skip("WebGPU not available")
    data = {"cat": [1, 1, 2], "val": [10, 20, 30]}
    arr = _to_dataframe(data)
    result = _aggregate(arr, group_by="cat", agg="sum")
    assert abs(result["val"][0] - 30.0) < 1e-4
    assert abs(result["val"][1] - 30.0) < 1e-4
    result_min = _aggregate(arr, group_by="cat", agg="min")
    assert abs(result_min["val"][0] - 10.0) < 1e-4
    result_max = _aggregate(arr, group_by="cat", agg="max")
    assert abs(result_max["val"][0] - 20.0) < 1e-4


# ---- Container tests ----

from _core import kernel as _kernel
from _core.container import compute_layout, pack_rows, extract_column


def test_compute_layout_single_column():
    schema = [{"name": "price", "dtype": "float32"}]
    layout = compute_layout(schema)
    assert layout["num_parts"] == 1
    c = layout["columns"]["price"]
    assert c["bit_offset"] == 0
    assert c["size"] == 32
    assert len(c["parts"]) == 1
    assert c["parts"][0] == (0, 0, 32)


def test_compute_layout_multi_column():
    schema = [
        {"name": "price", "dtype": "float32"},
        {"name": "rsi", "dtype": "scaled", "bit_width": 16,
         "scale": 0.01, "offset": 0.0, "min_int": -32768.0},
        {"name": "signal", "dtype": "enum", "bit_width": 2,
         "vocab": {0: "buy", 1: "sell"}, "null_slot": 2},
    ]
    layout = compute_layout(schema)
    assert layout["num_parts"] == 2  # 32+16+2 = 50 bits -> 2 parts
    c_price = layout["columns"]["price"]
    assert c_price["bit_offset"] == 0
    assert c_price["parts"][0] == (0, 0, 32)
    c_rsi = layout["columns"]["rsi"]
    assert c_rsi["bit_offset"] == 32
    assert c_rsi["parts"][0] == (1, 0, 16)
    c_sig = layout["columns"]["signal"]
    assert c_sig["bit_offset"] == 48
    assert c_sig["parts"][0] == (1, 16, 2)


def test_compute_layout_cross_boundary():
    schema = [{"name": "wide", "dtype": "scaled", "bit_width": 40,
               "scale": 1.0, "offset": 0.0, "min_int": 0.0}]
    layout = compute_layout(schema)
    assert layout["num_parts"] == 2
    c = layout["columns"]["wide"]
    assert c["size"] == 40
    assert len(c["parts"]) == 2
    assert c["parts"][0] == (0, 0, 32)  # bit 0-31 in part 0
    assert c["parts"][1] == (1, 0, 8)   # bit 32-39 in part 1 bits 0-7


def test_pack_rows_float32():
    schema = [{"name": "x", "dtype": "float32"}]
    packed = pack_rows(schema, {"x": [1.0, 2.5, -3.0]})
    assert packed["num_rows"] == 3
    assert packed["num_parts"] == 1
    extracted = extract_column(packed["rows"], "x", packed["layout"])
    for a, b in zip([1.0, 2.5, -3.0], extracted):
        assert abs(a - b) < 1e-6


def test_pack_rows_scaled():
    schema = [{"name": "val", "dtype": "scaled", "bit_width": 16,
               "scale": 0.01, "offset": 0.0, "min_int": -32768.0}]
    data_in = [10.0, 20.0, 30.0]
    packed = pack_rows(schema, {"val": data_in})
    extracted = extract_column(packed["rows"], "val", packed["layout"])
    for a, b in zip(data_in, extracted):
        assert abs(a - b) < 0.01


def test_container_table_roundtrip():
    _kernel.clear_all()
    schema = [
        {"name": "price", "dtype": "float32"},
        {"name": "rsi", "dtype": "scaled", "bit_width": 16,
         "scale": 0.01, "offset": 0.0, "min_int": -32768.0},
        {"name": "signal", "dtype": "enum", "bit_width": 2,
         "vocab": {0: "buy", 1: "sell"}, "null_slot": 2},
    ]
    data = {
        "price": [100.0, 101.5, 99.0],
        "rsi": [70.0, 55.0, 30.0],
        "signal": ["buy", "sell", "buy"],
    }
    from Tables._lib.tables_lib import _container_table_create, _get_column
    tbl = _container_table_create(schema, data)
    assert tbl["_num_rows"] == 3
    assert tbl["_num_parts"] == 2
    assert "price" in tbl["_layout"]["columns"]
    assert "rsi" in tbl["_layout"]["columns"]

    col_proxy = _get_column(tbl, "price")
    assert col_proxy["_col_name"] == "price"
    assert col_proxy["_table_id"] == tbl["_table_id"]
    assert col_proxy["_length"] == 3


def test_container_stats_float32():
    _kernel.clear_all()
    schema = [{"name": "val", "dtype": "float32"}]
    raw_vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    from Tables._lib.tables_lib import _container_table_create, _get_column
    tbl = _container_table_create(schema, {"val": raw_vals})
    col = _get_column(tbl, "val")
    from Stats.Stats import total, minimum, maximum, mean
    import numpy as np
    np_ref = np.array(raw_vals, dtype=np.float64)
    assert abs(total(col) - float(np.sum(np_ref))) < 1e-4
    assert abs(minimum(col) - float(np.min(np_ref))) < 1e-4
    assert abs(maximum(col) - float(np.max(np_ref))) < 1e-4
    assert abs(mean(col) - float(np.mean(np_ref))) < 1e-4


def test_container_stats_scaled():
    _kernel.clear_all()
    raw_vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    schema = [{"name": "val", "dtype": "scaled", "bit_width": 16,
               "scale": 0.01, "offset": 0.0, "min_int": -32768.0}]
    from Tables._lib.tables_lib import _container_table_create, _get_column
    tbl = _container_table_create(schema, {"val": raw_vals})
    col = _get_column(tbl, "val")
    from Stats.Stats import total, minimum, maximum, mean
    import numpy as np
    np_ref = np.array(raw_vals, dtype=np.float64)
    assert abs(total(col) - float(np.sum(np_ref))) / float(np.sum(np_ref)) < 0.01
    assert abs(minimum(col) - float(np.min(np_ref))) < 0.01
    assert abs(maximum(col) - float(np.max(np_ref))) < 0.01
    assert abs(mean(col) - float(np.mean(np_ref))) < 0.01


def test_container_stats_wgpu():
    if not _wgpu_available():
        import pytest
        pytest.skip("WebGPU not available")
    _kernel.clear_all()
    schema = [{"name": "val", "dtype": "float32"}]
    raw_vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    from Tables._lib.tables_lib import _container_table_create, _get_column
    tbl = _container_table_create(schema, {"val": raw_vals})
    col = _get_column(tbl, "val")
    from Stats.Stats import total, minimum, maximum, mean
    import numpy as np
    np_ref = np.array(raw_vals, dtype=np.float64)
    assert abs(total(col) - float(np.sum(np_ref))) < 1e-4
    assert abs(minimum(col) - float(np.min(np_ref))) < 1e-4
    assert abs(maximum(col) - float(np.max(np_ref))) < 1e-4
    assert abs(mean(col) - float(np.mean(np_ref))) < 1e-4


def test_container_multi_column_stats():
    _kernel.clear_all()
    schema = [
        {"name": "price", "dtype": "float32"},
        {"name": "rsi", "dtype": "scaled", "bit_width": 16,
         "scale": 0.01, "offset": 0.0, "min_int": -32768.0},
    ]
    data = {
        "price": [100.0, 101.5, 99.0, 102.0, 98.5],
        "rsi": [70.0, 55.0, 30.0, 65.0, 40.0],
    }
    from Tables._lib.tables_lib import _container_table_create, _get_column
    tbl = _container_table_create(schema, data)
    from Stats.Stats import total, mean
    import numpy as np
    price_proxy = _get_column(tbl, "price")
    np_price = np.array(data["price"], dtype=np.float64)
    assert abs(total(price_proxy) - float(np.sum(np_price))) < 1e-4
    assert abs(mean(price_proxy) - float(np.mean(np_price))) < 1e-4
    rsi_proxy = _get_column(tbl, "rsi")
    np_rsi = np.array(data["rsi"], dtype=np.float64)
    assert abs(total(rsi_proxy) - float(np.sum(np_rsi))) / float(np.sum(np_rsi)) < 0.01
    assert abs(mean(rsi_proxy) - float(np.mean(np_rsi))) < 0.01


# ---- Early + Late Compression tests ----

from _core import kernel as _kernel
from _core.container import compute_layout, pack_rows, extract_column


def test_early_compression_creates_tighter_layout():
    schema = [
        {"name": "price", "dtype": "float32",
         "compression": {"scaled": True, "bits": 12, "scale": 0.1, "offset": 0.0}},
    ]
    layout = compute_layout(schema)
    assert layout["num_parts"] == 1
    c = layout["columns"]["price"]
    assert c["size"] == 12
    assert c["parts"][0] == (0, 0, 12)
    assert c["meta"]["_is_compressed_float"]
    assert c["meta"]["scale"] == 0.1


def test_early_compression_pack_and_extract():
    schema = [
        {"name": "x", "dtype": "float32",
         "compression": {"scaled": True, "bits": 10, "scale": 0.01, "offset": 0.0}},
    ]
    # 10 bits, scale=0.01 => max representable = 1023*0.01 = 10.23
    raw = [5.0, 7.5, 10.0]
    packed = pack_rows(schema, {"x": raw})
    assert packed["num_parts"] == 1
    for row in packed["rows"]:
        assert row[0] < (1 << 10)
    extracted = extract_column(packed["rows"], "x", packed["layout"])
    for a, b in zip(raw, extracted):
        assert abs(a - b) < 0.01


def test_early_compression_multi_column():
    _kernel.clear_all()
    schema = [
        {"name": "price", "dtype": "float32",
         "compression": {"scaled": True, "bits": 12, "scale": 0.1, "offset": 100.0}},
        {"name": "volume", "dtype": "float32"},
    ]
    # 12 bits, scale=0.1, offset=100 => max = 4095*0.1+100 = 509.5
    data = {
        "price": [150.0, 200.0, 175.0],
        "volume": [1000.0, 2000.0, 1500.0],
    }
    from Tables._lib.tables_lib import _container_table_create, _get_column
    tbl = _container_table_create(schema, data)
    from Stats.Stats import total, mean
    import numpy as np
    vol_proxy = _get_column(tbl, "volume")
    np_vol = np.array(data["volume"], dtype=np.float64)
    assert abs(total(vol_proxy) - float(np.sum(np_vol))) < 1e-4
    price_proxy = _get_column(tbl, "price")
    np_price = np.array(data["price"], dtype=np.float64)
    assert abs(total(price_proxy) - float(np.sum(np_price))) / float(np.sum(np_price)) < 0.01


def test_dynamic_compress_float32_to_12bit():
    _kernel.clear_all()
    schema = [{"name": "val", "dtype": "float32"}]
    # Fit values within 12-bit range at scale=0.1: max=409.5
    raw = [50.0, 51.0, 52.0, 53.0, 54.0]
    from Tables._lib.tables_lib import _container_table_create, _get_column
    tbl = _container_table_create(schema, {"val": raw})
    from Tables._lib.tables_lib import _compress_column
    compressed_tbl = _compress_column(tbl, "val", target_bits=12,
                                      scale=0.1, offset=0.0)
    assert compressed_tbl["_num_parts"] <= tbl["_num_parts"]
    from Stats.Stats import total, minimum, maximum, mean
    import numpy as np
    np_ref = np.array(raw, dtype=np.float64)
    cp = _get_column(compressed_tbl, "val")
    assert abs(total(cp) - float(np.sum(np_ref))) / float(np.sum(np_ref)) < 0.01
    assert abs(minimum(cp) - float(np.min(np_ref))) < 0.01
    assert abs(maximum(cp) - float(np.max(np_ref))) < 0.01
    assert abs(mean(cp) - float(np.mean(np_ref))) < 0.01


def _wgpu_available_for_compression():
    try:
        from _core.backend import set_active, get_active_name
        set_active("wgpu")
        return get_active_name() == "wgpu"
    except Exception:
        return False


def test_dynamic_compress_wgpu():
    if not _wgpu_available_for_compression():
        import pytest
        pytest.skip("WebGPU not available")
    _kernel.clear_all()
    schema = [{"name": "val", "dtype": "float32"}]
    raw = [50.0, 51.0, 52.0, 53.0, 54.0]
    from Tables._lib.tables_lib import _container_table_create, _get_column
    tbl = _container_table_create(schema, {"val": raw})
    from Tables._lib.tables_lib import _compress_column
    compressed_tbl = _compress_column(tbl, "val", target_bits=12,
                                      scale=0.1, offset=0.0)
    from Stats.Stats import total, minimum, maximum, mean
    import numpy as np
    np_ref = np.array(raw, dtype=np.float64)
    cp = _get_column(compressed_tbl, "val")
    assert abs(total(cp) - float(np.sum(np_ref))) / float(np.sum(np_ref)) < 0.01
    assert abs(minimum(cp) - float(np.min(np_ref))) < 0.01
    assert abs(maximum(cp) - float(np.max(np_ref))) < 0.01
    assert abs(mean(cp) - float(np.mean(np_ref))) < 0.01


# ── Visualizer terminal test ──────────────────────────────────────────


def test_visualizer_cli_output():
    """Verify .info() CLI output contains expected sections (no crash)."""
    from _core import kernel
    from _core.context import create_context
    from _core.series import make_series
    from Tables._lib.tables_lib import _container_table_create
    import io

    kernel.clear_all()
    ctx = create_context("test")

    s = make_series([1.0, 2.0, 3.0], ctx)
    buf = io.StringIO()
    import sys
    old = sys.stdout
    sys.stdout = buf
    try:
        s.info()
    finally:
        sys.stdout = old
    out = buf.getvalue()
    assert "Series Info" in out
    assert "Standalone Series" in out
    assert "part0" in out
    assert "_value" in out

    kernel.clear_all()
    tbl = _container_table_create(
        [{"name": "a", "dtype": "float32"},
         {"name": "b", "dtype": "float32",
          "compression": {"scaled": True, "bits": 12, "scale": 0.1, "offset": 50.0}}],
        {"a": [1.0, 2.0], "b": [10.0, 20.0]},
    )
    buf = io.StringIO()
    sys.stdout = buf
    try:
        tbl.info()
    finally:
        sys.stdout = old
    out = buf.getvalue()
    assert "Table Info" in out
    assert "Multi-Column Dataframe" in out
    assert "part0" in out
    assert "part1" in out
    assert "Metrics" in out
