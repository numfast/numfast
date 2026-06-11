"""Тесты модуля Tables.

API возвращает dict[str, xp.ndarray] — каждая колонка
отдельный 1D-массив на GPU. Никакого pandas.
"""

try:
    import cupy as xp
except ImportError:
    import numpy as xp

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
