"""Тесты модуля Series.

Проверяем, что API возвращает xp.ndarray,
а не list (никакого CPU-transfer).
"""

try:
    import cupy as xp
except ImportError:
    import numpy as xp

from Series._lib.series_lib import _to_array, _rolling, _normalize


def test_to_array_returns_xp_ndarray():
    arr = _to_array([1, 2, 3])
    assert isinstance(arr, xp.ndarray)
    assert arr.shape == (3,)
    assert arr.dtype == xp.float64


def test_rolling_returns_2d():
    arr = _to_array([1, 2, 3, 4, 5])
    result = _rolling(arr, 3)
    assert isinstance(result, xp.ndarray)
    assert result.shape == (3, 3)


def test_rolling_window_smaller_than_data():
    arr = _to_array([1, 2])
    result = _rolling(arr, 5)
    assert result.shape == (1, 2)


def test_normalize_series_values():
    arr = _to_array([10, 20, 30, 40, 50])
    result = _normalize(arr)
    expected = xp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert xp.allclose(result, expected)


def test_normalize_constant_series():
    arr = _to_array([5, 5, 5])
    result = _normalize(arr)
    assert xp.allclose(result, xp.zeros(3))
