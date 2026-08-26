# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

import math
from _core.backend import get_xp, set_active

xp = get_xp()

from Series._lib.series_lib import _to_array, _rolling, _normalize, _rolling_series, _normalize_series


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


def test_rolling_series_list_output():
    result = _rolling_series([1, 2, 3, 4, 5], 3)
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == [1.0, 2.0, 3.0]
    assert result[1] == [2.0, 3.0, 4.0]
    assert result[2] == [3.0, 4.0, 5.0]


def test_rolling_series_window_too_large():
    result = _rolling_series([1, 2], 5)
    assert len(result) == 1
    assert result[0] == [1.0, 2.0]


def test_normalize_series_list_output():
    result = _normalize_series([10, 20, 30, 40, 50])
    assert isinstance(result, list)
    assert len(result) == 5
    for r, e in zip(result, [0.0, 0.25, 0.5, 0.75, 1.0]):
        assert abs(r - e) < 1e-6


def test_normalize_series_constant():
    result = _normalize_series([5, 5, 5])
    assert all(abs(v) < 1e-12 for v in result)


def test_normalize_series_single():
    result = _normalize_series([42])
    assert abs(result[0]) < 1e-12


def _wgpu_available() -> bool:
    try:
        set_active("wgpu")
        return True
    except Exception:
        return False


def test_rolling_wgpu():
    if not _wgpu_available():
        import pytest
        pytest.skip("WebGPU not available")
    result = _rolling_series([1, 2, 3, 4, 5], 3)
    assert len(result) == 3
    assert abs(result[0][0] - 1.0) < 1e-4
    assert abs(result[0][1] - 2.0) < 1e-4
    assert abs(result[0][2] - 3.0) < 1e-4
    assert abs(result[2][2] - 5.0) < 1e-4


def test_normalize_wgpu():
    if not _wgpu_available():
        import pytest
        pytest.skip("WebGPU not available")
    result = _normalize_series([10, 20, 30, 40, 50])
    for r, e in zip(result, [0.0, 0.25, 0.5, 0.75, 1.0]):
        assert abs(r - e) < 1e-4
