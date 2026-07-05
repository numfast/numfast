# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the Stats module.

Verifies:
- Each statistic returns a scalar
- One call computes everything — subsequent calls use cache
- clear_cache resets the cache
- all_stats returns the full report
- Edge cases: empty arrays, NaN, extreme values
"""

from Stats._lib.stats_lib import _compute, _get_moments, _clear_cache, MOMENTS, _compute_via_dispatcher, _merge_moments
from Stats.Stats import total, minimum, maximum, mean, var, std, count, range_, all_stats
from Series._lib import make_series
from _core.context import create_context
from _core import kernel
from _core.backend import set_active, get_active_name
import math


DATA = [1.0, 2.0, 3.0, 4.0, 5.0]


def test_total():
    assert total(DATA) == 15.0


def test_minimum():
    assert minimum(DATA) == 1.0


def test_maximum():
    assert maximum(DATA) == 5.0


def test_mean():
    assert mean(DATA) == 3.0


def test_count():
    assert count(DATA) == 5


def test_range():
    assert range_(DATA) == 4.0


def test_var_ddof0():
    assert abs(var(DATA, ddof=0) - 2.0) < 1e-12


def test_var_ddof1():
    assert abs(var(DATA, ddof=1) - 2.5) < 1e-12


def test_std_ddof0():
    assert abs(std(DATA, ddof=0) - math.sqrt(2.0)) < 1e-12


def test_std_ddof1():
    assert abs(std(DATA, ddof=1) - math.sqrt(2.5)) < 1e-12


def test_all_stats_full_report():
    report = all_stats(DATA)
    assert report["count"] == 5
    assert report["sum"] == 15.0
    assert report["min"] == 1.0
    assert report["max"] == 5.0
    assert report["mean"] == 3.0
    assert report["range"] == 4.0


def test_cache_single_compute():
    _clear_cache()
    key = id(DATA)
    assert key not in MOMENTS
    total(DATA)
    assert key in MOMENTS


def test_cache_reuses_result():
    _clear_cache()
    total(DATA)
    entries_after_first = len(MOMENTS)
    minimum(DATA)
    entries_after_second = len(MOMENTS)
    assert entries_after_second == entries_after_first


def test_cache_same_data_no_new_compute():
    _clear_cache()
    m1 = _get_moments(DATA)
    m2 = _get_moments(DATA)
    assert m1 is m2  # seconds call returns cached object
    assert len(MOMENTS) == 1
    # Calling via API also hits the same cache entry:
    total(DATA)
    assert len(MOMENTS) == 1  # no new entry added


def test_ddof_uses_cached_moments():
    _clear_cache()
    total(DATA)  # triggers _get_moments -> caches
    moments_count = len(MOMENTS)
    var(DATA, ddof=0)  # uses cached moments
    var(DATA, ddof=1)  # uses cached moments
    std(DATA, ddof=1)  # uses cached moments
    assert len(MOMENTS) == moments_count  # no new entries


def test_clear_cache_empties():
    total(DATA)
    assert len(MOMENTS) > 0
    _clear_cache()
    assert len(MOMENTS) == 0


def test_compute_matches_expected():
    result = _compute(DATA)
    assert result["count"] == 5
    assert result["sum"] == 15.0
    assert result["min"] == 1.0
    assert result["max"] == 5.0


# --- Edge cases ---


def test_empty_array():
    _clear_cache()
    empty: list[float] = []
    assert count(empty) == 0
    assert total(empty) == 0.0
    assert math.isnan(mean(empty))
    assert math.isnan(minimum(empty))
    assert math.isnan(maximum(empty))
    assert math.isnan(var(empty))
    assert math.isnan(std(empty))


def test_nan_in_data():
    _clear_cache()
    arr = [1.0, float("nan"), 3.0]
    s = total(arr)
    assert math.isnan(s)


def test_all_nan():
    _clear_cache()
    arr = [float("nan"), float("nan")]
    assert math.isnan(total(arr))
    assert math.isnan(minimum(arr))
    assert math.isnan(maximum(arr))


def test_single_element():
    _clear_cache()
    arr = [42.0]
    assert total(arr) == 42.0
    assert minimum(arr) == 42.0
    assert maximum(arr) == 42.0
    assert mean(arr) == 42.0
    assert count(arr) == 1
    assert range_(arr) == 0.0
    assert var(arr, ddof=0) == 0.0
    assert std(arr, ddof=0) == 0.0


def test_negative_values():
    _clear_cache()
    arr = [-5.0, -3.0, -1.0]
    assert total(arr) == -9.0
    assert minimum(arr) == -5.0
    assert maximum(arr) == -1.0
    assert mean(arr) == -3.0
    assert range_(arr) == 4.0


def test_extreme_values_no_overflow():
    _clear_cache()
    large = 1e12
    arr = [large, -large, large, -large]
    # sum should cancel out to 0, sum_sq should be manageable
    s = total(arr)
    assert abs(s) < 1e-6
    # std/variance should be meaningful
    v = var(arr, ddof=0)
    assert v > 0
    assert v < 1e26  # very rough upper bound


def test_very_large_values():
    _clear_cache()
    arr = [1e15, 2e15, 3e15]
    s = total(arr)
    m = mean(arr)
    assert abs(s - 6e15) < 1e9
    assert abs(m - 2e15) < 1e9


def test_all_identical():
    _clear_cache()
    arr = [7.0, 7.0, 7.0, 7.0]
    assert total(arr) == 28.0
    assert minimum(arr) == 7.0
    assert maximum(arr) == 7.0
    assert mean(arr) == 7.0
    assert range_(arr) == 0.0
    assert var(arr, ddof=0) == 0.0
    assert std(arr, ddof=0) == 0.0


def test_ddof_equals_n():
    _clear_cache()
    arr = [1.0, 2.0]
    # ddof == n --> variance should be 0
    assert var(arr, ddof=2) == 0.0
    assert std(arr, ddof=2) == 0.0


def test_ddof_greater_than_n():
    _clear_cache()
    arr = [1.0, 2.0]
    assert var(arr, ddof=5) == 0.0
    assert std(arr, ddof=5) == 0.0


def test_mixed_types():
    _clear_cache()
    arr = [1, 2, 3, 4, 5]
    assert total(arr) == 15
    assert mean(arr) == 3.0


def test_stats_raw_list_backward():
    _clear_cache()
    assert total([1.0, 2.0, 3.0]) == 6.0
    assert minimum([1.0, 2.0, 3.0]) == 1.0
    assert maximum([1.0, 2.0, 3.0]) == 3.0
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_stats_via_series_proxy():
    _clear_cache()
    set_active("numpy")  # честный CPU-тест, без WGSL
    kernel.clear_all()
    kernel.configure(chunk_size=1000)
    ctx = create_context("test")
    s = make_series([1.0, 2.0, 3.0, 4.0, 5.0], ctx)
    t = total(s)
    assert t == 15.0
    assert minimum(s) == 1.0
    assert maximum(s) == 5.0


def test_stats_via_series_proxy_wgsl():
    """WGSL-тест: если WGSL активен и падает — тест падает."""
    _clear_cache()
    kernel.clear_all()
    kernel.configure(chunk_size=1000)
    ctx = create_context("test")
    s = make_series([1.0, 2.0, 3.0, 4.0, 5.0], ctx)
    if get_active_name() == "wgpu":
        t = total(s)
        assert t == 15.0
        assert minimum(s) == 1.0
        assert maximum(s) == 5.0


def test_merge_moments_non_empty():
    a = {"count": 2, "sum": 3.0, "min": 1.0, "max": 2.0, "sum_sq": 5.0}
    b = {"count": 3, "sum": 12.0, "min": 3.0, "max": 5.0, "sum_sq": 50.0}
    m = _merge_moments(a, b)
    assert m["count"] == 5
    assert m["sum"] == 15.0
    assert m["min"] == 1.0
    assert m["max"] == 5.0
    assert m["sum_sq"] == 55.0


def test_merge_moments_empty_first():
    a = {"count": 0, "sum": 0.0, "min": float("nan"), "max": float("nan"), "sum_sq": 0.0}
    b = {"count": 3, "sum": 6.0, "min": 1.0, "max": 3.0, "sum_sq": 14.0}
    m = _merge_moments(a, b)
    assert m["count"] == 3
    assert m["sum"] == 6.0


def test_merge_moments_empty_second():
    a = {"count": 3, "sum": 6.0, "min": 1.0, "max": 3.0, "sum_sq": 14.0}
    b = {"count": 0, "sum": 0.0, "min": float("nan"), "max": float("nan"), "sum_sq": 0.0}
    m = _merge_moments(a, b)
    assert m["count"] == 3
