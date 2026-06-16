# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from Stats._lib.stats_lib import _get_moments, _mean, _var, _std, _clear_cache


def total(data) -> float:
    return _get_moments(data)["sum"]


def minimum(data) -> float:
    return _get_moments(data)["min"]


def maximum(data) -> float:
    return _get_moments(data)["max"]


def mean(data) -> float:
    return _mean(_get_moments(data))


def var(data, ddof: int = 0) -> float:
    return _var(_get_moments(data), ddof)


def std(data, ddof: int = 0) -> float:
    return _std(_get_moments(data), ddof)


def count(data) -> int:
    return _get_moments(data)["count"]


def range_(data) -> float:
    m = _get_moments(data)
    return m["max"] - m["min"]


def all_stats(data, ddof: int = 0) -> dict:
    m = _get_moments(data)
    return {
        "count": m["count"],
        "sum": m["sum"],
        "min": m["min"],
        "max": m["max"],
        "mean": _mean(m),
        "var": _var(m, ddof),
        "std": _std(m, ddof),
        "range": m["max"] - m["min"],
    }


def clear_cache():
    _clear_cache()
