# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only


def total(data) -> float:
    from Stats._lib.stats_lib import _get_moments
    return _get_moments(data)["sum"]


def minimum(data) -> float:
    from Stats._lib.stats_lib import _get_moments
    return _get_moments(data)["min"]


def maximum(data) -> float:
    from Stats._lib.stats_lib import _get_moments
    return _get_moments(data)["max"]


def mean(data) -> float:
    from Stats._lib.stats_lib import _get_moments, _mean
    return _mean(_get_moments(data))


def var(data, ddof: int = 0) -> float:
    from Stats._lib.stats_lib import _get_moments, _var
    return _var(_get_moments(data), ddof)


def std(data, ddof: int = 0) -> float:
    from Stats._lib.stats_lib import _get_moments, _std
    return _std(_get_moments(data), ddof)


def count(data) -> int:
    from Stats._lib.stats_lib import _get_moments
    return _get_moments(data)["count"]


def range_(data) -> float:
    from Stats._lib.stats_lib import _get_moments
    m = _get_moments(data)
    return m["max"] - m["min"]


def all_stats(data, ddof: int = 0) -> dict:
    from Stats._lib.stats_lib import _get_moments, _mean, _var, _std
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
    from Stats._lib.stats_lib import _clear_cache
    _clear_cache()
