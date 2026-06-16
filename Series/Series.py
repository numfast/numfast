# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from Series._lib.series_lib import _to_array, _rolling, _normalize, _rolling_series, _normalize_series


def series_main(data: list | tuple) -> object:
    return _to_array(data)


def rolling_window(data: list | tuple, window: int = 3) -> object:
    return _rolling_series(list(data), window)


def normalize_series(data: list | tuple) -> object:
    return _normalize_series(list(data))
