# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""Внутренняя фабрика серий — не экспортируется в публичный API.

Пользовательский код использует:
    from numfast import NumericSeries
    s = NumericSeries(data, ctx)

Внутренний код использует при необходимости:
    from _core.series_factory import make_series
"""

from _core.series.numeric_series import make_series, NumericSeries


__all__ = ["make_series", "NumericSeries"]
