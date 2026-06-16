# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""Series Type System — иерархия типов данных NumFast.

На текущем этапе полностью реализован только NumericSeries.
Остальные типы являются архитектурными заглушками будущих подсистем.

Публичный API:
    from _core.series import NumericSeries, ObjectSeries, ...
"""

from _core.series.numeric_series import NumericSeries, make_series, \
    series_add, series_sub, _series_data_proxy as series_data_proxy, \
    _series_len as series_len
from _core.series.object_series import ObjectSeries
from _core.series.text_series import TextSeries
from _core.series.image_series import ImageSeries
from _core.series.tensor_series import TensorSeries
from _core.series_types import SeriesKind

__all__ = [
    "NumericSeries",
    "ObjectSeries",
    "TextSeries",
    "ImageSeries",
    "TensorSeries",
    "SeriesKind",
    "make_series",
    "series_add",
    "series_sub",
    "series_len",
    "series_data_proxy",
]
