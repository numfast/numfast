# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""Compatibility shim — реэкспорт из новой иерархии типов.

В будущем будет удалён после стабилизации Vis.
"""

from _core.series.numeric_series import (
    NumericSeries,
    make_series,
    series_add,
    series_sub,
    _series_data_proxy as series_data_proxy,
    _series_len as series_len,
)
