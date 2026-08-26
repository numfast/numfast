# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from _core.series_types import SeriesKind
from .base_series import BaseSeries


class ImageSeries(BaseSeries):
    kind = SeriesKind.IMAGE

    def __init__(self, data=None, shape: tuple[int, ...] | None = None):
        self._proxy = None
        if shape is not None:
            self._shape = shape
        elif data is not None:
            self._shape = data.shape if hasattr(data, 'shape') else (len(data),)
        else:
            self._shape = (0,)

    def __len__(self) -> int:
        return self._shape[0]

    def info(self):
        print(f"ImageSeries (stub) — shape={self._shape}")
