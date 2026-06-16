# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from _core.series_types import SeriesKind
from _core.series.base_series import BaseSeries


class TensorSeries(BaseSeries):
    """Заглушка: многомерные тензоры для AI/ML.

    Назначение: матрицы, тензоры, данные для машинного обучения.

    Реализация будет перенесена в отдельный AI/ML репозиторий.

    Статус: архитектурный каркас.

    Пример:
        t = TensorSeries(shape=(32, 3, 224, 224))
    """

    kind = SeriesKind.TENSOR

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
        print(f"TensorSeries (stub) — shape={self._shape}, ndim={self.ndim}")
