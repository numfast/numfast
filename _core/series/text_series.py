# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from _core.series_types import SeriesKind
from _core.series.base_series import BaseSeries


class TextSeries(BaseSeries):
    """Заглушка: строковые данные и NLP.

    Назначение: строки, токены, NLP-данные, текстовая аналитика.

    Поведение:
        Все вычислительные операции — NotImplementedError.

    Статус: архитектурная заглушка будущей подсистемы.
    """

    kind = SeriesKind.TEXT

    def __init__(self, data=None):
        self._proxy = None
        self._shape = (len(data) if data else 0,)

    def __len__(self) -> int:
        return self._shape[0]

    def info(self):
        print(f"TextSeries (stub) — {self._shape[0]} elements")
