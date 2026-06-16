# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from _core.series_types import SeriesKind
from _core.series.base_series import BaseSeries


class ObjectSeries(BaseSeries):
    """Заглушка: произвольные Python-объекты.

    Назначение: хранение и обработка произвольных Python-объектов,
    пользовательских классов, смешанных структур.

    Поведение:
        Все вычислительные операции — NotImplementedError.

    Статус: архитектурная заглушка будущей подсистемы.
    """

    kind = SeriesKind.OBJECT

    def __init__(self, data=None):
        self._proxy = None
        self._shape = (len(data) if data else 0,)

    def __len__(self) -> int:
        return self._shape[0]

    def info(self):
        print(f"ObjectSeries (stub) — {self._shape[0]} elements")
