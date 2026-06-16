# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from enum import Enum


class SeriesKind(Enum):
    """Тип данных серии.

    Определяет, какую подсистему NumFast нужно использовать
    для работы с серией.
    """

    NUMERIC = "numeric"
    OBJECT = "object"
    TEXT = "text"
    IMAGE = "image"
    TENSOR = "tensor"

    @property
    def is_numeric(self) -> bool:
        return self == SeriesKind.NUMERIC

    def __str__(self):
        return self.value
