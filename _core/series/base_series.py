# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from abc import ABC, abstractmethod
from typing import Any

from _core.series_types import SeriesKind


class BaseSeries(ABC):
    """Абстрактная основа для всех типов серий NumFast.

    Определяет общий контракт: тип данных, размерность, длину,
    управление жизненным циклом и диагностику.

    На текущем этапе полностью реализован только NumericSeries.
    Остальные типы являются архитектурными заглушками будущих подсистем.
    """

    kind: SeriesKind
    _shape: tuple[int, ...]
    _proxy: Any  # _ProxyDict из kernel (только для реализованных типов)

    @property
    def ndim(self) -> int:
        """Количество измерений (1 для Numeric, 2+ для Image/Tensor)."""
        return len(self.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        """Размерность серии, numpy-совместимо.

        - Numeric:  (length,)
        - Image:    (H, W) или (H, W, C)
        - Tensor:   произвольно
        """
        return self._shape

    @abstractmethod
    def __len__(self) -> int:
        """Количество элементов в первом измерении."""
        ...

    def info(self) -> None:
        """Показать диагностическую информацию о серии."""
        raise NotImplementedError(
            f"info() не реализован для {type(self).__name__}"
        )

    def release(self) -> None:
        """Освободить ресурсы серии в Kernel."""
        if self._proxy is not None:
            self._proxy.release()

    def _validate(self) -> None:
        """Проверить живость proxy (RuntimeError если мёртв)."""
        if self._proxy is not None:
            self._proxy._validate()

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} kind={self.kind} "
            f"shape={self.shape} at {id(self):#x}>"
        )
