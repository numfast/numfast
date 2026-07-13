# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from abc import ABC, abstractmethod
from typing import Any

from _core.series_types import SeriesKind


class BaseSeries(ABC):
    kind: SeriesKind
    _shape: tuple[int, ...]
    _proxy: Any

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @abstractmethod
    def __len__(self) -> int:
        ...

    def info(self) -> None:
        raise NotImplementedError(
            f"info() не реализован для {type(self).__name__}"
        )

    def release(self) -> None:
        if self._proxy is not None:
            self._proxy.release()

    def _validate(self) -> None:
        if self._proxy is not None:
            self._proxy._validate()

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} kind={self.kind} "
            f"shape={self.shape} at {id(self):#x}>"
        )
