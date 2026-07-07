"""NumericSeries — функция от индекса к значению.

Конституция:
  - read(i) → int32
  - write(i, value)
  - .meta — логические метаданные (offset, scale, valid_from, signed)
  - Не контейнер памяти. Не имеет .data, .to_numpy(), .restore().
  - Физическое хранение — тайна реализации.

I1: Операция не имеет права требовать материализации данных.
I2: Runtime не имеет права знать физический формат хранения.

Series Storage Invariant:
  min(read()) == 1  — любая серия нормализована так, что
  минимальное логическое значение всегда 1. Значение 0
  в вычислительных сериях не используется.

  Из этого следует:
    offset = min_real - 1
    real_value = read(i) + offset
    (только для отображения/экспорта, не для вычислений)

Series — это функция:
  Series(i) → value
  Не массив. Не контейнер. Не буфер.
"""

from copy import copy
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SeriesMeta:
    """Логические метаданные серии.

    Мод имеет доступ к этим метаданным через series.meta.
    Мод НЕ имеет доступа к физическому хранению (Packed, Raw, GPU, uint16).

    Attributes:
        offset: конверсия к реалу: real = read(i) + offset.
                Для отображения/экспорта.
        scale: множитель (10^power). Для отображения/экспорта.
        valid_from: индекс первого полноценного значения.
        signed: True если серия может содержать отрицательные значения.
    """
    offset: int = 0
    scale: int = 1
    valid_from: int = 0
    signed: bool = True


class NumericSeries:
    """Абстракция одномерного числового ряда int32.

    Series — это функция от индекса к значению.
    
    Все серии нормализованы: min(read()) == 1.
    Значение 0 в вычислительных сериях не встречается.

    Args:
        data: начальные значения. Нормализованы: min ≥ 1.
              После создания — доступ только через read()/write().
        offset: конверсия к реалу: real = read(i) + offset.
                Для отображения/экспорта. Не для вычислений.
        scale: множитель (10^power). Для отображения/экспорта.
        valid_from: индекс первого полноценного значения.
        signed: True если серия знаковая.
    """

    def __init__(
        self,
        data: np.ndarray,
        offset: int = 0,
        scale: int = 1,
        valid_from: int = 0,
        signed: bool = True,
    ):
        # Внутреннее хранение — numpy (может быть заменено на StorageBuffer,
        # Packed, Delta, mmap — API не меняется)
        self._values = np.asarray(data, dtype=np.int32).copy()
        self._meta = SeriesMeta(
            offset=int(offset),
            scale=int(scale),
            valid_from=int(valid_from),
            signed=bool(signed),
        )

    # ── Публичный API ─────────────────────────────────────

    def read(self, i: int) -> int:
        """Логическое значение серии в позиции i.

        Значение нормализовано: min ≥ 1.
        Это ИСТИННОЕ значение серии — не требует конверсии.
        Только для отображения/экспорта нужно real = read(i) + offset.

        Args:
            i: индекс (0..n-1). Выход за границы — undefined (не краш).

        Returns:
            int32 значение (нормализованное, min ≥ 1).
        """
        if i < 0 or i >= len(self._values):
            return 0  # undefined, но не краш
        return int(self._values[i])

    def write(self, i: int, value: int) -> None:
        """Записать значение в позицию i.

        Args:
            i: индекс (0..n-1).
            value: int32 значение (нормализованное, min ≥ 1).
        """
        if i < 0 or i >= len(self._values):
            return  # silently ignore (undefined)
        self._values[i] = np.int32(value)

    # ── Метаданные ────────────────────────────────────────

    @property
    def meta(self) -> SeriesMeta:
        """Логические метаданные серии.

        Мод ЧИТАЕТ meta для создания новых серий с совместимыми параметрами.
        Мод НЕ ИЗМЕНЯЕТ meta напрямую.

        Returns:
            SeriesMeta (offset, scale, valid_from, signed).
        """
        return copy(self._meta)

    @property
    def valid_from(self) -> int:
        """Индекс первого полноценного значения."""
        return self._meta.valid_from

    @property
    def offset(self) -> int:
        """Конверсия к реалу: real = read(i) + offset.
        Только для отображения/экспорта. Не для вычислений."""
        return self._meta.offset

    @property
    def scale(self) -> int:
        """Множитель (10^power). Только для отображения/экспорта."""
        return self._meta.scale

    @property
    def signed(self) -> bool:
        """True если серия знаковая."""
        return self._meta.signed

    @property
    def n(self) -> int:
        """Длина ряда (количество элементов)."""
        return len(self._values)

    # ── Запрещённые операции (I1) ─────────────────────────

    @property
    def data(self) -> None:
        """Запрещено (I1). Операция не имеет права на materialization."""
        raise RuntimeError(
            "NumericSeries.data — запрещено (I1). "
            "Используй series.read(i)."
        )

    def to_numpy(self) -> None:
        """Запрещено (I1). Операция не имеет права на materialization."""
        raise RuntimeError(
            "NumericSeries.to_numpy() — запрещено (I1). "
            "Используй series.read(i)."
        )

    # ── Внутреннее (только для Runtime/Driver) ────────────

    def _raw(self) -> np.ndarray:
        """Доступ к буферу — ТОЛЬКО для Driver.

        Driver — часть Core Runtime, не Mod.
        Mod'ам доступ только через read()/write().
        """
        return self._values

    def __repr__(self) -> str:
        return (
            f"NumericSeries(n={self.n}, "
            f"offset={self._meta.offset}, scale={self._meta.scale}, "
            f"valid_from={self._meta.valid_from})"
        )
