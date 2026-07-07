"""CPU Driver — numpy backend.

Driver не знает операции.
Знает только read/write/dispatch/allocate.

Конституция I4: Driver не знает операции.
"""

from numfast.core.nseries import NumericSeries


class CpuDriver:
    """CPU Driver — numpy backend.

    Args:
        name: идентификатор драйвера ("cpu").
    """

    def __init__(self, name: str = "cpu"):
        self.name = name

    def read(self, series: NumericSeries, i: int) -> int:
        """Прочитать логическое значение.

        Args:
            series: NumericSeries.
            i: индекс (0..n-1).

        Returns:
            int32 значение (× scale).
        """
        return series.read(i)

    def write(self, series: NumericSeries, i: int, value: int) -> None:
        """Записать логическое значение.

        Args:
            series: NumericSeries.
            i: индекс (0..n-1).
            value: int32 значение (× scale).
        """
        series.write(i, value)

    def allocate(self, n: int, meta=None, offset=0, scale=1, valid_from=0) -> NumericSeries:
        """Выделить память под Series.

        Args:
            n: длина ряда.
            meta: SeriesMeta для копирования метаданных (приоритетнее offset/scale).
            offset: смещение (если нет meta).
            scale: множитель (если нет meta).
            valid_from: индекс первого полноценного значения.

        Returns:
            NumericSeries, заполненный единицами.
        """
        if meta is not None:
            from numfast.core.nseries import SeriesMeta
            # Accept both SeriesMeta and dict-like
            if hasattr(meta, 'offset'):
                offset = meta.offset
                scale = meta.scale
            else:
                offset = meta.get('offset', 0)
                scale = meta.get('scale', 1)

        return NumericSeries(
            data=[1] * n,  # 1 — минимальное значение по Storage Invariant
            offset=offset,
            scale=scale,
            valid_from=valid_from,
        )

    def dispatch(self, kernel: str, bindings: dict, workgroups: tuple) -> None:
        """Запустить kernel (CPU — no-op).

        Для CPU kernel выполняется синхронно в Python.
        `dispatch` нужен только для GPU (wgpu).

        Args:
            kernel: имя или WGSL-код.
            bindings: словарь {binding: буфер}.
            workgroups: (x, y, z) размер.
        """
        pass  # CPU — inline execution, dispatch не требуется

    def barrier(self) -> None:
        """Синхронизация (CPU — no-op)."""
        pass  # CPU — синхронно

    def __repr__(self) -> str:
        return f"CpuDriver(name='{self.name}')"
