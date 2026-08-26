"""CreatedArray -- владелец результата creation-операций (S207).

Memory ownership: raw numpy + опциональная ссылка на GPU-буфер пула.
Aliasing НЕТ: каждый вызов api создаёт новый raw; view() в v1 запрещён.
Освобождение: явный .free() -> pool.release(raw); страховка __del__;
вытеснение -- существующий GpuBufferPool (refcount -> cache -> evict).
Никаких изменений GpuBufferPool/wgpu_driver (Drivers FROZEN).
"""

from dataclasses import dataclass, field

import numpy as np

_NP_DTYPE = {"float32": np.float32, "int32": np.int32}


@dataclass
class CreatedArray:
    """Результат creation-операции: numpy raw (+ gpu-буфер пула или None)."""

    raw: np.ndarray
    gpu_buffer: object | None
    shape: tuple
    dtype: str
    _pool: object | None = field(default=None, repr=False, compare=False)
    _released: bool = field(default=False, repr=False, compare=False)

    def free(self):
        """Явное освобождение: pool release + сброс ссылок."""
        if self._released:
            return
        if self._pool is not None and self.gpu_buffer is not None:
            self._pool.release(self.raw)
        self.gpu_buffer = None
        self._pool = None
        self.raw = None
        self._released = True

    def __len__(self):
        if self._released or self.raw is None:
            return 0
        return int(self.shape[0])

    def to_numpy(self) -> np.ndarray:
        """Копия raw (граница владения: наружу отдаём копию)."""
        if self._released or self.raw is None:
            return np.empty(0, dtype=_NP_DTYPE.get(self.dtype, np.float32))
        return np.array(self.raw, copy=True)

    def view(self):
        raise NotImplementedError("view() planned after D-3 (unfreeze Series)")

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass
