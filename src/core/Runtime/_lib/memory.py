"""MemoryManager — управление блоками памяти.

Единственный компонент, который выделяет и освобождает память.
Моды НЕ имеют к нему доступа.

CPU Driver использует MemoryManager для:
  - выделения входных/выходных буферов
  - рабочей памяти (workspace)
  - переиспользования блоков
  - освобождения после выполнения

WebGPU/CUDA Drivers будут использовать ту же логику,
но с GPU-памятью вместо RAM.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class MemoryBlock:
    """Один блок памяти в пуле.

    Может быть свободен (free), выделен (allocated) или закреплён (pinned).
    Refcount управляет временем жизни:
      - 0 → свободен, может быть переиспользован
      - 1+ → используется, освобождается при release до 0
    """

    def __init__(self, block_id: int, data: Any, size: int, dtype: str = "float"):
        self.id = block_id
        self.data = data           # numpy array (CPU) or GPU buffer
        self.size = size           # количество элементов
        self.dtype = dtype
        self.refcount = 0
        self.pinned = False        # закреплён (не освобождать)

    # ── ABI: read/write/length (единый для всех MemoryBlock) ──────────

    def read(self, idx: int) -> float:
        """Прочитать элемент по индексу."""
        return float(self.data[idx])

    def write(self, idx: int, value):
        """Записать элемент по индексу."""
        self.data[idx] = value

    def length(self) -> int:
        """Количество элементов."""
        return self.size

    # ── Управление временем жизни ─────────────────────────────────────

    def acquire(self):
        """Увеличить счётчик использований."""
        self.refcount += 1

    def release(self) -> bool:
        """Уменьшить счётчик. Вернуть True, если блок свободен."""
        if self.refcount > 0:
            self.refcount -= 1
        return self.refcount == 0 and not self.pinned

    def __repr__(self):
        return f"Block({self.id}, size={self.size}, ref={self.refcount})"


class MemoryManager(ABC):
    """Абстрактный интерфейс менеджера памяти.

    Все реализации (CPU, WebGPU, CUDA) следуют этому контракту.
    """

    @abstractmethod
    def alloc(self, size: int, dtype: str = "float") -> MemoryBlock:
        """Выделить блок размером size элементов."""
        ...

    @abstractmethod
    def alloc_block(self, data: Any, name: str = "") -> MemoryBlock:
        """Создать блок из готовых данных (для входных буферов)."""
        ...

    @abstractmethod
    def release(self, block: MemoryBlock):
        """Освободить блок (уменьшить refcount)."""
        ...

    def free_all(self):
        """Освободить все блоки (для тестов). По умолчанию no-op."""
        pass

    @property
    def stats(self) -> dict:
        """Статистика пула. По умолчанию пусто."""
        return {}


class CpuMemoryPool(MemoryManager):
    """Пул блоков памяти на numpy.

    Блоки выделяются один раз и переиспользуются.
    Эталонная реализация MemoryManager.
    """

    def __init__(self, pool_size: int = 1024 * 1024 * 100):
        import numpy as np
        self._blocks: list[MemoryBlock] = []
        self._next_id = 0
        self._total_allocated = 0
        self._max_pool_size = pool_size
        self._np = np

    def alloc(self, size: int, dtype: str = "float") -> MemoryBlock:
        """Выделить блок размером size элементов.

        Сначала ищет свободный блок подходящего размера.
        Если не находит — создаёт новый.
        """
        # Ищем свободный блок достаточного размера
        for block in self._blocks:
            if block.refcount == 0 and not block.pinned and block.size >= size:
                block.acquire()
                return block

        # Создаём новый блок
        import numpy as np
        data = np.zeros(size, dtype=np.float64 if dtype == "float" else np.int64)
        block = MemoryBlock(self._next_id, data, size, dtype)
        self._next_id += 1
        self._blocks.append(block)
        block.acquire()
        self._total_allocated += size
        return block

    def alloc_block(self, data: Any, name: str = "") -> MemoryBlock:
        """Создать блок из готовых данных (для входных буферов)."""
        import numpy as np
        size = len(data) if hasattr(data, '__len__') else 1
        dtype = "float" if isinstance(data, np.ndarray) and data.dtype.kind == 'f' else "float"
        block = MemoryBlock(self._next_id, data, size, dtype)
        self._next_id += 1
        self._blocks.append(block)
        block.acquire()
        block.name = name
        return block

    def release(self, block: MemoryBlock):
        """Освободить блок (уменьшить refcount)."""
        is_free = block.release()
        # Блок остаётся в пуле для переиспользования
        return is_free

    def free_all(self):
        """Освободить все блоки (для тестов)."""
        for block in self._blocks:
            block.refcount = 0

    @property
    def stats(self) -> dict:
        """Статистика пула."""
        free = sum(1 for b in self._blocks if b.refcount == 0 and not b.pinned)
        used = sum(1 for b in self._blocks if b.refcount > 0)
        return {
            "total_blocks": len(self._blocks),
            "free": free,
            "used": used,
            "total_elements": self._total_allocated,
        }
