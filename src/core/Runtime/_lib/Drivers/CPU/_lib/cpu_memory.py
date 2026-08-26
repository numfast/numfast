"""CpuMemoryManager — MemoryManager для CPU.

Расширяет базовый MemoryManager:
  - работает с numpy массивами
  - умеет создавать блоки из numpy
  - хранит выходы задач для последующего чтения
"""

import numpy as np
from typing import Optional, Any
from Runtime._lib.memory import CpuMemoryPool, MemoryBlock


class CpuMemoryManager(CpuMemoryPool):
    """MemoryManager для CPU.

    Добавляет:
      - alloc_array(size) -> numpy array (без создания MemoryBlock)
      - store_output(name, data, is_tmp) -> сохранить выход задачи
      - get_task_output(task_id, output_idx) -> прочитать выход задачи
      - resolve_output(name) -> прочитать выход по имени
      - resolve_output_by_id(task_id, output_idx) -> прочитать выход по id
      - release_task_output(task_id, output_idx) -> освободить
      - free_tmp_blocks() -> очистить временные блоки
    """

    def __init__(self, pool_size: int = 1024 * 1024 * 100):
        super().__init__(pool_size)
        self._outputs: dict = {}  # name -> array or (task_id, out_idx) -> array
        self._tmp_outputs: set = set()  # временные выходы

    def alloc_array(self, size: int, dtype=np.float64) -> np.ndarray:
        """Выделить numpy array (без обёртки MemoryBlock)."""
        return np.zeros(size, dtype=dtype)

    def store_output(self, name: str, task_id: int, output_idx: int,
                     data: np.ndarray, is_tmp: bool = True):
        """Сохранить выход задачи."""
        self._outputs[name] = data
        self._outputs[(task_id, output_idx)] = data
        if is_tmp:
            self._tmp_outputs.add(name)
            self._tmp_outputs.add((task_id, output_idx))

    def resolve_output(self, name: str) -> Optional[np.ndarray]:
        """Прочитать сохранённый выход по имени."""
        return self._outputs.get(name)

    def resolve_output_by_id(self, task_id: int, output_idx: int) -> Optional[np.ndarray]:
        """Прочитать выход задачи по id и индексу."""
        return self._outputs.get((task_id, output_idx))

    def get_task_output(self, task_id: int, output_idx: int) -> Optional[np.ndarray]:
        """Прочитать выход задачи по id и индексу."""
        key = (task_id, output_idx)
        return self._outputs.get(key)

    def release_task_output(self, task_id: int, output_idx: int):
        """Освободить выход задачи (если временный)."""
        key = (task_id, output_idx)
        name_key = None
        for k in list(self._outputs.keys()):
            if isinstance(k, str) and self._outputs.get(k) is self._outputs.get(key):
                name_key = k
                break
        if key in self._outputs:
            del self._outputs[key]
        if name_key in self._outputs:
            del self._outputs[name_key]
        self._tmp_outputs.discard(key)
        if name_key:
            self._tmp_outputs.discard(name_key)

    def free_tmp_blocks(self):
        """Очистить все временные блоки."""
        for key in list(self._tmp_outputs):
            if key in self._outputs:
                del self._outputs[key]
        self._tmp_outputs.clear()
