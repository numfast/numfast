"""CpuThreadPool — многопоточный пул для CPU Driver.

Не путать с Scheduler (планировщик волн зависимостей).
CpuThreadPool — исполнитель: берёт готовые пакеты и выполняет параллельно.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class CpuThreadPool:
    """Пул потоков для параллельного выполнения пакетов CPU Driver.

    Использует ThreadPoolExecutor. Не знает про Task, Job, ResourceRef.
    """

    def __init__(self, num_threads: Optional[int] = None):
        if num_threads is None:
            num_threads = os.cpu_count() or 4
            logger.info(f"  CpuThreadPool: using {num_threads} threads")
        self._num_threads = num_threads
        self._executor = ThreadPoolExecutor(max_workers=num_threads)

    @property
    def num_threads(self) -> int:
        return self._num_threads

    def map(self, func: Callable, items: list) -> list:
        """Параллельно применить func к каждому элементу items."""
        if len(items) <= 1:
            return [func(item) for item in items]
        return list(self._executor.map(func, items))

    def submit(self, func: Callable, *args, **kwargs):
        """Запустить одну задачу асинхронно."""
        return self._executor.submit(func, *args, **kwargs)

    def shutdown(self):
        """Завершить пул потоков."""
        self._executor.shutdown(wait=True)
