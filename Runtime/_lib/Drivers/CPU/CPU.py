"""CPU Driver — эталонная реализация драйвера NumFast.

CPU Driver:
- работает через пул блоков (MemoryManager)
- многопоточный (CpuThreadPool)
- моды видят только ExecutionContext
- архитектура повторяет будущие WebGPU/CUDA драйверы

Не является "упрощённой CPU-версией".
Является референсной реализацией всей архитектуры Runtime.
"""

from ._lib.cpu_driver import CpuDriver
from ._lib.thread_pool import CpuThreadPool
from ._lib.cpu_memory import CpuMemoryManager
from ._lib.worker_pool import CpuWorkerPool
