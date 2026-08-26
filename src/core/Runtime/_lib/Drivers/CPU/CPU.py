"""CPU Driver — эталонная реализация драйвера NumFast.

CPU Driver:
- работает через пул блоков (MemoryManager)
- многопоточный (CpuThreadPool)
- моды видят только ExecutionContext
- архитектура повторяет будущие WebGPU/CUDA драйверы
"""

def cpu_driver():
    """Create a CPU driver instance."""
    from _lib.cpu_driver import CpuDriver
    return CpuDriver()
