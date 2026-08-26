"""Driver ABC — контракт для всех драйверов.

Driver — исполнитель устройства.
Он не знает Task, Job, Compiler, ResourceRef, ExecutionPlan.
Он получает только ExecutionPacket.

Driver Interface:
  - compile(kernel)    — предварительная компиляция ядра (для GPU; CPU — no-op)
  - execute(packet)    — выполнить один пакет
  - execute_wave(wave) — выполнить волну независимых пакетов
  - wait()             — дождаться завершения всех операций
  - release()          — освободить ресурсы
  - memory             — MemoryManager (опционально)
  - store_output()     — сохранить выход задачи
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..scheduler import ScheduleWave


class Driver(ABC):
    """Базовый класс драйвера устройства.

    CPU, WebGPU, CUDA — все реализуют этот интерфейс.
    """

    def __init__(self, profiler=None):
        self.kernel_table: dict[str, dict] = {}
        self._profiler = profiler

    @property
    def profiler(self):
        return self._profiler

    @profiler.setter
    def profiler(self, p):
        self._profiler = p

    # ── Основной интерфейс ────────────────────────────────────────────

    def compile(self, kernel: str) -> Any:
        """Предварительная компиляция ядра.

        CPU: no-op, возвращает функцию.
        WebGPU: компилирует WGSL в шейдер.
        CUDA: компилирует CUDA C в PTX.
        """
        return None

    @abstractmethod
    def execute(self, packet: "ExecutionPacket"):
        """Выполнить один ExecutionPacket."""
        ...

    def execute_wave(self, wave: "ScheduleWave"):
        """Выполнить волну независимых пакетов.

        Базовая реализация — последовательно.
        CPU Driver переопределяет для параллельного выполнения.
        WebGPU — объединяет в один command buffer.
        """
        for p in wave.packets:
            self.execute(p)

    def wait(self):
        """Дождаться завершения всех операций (GPU sync)."""
        pass

    def release(self):
        """Освободить ресурсы драйвера."""
        pass

    # ── Управление памятью ────────────────────────────────────────────

    @property
    def memory(self) -> Optional["MemoryManager"]:
        """MemoryManager драйвера.

        None, если драйвер не использует MemoryManager (например, Remote Driver).
        """
        return None

    def store_output(self, name: str, task_id: int, output_idx: int, data: Any):
        """Сохранить выход задачи для последующего чтения.

        Базовая реализация — no-op.
        CPU Driver сохраняет выходы для resolve_output().
        """
        pass

    def max_dispatch_elements(self):
        """Максимальное число элементов в одном dispatch (для чанкования).

        None = нет ограничения dispatch (CPU и другие драйверы).
        GPU-драйверы переопределяют фактическим лимитом.
        """
        return None
