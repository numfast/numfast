"""CpuDriver — эталонная реализация Driver Interface.

Жизненный цикл совпадает с GPU Driver:
  alloc   → создание буферов (через CpuMemoryManager)
  bind    → привязка буферов к ExecutionPacket (выполнено до execute)
  dispatch → вызов cpu-функции ядра
  wait    → sync (no-op для CPU)
  release → освобождение ресурсов

CpuDriver НЕ знает про ThreadPool — многопоточность вынесена в CpuWorkerPool.
CpuDriver НЕ использует JIT — reference implementation прежде всего.
Любая оптимизация (Numba, SIMD, C++) может быть реализована как отдельный
драйвер или worker pool без изменения ABI.
"""

import logging
import time
from typing import Optional

from Runtime._lib.Drivers.base import Driver
from Runtime._lib.packet import ExecutionPacket
from Runtime._lib.scheduler import ScheduleWave
from .cpu_memory import CpuMemoryManager
from .worker_pool import CpuWorkerPool

logger = logging.getLogger(__name__)


class CpuDriver(Driver):
    """CPU Driver — эталонная реализация Driver Interface.
    
    Последовательность операций совпадает с WebGpuDriver/CudaDriver:
      1. compile(kernel)  — no-op (функция уже в kernel_table)
      2. execute(packet)  — alloc → bind → dispatch → readback
      3. execute_wave(wave) — параллельно через CpuWorkerPool
      4. wait()           — no-op (синхронно)
      5. release()        — очистка памяти
    """
    
    def __init__(self, profiler=None):
        super().__init__(profiler)
        self._memory_mgr = CpuMemoryManager()
        self._worker_pool = CpuWorkerPool(max_workers=None)
        self._output_store: dict = {}
    
    # ── MemoryManager (override from Driver ABC) ──────────────────────
    
    @property
    def memory(self) -> CpuMemoryManager:
        """MemoryManager драйвера (numpy-пул)."""
        return self._memory_mgr
    
    def store_output(self, name: str, task_id: int, output_idx: int, data):
        """Сохранить выход задачи."""
        self._output_store[name] = data
        self._output_store[(task_id, output_idx)] = data
        self._memory_mgr._outputs[name] = data
        self._memory_mgr._outputs[(task_id, output_idx)] = data
    
    # ── Основной интерфейс ────────────────────────────────────────────
    
    def compile(self, kernel: str):
        """'Компиляция' ядра — no-op для CPU.
        
        Функция cpu() уже зарегистрирована в kernel_table.
        Возвращаем имя ядра для единообразия с GPU Driver.
        """
        if kernel not in self.kernel_table:
            raise KeyError(f"Kernel '{kernel}' not found")
        if self.kernel_table[kernel]["drivers"]["cpu"] is None:
            raise ValueError(f"Kernel '{kernel}' has no CPU implementation")
        return kernel
    
    def execute(self, packet: ExecutionPacket):
        """Выполнить один ExecutionPacket.
        
        Этапы (совпадают с WebGPU):
          1. alloc     — буферы уже выделены Runtime._build_packet
          2. bind      — BlockView уже привязан к буферам
          3. dispatch  — вызов cpu-функции ядра
          4. readback  — данные уже в CPU-памяти (BlockView._raw)
        """
        entry = self.kernel_table.get(packet.kernel)
        if entry is None:
            raise KeyError(f"Kernel '{packet.kernel}' not found")
        
        cpu_fn = entry["drivers"].get("cpu")
        if cpu_fn is None:
            raise ValueError(f"Kernel '{packet.kernel}' has no CPU implementation")
        
        # Profiler: start
        prof = self._profiler
        if prof:
            prof.on_packet_start(packet)
        
        from Runtime._lib.mod_iface import ExecutionContext, Buffer
        
        ctx = ExecutionContext(
            inputs=[Buffer(view=bv.view, dtype=bv.dtype, size=bv.size) for bv in packet.input_buffers],
            outputs=[Buffer(view=bv.view, dtype=bv.dtype, size=bv.size) for bv in packet.output_buffers],
            workspace=[Buffer(view=bv.view, dtype=bv.dtype, size=bv.size) for bv in packet.workspace_buffers],
            uniforms=dict(packet.uniforms),
        )
        
        cpu_fn(ctx)
        
        # Profiler: finish
        if prof:
            prof.on_packet_finish(packet)
    
    def execute_wave(self, wave: ScheduleWave):
        """Выполнить волну независимых пакетов.
        
        Делегирует параллельное выполнение CpuWorkerPool.
        Worker pool не знает про Driver — только packet + executor_fn.
        """
        self._worker_pool.execute_wave(wave, self.execute)
    
    def wait(self):
        """Дождаться завершения — no-op для CPU (выполнение синхронно)."""
        pass
    
    def release(self):
        """Освободить ресурсы."""
        self._memory_mgr = CpuMemoryManager()
        self._output_store.clear()
    
    def resolve_output(self, name):
        """Найти выходной массив по имени (для тестов)."""
        raw = self._output_store.get(name)
        if raw is not None:
            return raw
        if hasattr(self._memory_mgr, '_outputs') and name in self._memory_mgr._outputs:
            return self._memory_mgr._outputs[name]
        return None
