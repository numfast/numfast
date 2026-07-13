"""CpuWorkerPool — пул потоков для выполнения волн.

Вынесен из CpuDriver, чтобы:
- Driver не знал про ThreadPool
- Можно было заменить на другую стратегию (Static, Dynamic, NUMA, SIMD)
- Жизненный цикл CPU Driver совпадал с GPU Driver
"""

from concurrent.futures import ThreadPoolExecutor
from Runtime._lib.packet import ExecutionPacket
from Runtime._lib.scheduler import ScheduleWave


class CpuWorkerPool:
    """Пул потоков для параллельного выполнения волн.
    
    Получает готовые ExecutionPacket и функцию выполнения.
    Не знает про Task, Job, ResourceRef — только packet + executor.
    """
    
    def __init__(self, max_workers=None):
        self._max_workers = max_workers
    
    def execute_wave(self, wave: ScheduleWave, executor_fn):
        """Выполнить волну независимых пакетов параллельно.
        
        Args:
            wave: волна с независимыми пакетами
            executor_fn: callable(packet) → None — функция выполнения одного пакета
        """
        if len(wave.packets) <= 1:
            for p in wave.packets:
                executor_fn(p)
            return
        
        with ThreadPoolExecutor(max_workers=self._max_workers or min(32, len(wave.packets))) as pool:
            pool.map(executor_fn, wave.packets)
