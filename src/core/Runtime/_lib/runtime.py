"""Runtime — верхний уровень исполнительного слоя NumFast.

Pipeline:
  Compiler.compile(jobs) -> Task list
  planner(tasks) -> ExecutionGraph
  builder(graph) -> ExecutionPackets
  Driver.execute_wave(wave)

Правило: Runtime НЕ создаёт память.
Вся память — через Driver.
"""

import time as _time
from typing import Optional
from .scheduler import Scheduler, ScheduleWave
from .packet import ExecutionPacket, BufferView as BV
from .mod_iface import BlockView

# Runtime version
RUNTIME_VERSION = "1.0.0"
KERNEL_ABI_VERSION = 1
EXTENSION_API_VERSION = 1


def _alloc_driver_array(driver, size: int, dtype="float"):
    """Выделить массив через MemoryManager драйвера или fallback.

    Driver владеет памятью. Runtime только запрашивает.
    """
    mm = getattr(driver, 'memory', None)
    if mm is not None and hasattr(mm, 'alloc_array'):
        return mm.alloc_array(size)
    import numpy as np
    return np.zeros(size, dtype=np.float64 if dtype == "float" else np.int64)


class Runtime:
    """Основной оркестратор."""

    def __init__(self, driver: Optional["Driver"] = None, optimizer_level: int = 2,
                 scheduler_max_elements: Optional[int] = None):
        self.kernel_table: dict[str, dict] = {}
        self._driver = driver
        self._scheduler = Scheduler()
        self._optimizer_level = optimizer_level
        self._scheduler_max_elements = scheduler_max_elements
        # Version info
        self.version = RUNTIME_VERSION
        self.kernel_abi_version = KERNEL_ABI_VERSION
        self.extension_api_version = EXTENSION_API_VERSION

    def register_kernel(self, alias, describe=None, cpu=None, wgsl=None, cuda=None, abi_version=1, capabilities=None):
        self.kernel_table[alias] = {
            "describe": describe,
            "drivers": {
                "cpu": cpu,
                "wgsl": wgsl,
                "cuda": cuda,
            },
            "abi_version": abi_version,
            "capabilities": capabilities or {},
        }

    def load_extension(self, extension_module):
        """Загрузить внешнее расширение.

        Расширение — Python-модуль с функцией register_all(runtime).

        Args:
            extension_module: модуль Python, имеющий register_all(runtime)

        Raises:
            TypeError: если модуль не имеет register_all
            RuntimeError: если версии несовместимы
        """
        if not hasattr(extension_module, 'register_all'):
            raise TypeError(
                f"Extension {extension_module.__name__} has no register_all(runtime)"
            )

        if hasattr(extension_module, 'EXTENSION_META'):
            meta = extension_module.EXTENSION_META
            req_runtime = meta.get('min_runtime_version', '0.0.0')
            req_kernel_abi = meta.get('kernel_abi_version', 0)
            req_ext_api = meta.get('extension_api_version', 0)

            if req_runtime > self.version:
                raise RuntimeError(
                    f"Extension requires Runtime >= {req_runtime}, "
                    f"current is {self.version}"
                )
            if req_kernel_abi > self.kernel_abi_version:
                raise RuntimeError(
                    f"Extension requires Kernel ABI >= {req_kernel_abi}, "
                    f"current is {self.kernel_abi_version}"
                )
            if req_ext_api > self.extension_api_version:
                raise RuntimeError(
                    f"Extension requires Extension API >= {req_ext_api}, "
                    f"current is {self.extension_api_version}"
                )

        extension_module.register_all(self)

    def compile(self, jobs: list[dict]) -> list["Task"]:
        from .compiler import compile as _compile
        return _compile(jobs, self.kernel_table)

    def execute(self, tasks: list["Task"], source_data=None):
        """Выполнить список задач через ExecutionGraph pipeline.

        1. planner(tasks) -> ExecutionGraph
        2. builder(graph) -> list[ExecutionPacket]
        3. Driver.execute_wave() для каждого пакета
        """
        if self._driver is None:
            from .Drivers.CPU._lib.cpu_driver import CpuDriver
            self._driver = CpuDriver()

        self._driver.kernel_table = self.kernel_table

        # Step 1: Task list -> ExecutionGraph
        from .planner import planner
        graph = planner(tasks, self.kernel_table)

        # Step 2: Optimize graph
        if self._optimizer_level > 0:
            from .optimizer import optimize_graph
            graph = optimize_graph(graph, level=self._optimizer_level)

        # Step 3: ExecutionGraph -> ExecutionPackets
        # Step 3a: ExecutionScheduler — чанкование больших elementwise-графов
        from .execution_scheduler import ExecutionScheduler
        sch = ExecutionScheduler()
        sch_packets = sch.execute(
            graph, source_data, self._driver, self.kernel_table,
            max_elements=self._scheduler_max_elements,
        )
        if sch_packets is None:
            # Обычный путь: builder + поочерёдное исполнение волн
            from .builder import builder as _builder
            packets = _builder(graph, source_data, self._driver, self.kernel_table)

            # Step 4: Execute via Driver
            for packet in packets:
                sw = ScheduleWave(packets=[packet])
                self._driver.execute_wave(sw)

    @property
    def optimizer_level(self):
        return self._optimizer_level

    @optimizer_level.setter
    def optimizer_level(self, level):
        self._optimizer_level = level

    @property
    def driver(self):
        return self._driver

    @driver.setter
    def driver(self, d):
        self._driver = d


import logging
logger = logging.getLogger(__name__)


class ProfiledRuntime(Runtime):
    """Runtime с ExecutionProfiler на уровне Driver."""

    def __init__(self, driver=None, optimizer_level=2):
        from .profiler import ExecutionProfiler
        super().__init__(driver, optimizer_level=optimizer_level)
        self._exec_profiler = ExecutionProfiler()
        if self._driver is not None:
            self._driver.profiler = self._exec_profiler

    @property
    def exec_profiler(self):
        return self._exec_profiler

    def execute(self, tasks: list["Task"], source_data=None):
        if self._driver is None:
            from .Drivers.CPU._lib.cpu_driver import CpuDriver
            self._driver = CpuDriver()
        self._driver.profiler = self._exec_profiler
        return super().execute(tasks, source_data)
