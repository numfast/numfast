"""Runtime — верхний уровень исполнительного слоя NumFast.

Объединяет:
  - kernel_table (alias -> {describe, drivers: {cpu, wgsl, cuda}})
  - Compiler (jobs -> tasks)
  - Scheduler (tasks -> waves)
  - Driver (execute waves)

Правило: Runtime НЕ создаёт память.
Вся память — через Driver.
"""

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

    def __init__(self, driver: Optional["Driver"] = None):
        self.kernel_table: dict[str, dict] = {}
        self._driver = driver
        self._scheduler = Scheduler()
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
        """Выполнить список задач.

        1. Scheduler: tasks -> waves (по зависимостям)
        2. Для каждой волны: build packets -> driver.execute_wave()
        """
        if self._driver is None:
            from .Drivers.CPU._lib.cpu_driver import CpuDriver
            self._driver = CpuDriver()

        self._driver.kernel_table = self.kernel_table

        waves = self._scheduler.schedule(tasks)

        for wave_idx, wave in enumerate(waves):
            packets = []
            for task in wave.packets:
                packet = self._build_packet(task, source_data)
                packets.append(packet)

            sw = ScheduleWave(packets=packets)
            self._driver.execute_wave(sw)

    def _build_packet(self, task, source_data) -> "ExecutionPacket":
        """Task -> ExecutionPacket.

        Resolves ResourceRefs, allocates outputs and workspace
        через MemoryManager драйвера.
        """
        from .packet import BufferView as BV

        input_views = []
        for ref in task.inputs:
            if ref.type == "input":
                arr = source_data[ref.column]
                input_views.append(BV(view=BlockView(arr), dtype="float", size=len(arr)))
            elif ref.type == "task":
                arr = self._driver.resolve_output((ref.task_id, ref.output_idx))
                if arr is None:
                    # Fallback: try by name via output_store
                    arr = self._driver._output_store.get((ref.task_id, ref.output_idx))
                input_views.append(BV(view=BlockView(arr), dtype="float", size=len(arr)))
            else:
                raise ValueError(f"Unknown ref type: {ref.type}")

        input_sizes = [v.size for v in input_views]

        # Dispatch size = first input (controls workgroup grid)
        dispatch_size = input_sizes[0] if input_sizes else 0

        # Output sizes from descriptor or default
        if task.output_size_fn is not None:
            output_sizes = task.output_size_fn(input_sizes)
        else:
            # Default: all outputs = first input size (backward compat)
            output_sizes = [dispatch_size] * task.num_outputs

        if len(output_sizes) != task.num_outputs:
            raise ValueError(
                f"output_size_fn returned {len(output_sizes)} sizes, "
                f"expected {task.num_outputs}"
            )

        output_views = []
        output_raws = []
        for i in range(task.num_outputs):
            out_size = output_sizes[i]
            raw = _alloc_driver_array(self._driver, out_size)
            output_raws.append(raw)
            output_views.append(BV(view=BlockView(raw), dtype="float", size=out_size))

        workspace_views = []
        for ws_spec in task.workspace:
            ws_size = ws_spec.elements if ws_spec.elements > 0 else dispatch_size
            raw = _alloc_driver_array(self._driver, ws_size)
            workspace_views.append(BV(view=BlockView(raw), dtype=ws_spec.dtype, size=ws_size))

        uniforms = dict(task.uniforms)

        bindings = list(input_views) + list(workspace_views) + list(output_views)

        if task.dispatch is not None:
            dispatch = task.dispatch
        else:
            wg_size = 64  # default workgroup size
            dx = (dispatch_size + wg_size - 1) // wg_size
            dispatch = (dx, 1, 1)

        packet = ExecutionPacket(
            kernel=task.op,
            input_buffers=input_views,
            output_buffers=output_views,
            workspace_buffers=workspace_views,
            uniforms=uniforms,
            bindings=bindings,
            dispatch=dispatch,
            abi_version=self.kernel_table.get(task.op, {}).get("abi_version", 1),
        )

        # Сохранить выходы через Driver (не напрямую)
        for i, raw in enumerate(output_raws):
            out_name = task.out_names[i]
            self._driver.store_output(out_name, task.id, i, raw)

        return packet

    @property
    def driver(self):
        return self._driver

    @driver.setter
    def driver(self, d):
        self._driver = d


import logging
logger = logging.getLogger(__name__)


class ProfiledRuntime(Runtime):
    """Runtime со встроенным профайлером.

    Usage:
        runtime = ProfiledRuntime()
        register_trading_kernels(runtime)
        runtime.execute(tasks, data)
        runtime.profiler.print_report()
    """

    def __init__(self, driver=None):
        from .profiler import RuntimeProfiler
        super().__init__(driver)
        self.profiler = RuntimeProfiler()

    def compile(self, jobs: list[dict]) -> list["Task"]:
        with self.profiler.phase("compile"):
            return super().compile(jobs)

    def execute(self, tasks: list["Task"], source_data=None):
        with self.profiler.phase("execute"):
            return super().execute(tasks, source_data)
