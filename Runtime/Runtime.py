"""Runtime Extension — execution layer for NumFast.

Kernel table, Compiler, Drivers.
"""

from ._lib.mod_iface import Buffer, InputSlot, OutputSlot, ExecutionPlan, ExecutionContext, BufferSpec, BlockView, validate_uniforms, KernelValidator
from ._lib.task import Task, ResourceRef
from ._lib.compiler import compile
from ._lib.runtime import Runtime, ProfiledRuntime
from ._lib.packet import ExecutionPacket, BufferView
from ._lib.Drivers.base import Driver
from ._lib.Drivers.CPU._lib.cpu_driver import CpuDriver
from ._lib.Drivers.CPU._lib.thread_pool import CpuThreadPool
from ._lib.Drivers.CPU._lib.cpu_memory import CpuMemoryManager
from ._lib.Drivers.CPU._lib.worker_pool import CpuWorkerPool
from ._lib.scheduler import Scheduler, ScheduleWave


def setup(kernel):
    kernel.metadata.setdefault("Runtime", {})
    kernel.metadata["Runtime"]["version"] = "0.1.0"
    kernel.metadata["Runtime"]["types"] = [
        "Buffer", "BufferSpec", "InputSlot", "OutputSlot", "ExecutionPlan", "ExecutionContext",
        "BlockView", "BufferView", "ExecutionPacket",
        "validate_uniforms", "KernelValidator",
        "Task", "ResourceRef",
        "Runtime", "ProfiledRuntime", "Driver", "CpuDriver", "Scheduler", "CpuMemoryManager",
        "CpuWorkerPool",
        "CpuThreadPool",
        "ScheduleWave",
    ]
