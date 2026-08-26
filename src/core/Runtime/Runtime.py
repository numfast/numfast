"""Runtime Extension -- execution layer for NumFast.
Builder entry point.
"""

_runtime_instance = None


def _get_runtime():
    global _runtime_instance
    if _runtime_instance is None:
        from Runtime._lib.runtime import Runtime as _Runtime
        _runtime_instance = _Runtime()
    return _runtime_instance


def _get_cpu_driver():
    return _get_runtime()._driver


def execute(tasks, source_data=None):
    return _get_runtime().execute(tasks, source_data)


def register_kernel(alias, describe=None, cpu=None, wgsl=None, **kw):
    return _get_runtime().register_kernel(alias, describe, cpu, wgsl, **kw)


def setup(kernel):
    rt = _get_runtime()
    kernel.metadata.setdefault("Runtime", {})
    kernel.metadata["Runtime"]["version"] = "0.1.0"


def compile(task):
    from Runtime._lib.compiler import compile as _compile
    return _get_runtime().compile(task)


def Runtime(driver=None):
    from Runtime._lib.runtime import Runtime as _Runtime
    rt = _Runtime(driver)
    singleton = _get_runtime()
    if singleton.kernel_table:
        rt.kernel_table = singleton.kernel_table
    return rt


def ProfiledRuntime(*args, **kwargs):
    from Runtime._lib.runtime import ProfiledRuntime as _PR
    return _PR(*args, **kwargs)


def Driver(*args, **kwargs):
    from Runtime._lib.Drivers.base import Driver as _Driver
    return _Driver(*args, **kwargs)


def CpuDriver():
    from Runtime._lib.Drivers.CPU._lib.cpu_driver import CpuDriver as _CD
    return _CD()


def GpuDriver():
    from Runtime._lib.Drivers.WebGPU import WebGpuDriver as _WD
    return _WD()


def Buffer(*args, **kwargs):
    from Runtime._lib.mod_iface import Buffer as _Buffer
    return _Buffer(*args, **kwargs)


def BufferSpec(*args, **kwargs):
    from Runtime._lib.mod_iface import BufferSpec as _BS
    return _BS(*args, **kwargs)


def InputSlot(*args, **kwargs):
    from Runtime._lib.mod_iface import InputSlot as _IS
    return _IS(*args, **kwargs)


def OutputSlot(*args, **kwargs):
    from Runtime._lib.mod_iface import OutputSlot as _OS
    return _OS(*args, **kwargs)


def ExecutionPlan(*args, **kwargs):
    from Runtime._lib.mod_iface import ExecutionPlan as _EP
    return _EP(*args, **kwargs)


def ExecutionContext(*args, **kwargs):
    from Runtime._lib.mod_iface import ExecutionContext as _EC
    return _EC(*args, **kwargs)


def BlockView(*args, **kwargs):
    from Runtime._lib.mod_iface import BlockView as _BV
    return _BV(*args, **kwargs)


def validate_uniforms(*args, **kwargs):
    from Runtime._lib.mod_iface import validate_uniforms as _vu
    return _vu(*args, **kwargs)


def KernelValidator(*args, **kwargs):
    from Runtime._lib.mod_iface import KernelValidator as _KV
    return _KV(*args, **kwargs)


def Task(*args, **kwargs):
    from Runtime._lib.task import Task as _Task
    return _Task(*args, **kwargs)


def ResourceRef(*args, **kwargs):
    from Runtime._lib.task import ResourceRef as _RR
    return _RR(*args, **kwargs)


def ExecutionPacket(*args, **kwargs):
    from Runtime._lib.packet import ExecutionPacket as _EP
    return _EP(*args, **kwargs)


def BufferView(*args, **kwargs):
    from Runtime._lib.packet import BufferView as _BV
    return _BV(*args, **kwargs)


def Scheduler(*args, **kwargs):
    from Runtime._lib.scheduler import Scheduler as _S
    return _S(*args, **kwargs)


def ScheduleWave(*args, **kwargs):
    from Runtime._lib.scheduler import ScheduleWave as _SW
    return _SW(*args, **kwargs)


def CpuMemoryManager(*args, **kwargs):
    from Runtime._lib.Drivers.CPU._lib.cpu_memory import CpuMemoryManager as _CMM
    return _CMM(*args, **kwargs)


def CpuWorkerPool(*args, **kwargs):
    from Runtime._lib.Drivers.CPU._lib.worker_pool import CpuWorkerPool as _CWP
    return _CWP(*args, **kwargs)


def CpuThreadPool(*args, **kwargs):
    from Runtime._lib.Drivers.CPU._lib.thread_pool import CpuThreadPool as _CTP
    return _CTP(*args, **kwargs)


__all__ = [
    "Buffer", "BufferSpec", "InputSlot", "OutputSlot", "ExecutionPlan", "ExecutionContext",
    "BlockView", "BufferView", "ExecutionPacket",
    "validate_uniforms", "KernelValidator",
    "Task", "ResourceRef",
    "compile",
    "Runtime", "ProfiledRuntime",
    "Driver", "CpuDriver", "GpuDriver", "Scheduler", "CpuMemoryManager", "CpuWorkerPool",
    "CpuThreadPool",
]
