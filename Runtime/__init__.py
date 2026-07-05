"""Runtime — исполнительный слой NumFast.

Cодержит:
  - kernel_table (alias -> {describe, cpu})
  - Compiler (jobs -> tasks)
  - Drivers (CPU, WebGPU, CUDA)

Никаких активных объектов. Всё — метаданные Context.
"""

from .Runtime import (
    Buffer, BufferSpec, InputSlot, OutputSlot, ExecutionPlan, ExecutionContext, BlockView,
    Task, ResourceRef,
    ExecutionPacket, BufferView,
    validate_uniforms, KernelValidator,
    compile,
    Runtime, ProfiledRuntime,
    Driver, CpuDriver, Scheduler, CpuMemoryManager, CpuWorkerPool,
    CpuThreadPool,
)

__all__ = [
    "Buffer", "BufferSpec", "InputSlot", "OutputSlot", "ExecutionPlan", "ExecutionContext",
    "BlockView", "BufferView", "ExecutionPacket",
    "validate_uniforms", "KernelValidator",
    "Task", "ResourceRef",
    "compile",
    "Runtime", "ProfiledRuntime",
    "Driver", "CpuDriver", "Scheduler", "CpuMemoryManager", "CpuWorkerPool",
    "CpuThreadPool",
]
