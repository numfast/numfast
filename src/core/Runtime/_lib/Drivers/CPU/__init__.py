from .CPU import cpu_driver

def __getattr__(name):
    if name == "CpuDriver":
        from ._lib.cpu_driver import CpuDriver
        return CpuDriver
    if name == "CpuThreadPool":
        from ._lib.thread_pool import CpuThreadPool
        return CpuThreadPool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["cpu_driver", "CpuDriver", "CpuThreadPool"]
