def __getattr__(name):
    if name == "CpuDriver":
        from .CPU import CpuDriver
        return CpuDriver
    if name == "WebGpuDriver":
        from .WebGPU import WebGpuDriver
        return WebGpuDriver
    if name == "CudaDriver":
        from .CUDA import CudaDriver
        return CudaDriver
    if name == "OpenClDriver":
        from .OpenCL import OpenClDriver
        return OpenClDriver
    if name == "MetalDriver":
        from .Metal import MetalDriver
        return MetalDriver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["CpuDriver", "WebGpuDriver", "CudaDriver", "OpenClDriver", "MetalDriver"]
