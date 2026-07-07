"""wgpu device management — ленивый синглтон.

Позволяет всем компонентам использовать один GPUDevice.
"""

import wgpu.backends.wgpu_native  # noqa: F401 — инициализация native backend


_device = None


def get_device() -> wgpu.GPUDevice:
    """Вернуть синглтон GPUDevice (ленивая инициализация)."""
    global _device
    if _device is None:
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance"
        )
        _device = adapter.request_device_sync(
            required_limits={
                "max_storage_buffer_binding_size": 256 * 1024 * 1024,  # 256 MB
            }
        )
    return _device
