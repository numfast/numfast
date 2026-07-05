"""WebGPU Driver — реализация Driver Interface через wgpu-py.

Без numpy. Конвертация BlockView ↔ bytes через WebGpuSerializer.
"""

from typing import Optional

import wgpu

from ...base import Driver
from ....packet import ExecutionPacket
from ....scheduler import ScheduleWave
from .wgsl_serializer import blockview_to_bytes, bytes_to_blockview, pack_uniforms


class WebGpuDriver(Driver):
    """WebGPU Driver через wgpu-py compute shaders.

    BlockView → serializer → bytes → wgpu buffer → dispatch → readback → serializer → BlockView.
    """

    def __init__(self):
        super().__init__()
        self._adapter: Optional[wgpu.GPUAdapter] = None
        self._device: Optional[wgpu.GPUDevice] = None
        self._queue: Optional[wgpu.GPUQueue] = None
        self._shader_cache: dict[str, wgpu.GPUShaderModule] = {}
        self._pipeline_cache: dict[str, wgpu.GPUComputePipeline] = {}
        # Memory compatibility layer for Runtime._build_packet
        self._output_store = {}
        self._output_buffers = {}
        self._init_device()

    def resolve_output(self, name):
        raw = self._output_store.get(name)
        if raw is not None:
            return raw
        return None

    def store_output(self, name: str, task_id: int, output_idx: int, data):
        """Сохранить выход задачи для последующего чтения."""
        self._output_store[name] = data
        self._output_store[(task_id, output_idx)] = data

    def _init_device(self):
        self._adapter = wgpu.gpu.request_adapter_sync(power_preference="low-power")
        self._device = self._adapter.request_device_sync()
        self._queue = self._device.queue

    def compile(self, kernel: str):
        if kernel not in self.kernel_table:
            raise KeyError(f"Kernel '{kernel}' not found in kernel_table")
        entry = self.kernel_table[kernel]
        wgsl_source = entry["drivers"].get("wgsl")
        if wgsl_source is None:
            raise ValueError(f"Kernel '{kernel}' has no WGSL implementation")
        if kernel not in self._shader_cache:
            self._shader_cache[kernel] = self._device.create_shader_module(code=wgsl_source)
        return wgsl_source

    def execute(self, packet: ExecutionPacket):
        """Выполнить один ExecutionPacket на WebGPU.

        Поддерживает:
        - storage buffers (input, output, workspace)
        - uniform buffers (из packet.uniforms)
        """
        kernel = packet.kernel
        entry = self.kernel_table.get(kernel)
        if entry is None:
            raise KeyError(f"Kernel '{kernel}' not found")

        wgsl_source = entry["drivers"].get("wgsl")
        if wgsl_source is None:
            raise ValueError(f"Kernel '{kernel}' has no WGSL implementation")

        # --- 1. Get or compile shader ---
        if kernel not in self._shader_cache:
            self._shader_cache[kernel] = self._device.create_shader_module(code=wgsl_source)
        shader = self._shader_cache[kernel]

        # --- 2. Get or create pipeline ---
        if kernel not in self._pipeline_cache:
            self._pipeline_cache[kernel] = self._device.create_compute_pipeline(
                layout="auto",
                compute={"module": shader, "entry_point": "main"},
            )
        pipeline = self._pipeline_cache[kernel]

        # --- 3. Create storage buffers ---
        gpu_buffers = []
        bind_entries = []

        for bi, bv in enumerate(packet.bindings):
            size_bytes = bv.size * 4  # float32 = 4 bytes

            if bi < len(packet.input_buffers):
                # Input: convert BlockView to bytes and upload
                data_bytes = blockview_to_bytes(bv.view)
                gpu_buf = self._device.create_buffer_with_data(
                    data=data_bytes[:size_bytes],
                    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                )
            else:
                # Output or workspace: empty buffer
                gpu_buf = self._device.create_buffer(
                    size=size_bytes,
                    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
                )

            gpu_buffers.append(gpu_buf)
            bind_entries.append({
                "binding": bi,
                "resource": {"buffer": gpu_buf, "offset": 0, "size": size_bytes},
            })

        # --- 4. Add uniform buffer (if any) ---
        uniform_bi = len(packet.bindings)
        if packet.uniforms:
            uniform_bytes = pack_uniforms(packet.uniforms)
            uniform_buf = self._device.create_buffer_with_data(
                data=uniform_bytes,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            bind_entries.append({
                "binding": uniform_bi,
                "resource": {"buffer": uniform_buf, "offset": 0, "size": len(uniform_bytes)},
            })

        # --- 5. Create bind group ---
        bind_group = self._device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=bind_entries,
        )

        # --- 6. Dispatch ---
        dx, dy, dz = packet.dispatch
        encoder = self._device.create_command_encoder()
        pass_ = encoder.begin_compute_pass()
        pass_.set_pipeline(pipeline)
        pass_.set_bind_group(0, bind_group)
        pass_.dispatch_workgroups(dx, dy, dz)
        pass_.end()
        self._queue.submit([encoder.finish()])

        # --- 7. Read back outputs ---
        num_inputs = len(packet.input_buffers)
        num_ws = len(packet.workspace_buffers)
        for oi, bv in enumerate(packet.output_buffers):
            bi = num_inputs + num_ws + oi
            gpu_buf = gpu_buffers[bi]
            try:
                result_bytes = self._queue.read_buffer(gpu_buf)
                bytes_to_blockview(result_bytes, bv.view)
            except Exception as e:
                print(f"  read_buffer error on output[{oi}]: {e}")

    def execute_wave(self, wave: ScheduleWave):
        for p in wave.packets:
            self.execute(p)

    def wait(self):
        pass

    def release(self):
        self._shader_cache.clear()
        self._pipeline_cache.clear()
        if self._device:
            self._device.destroy()
            self._device = None
