"""WebGPU Driver — wgpu-py compute backend.

Supports both float32 and uint32 (packed) buffers.
Full profiler integration.
"""

import time
from typing import Optional

import wgpu

from Runtime._lib.Drivers.base import Driver
from Runtime._lib.packet import ExecutionPacket
from Runtime._lib.scheduler import ScheduleWave
from .wgsl_serializer import blockview_to_bytes, bytes_to_blockview, pack_uniforms


class WebGpuDriver(Driver):
    """WebGPU Driver through wgpu-py compute shaders.

    Supports:
    - float32 storage buffers
    - uint32 packed buffers (for PackedTable data)
    - uniform buffers
    - ExecutionProfiler integration
    - upload/dispatch/readback timing
    """

    def __init__(self, profiler=None):
        super().__init__(profiler)
        self._adapter: Optional[wgpu.GPUAdapter] = None
        self._device: Optional[wgpu.GPUDevice] = None
        self._queue: Optional[wgpu.GPUQueue] = None
        self._shader_cache: dict = {}
        self._pipeline_cache: dict = {}
        self._output_store: dict = {}
        self._init_device()

    def resolve_output(self, name):
        return self._output_store.get(name)

    def store_output(self, name: str, task_id: int, output_idx: int, data):
        self._output_store[name] = data
        self._output_store[(task_id, output_idx)] = data

    def _init_device(self):
        self._adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self._device = self._adapter.request_device_sync()
        self._queue = self._device.queue

    def _get_packed_info(self, packet: ExecutionPacket) -> dict:
        """Determine if bindings are float or uint based on kernel caps."""
        entry = self.kernel_table.get(packet.kernel, {})
        caps = entry.get("capabilities", {})
        return {
            "input_dtype": caps.get("input_dtype", "float"),
            "output_dtype": caps.get("output_dtype", "float"),
            "elem_size": 4,
        }

    def compile(self, kernel: str):
        if kernel not in self.kernel_table:
            raise KeyError(f"Kernel '{kernel}' not found")
        entry = self.kernel_table[kernel]
        wgsl_source = entry["drivers"].get("wgsl")
        if wgsl_source is None:
            raise ValueError(f"Kernel '{kernel}' has no WGSL implementation")
        if kernel not in self._shader_cache:
            self._shader_cache[kernel] = self._device.create_shader_module(code=wgsl_source)
        return wgsl_source

    def execute(self, packet: ExecutionPacket):
        """Execute one ExecutionPacket on WebGPU.

        Stages (timed separately):
        1. upload — CPU to GPU transfer
        2. dispatch — GPU compute
        3. readback — GPU to CPU transfer
        """
        kernel = packet.kernel
        entry = self.kernel_table.get(kernel)
        if entry is None:
            raise KeyError(f"Kernel '{kernel}' not found")

        wgsl_source = entry["drivers"].get("wgsl")
        if wgsl_source is None:
            raise ValueError(f"Kernel '{kernel}' has no WGSL implementation")

        packed_info = self._get_packed_info(packet)
        elem_size = packed_info["elem_size"]

        # --- Profiler start ---
        prof = self._profiler
        if prof:
            prof.on_packet_start(packet)

        # --- 1. Resolve WGSL ---
        if callable(wgsl_source):
            wgsl_source = wgsl_source(packet.uniforms)
        if not isinstance(wgsl_source, str):
            raise TypeError(f"wgsl_source for '{kernel}' must be str")

        # --- 2. Compile shader ---
        t_compile_start = time.perf_counter_ns()
        cache_key = (kernel, hash(wgsl_source))
        if cache_key not in self._shader_cache:
            self._shader_cache[cache_key] = self._device.create_shader_module(code=wgsl_source)
        shader = self._shader_cache[cache_key]
        t_compile_ns = time.perf_counter_ns() - t_compile_start
        packet.profile["compile_ns"] = packet.profile.get("compile_ns", 0) + t_compile_ns

        # --- 3. Create pipeline ---
        t_pipe_start = time.perf_counter_ns()
        num_inputs = len(packet.input_buffers)
        num_bindings = len(packet.bindings)
        has_uniform = bool(packet.uniforms)
        layout_key = (kernel, num_bindings, has_uniform)

        if layout_key not in self._pipeline_cache:
            bgl_entries = []
            for i in range(num_bindings):
                buf_type = wgpu.BufferBindingType.read_only_storage if i < num_inputs else wgpu.BufferBindingType.storage
                bgl_entries.append({
                    "binding": i,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": buf_type, "has_dynamic_offset": False, "min_binding_size": 0},
                })
            if has_uniform:
                bgl_entries.append({
                    "binding": num_bindings,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.uniform, "has_dynamic_offset": False, "min_binding_size": 0},
                })
            bgl = self._device.create_bind_group_layout(entries=bgl_entries)
            layout = self._device.create_pipeline_layout(bind_group_layouts=[bgl])
            self._pipeline_cache[layout_key] = self._device.create_compute_pipeline(
                layout=layout, compute={"module": shader, "entry_point": "main"},
            )
        pipeline = self._pipeline_cache[layout_key]
        t_pipe_ns = time.perf_counter_ns() - t_pipe_start
        packet.profile["pipeline_ns"] = packet.profile.get("pipeline_ns", 0) + t_pipe_ns

        # --- 4. Upload: CPU -> GPU ---
        t_upload_start = time.perf_counter_ns()
        gpu_buffers = []
        bind_entries = []

        for bi, bv in enumerate(packet.bindings):
            dtype = packed_info["input_dtype"] if bi < num_inputs else packed_info["output_dtype"]
            size_bytes = bv.size * elem_size

            if bi < num_inputs:
                data_bytes = blockview_to_bytes(bv.view, dtype=dtype)
                gpu_buf = self._device.create_buffer_with_data(
                    data=data_bytes[:size_bytes],
                    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                )
            else:
                gpu_buf = self._device.create_buffer(
                    size=size_bytes,
                    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
                )

            gpu_buffers.append(gpu_buf)
            bind_entries.append({
                "binding": bi,
                "resource": {"buffer": gpu_buf, "offset": 0, "size": size_bytes},
            })

        # Uniform buffer
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

        t_upload_ns = time.perf_counter_ns() - t_upload_start
        packet.profile["upload_ns"] = packet.profile.get("upload_ns", 0) + t_upload_ns

        # --- 5. Bind group ---
        bgl = pipeline.get_bind_group_layout(0)
        bind_group = self._device.create_bind_group(layout=bgl, entries=bind_entries)

        # --- 6. Dispatch ---
        t_dispatch_start = time.perf_counter_ns()
        dx, dy, dz = packet.dispatch
        encoder = self._device.create_command_encoder()
        pass_ = encoder.begin_compute_pass()
        pass_.set_pipeline(pipeline)
        pass_.set_bind_group(0, bind_group)
        pass_.dispatch_workgroups(dx, dy, dz)
        pass_.end()
        self._queue.submit([encoder.finish()])
        self._device.poll()
        t_dispatch_ns = time.perf_counter_ns() - t_dispatch_start
        packet.profile["dispatch_ns"] = packet.profile.get("dispatch_ns", 0) + t_dispatch_ns

        # --- 7. Readback: GPU -> CPU ---
        t_readback_start = time.perf_counter_ns()
        num_ws = len(packet.workspace_buffers)
        for oi, bv in enumerate(packet.output_buffers):
            bi = num_inputs + num_ws + oi
            if bi < len(gpu_buffers):
                gpu_buf = gpu_buffers[bi]
                try:
                    result_bytes = self._queue.read_buffer(gpu_buf)
                    bytes_to_blockview(result_bytes, bv.view, dtype=packed_info["output_dtype"])
                except Exception as e:
                    print(f"  read_buffer error on output[{oi}]: {e}")
        t_readback_ns = time.perf_counter_ns() - t_readback_start
        packet.profile["readback_ns"] = packet.profile.get("readback_ns", 0) + t_readback_ns

        packet.profile["exec_time_ns"] = (
            t_upload_ns + t_dispatch_ns + t_readback_ns + t_pipe_ns + t_compile_ns
        )

        # --- Profiler finish ---
        if prof:
            prof.on_packet_finish(packet)

    def execute_wave(self, wave: ScheduleWave):
        for p in wave.packets:
            self.execute(p)

    def wait(self):
        self._device.poll()

    def release(self):
        self._shader_cache.clear()
        self._pipeline_cache.clear()
        if self._device:
            self._device.destroy()
            self._device = None
