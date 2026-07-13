"""GPU compute pipeline for downsampling visible window.

Держит полный датасет на GPU постоянно — перезаписывает только при add_series.
Без пересылки всего датасета на каждый кадр.
"""

from __future__ import annotations

import numpy as np
import struct
import wgpu

WGSL_DOWNSAMPLE = """
struct Params {
    x_min: u32,
    x_max: u32,
    out_len: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let bucket = id.x;
    if (bucket >= params.out_len) { return; }

    let window = params.x_max - params.x_min;
    let bucket_size = max(1u, window / params.out_len);
    let start = params.x_min + bucket * bucket_size;
    let end = min(params.x_max, start + bucket_size);
    if (end <= start) { return; }

    var sum: f32 = 0.0;
    for (var i = start; i < end; i = i + 1u) {
        sum = sum + input[i];
    }
    output[bucket] = sum / f32(end - start);
}
"""


class GPUEngine:
    """Хранит полный датасет на GPU, даунсемплит без копирования на кадр."""

    def __init__(self, target_width=2000):
        self.adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance"
        )
        self.device = self.adapter.request_device_sync()
        self._pipeline = None
        self._bind_group_layout = None
        self._target = target_width

        self._create_pipeline()
        self._create_buffers()

        # Series buffers — создаются при upload
        self._series_bufs: list = []  # each: {buffer, size}
        self._out_buf = None
        self._staging = None
        self._params_buf = None

    def _create_pipeline(self):
        shader = self.device.create_shader_module(code=WGSL_DOWNSAMPLE)
        self._bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.storage}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.uniform}},
            ]
        )
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._bind_group_layout]
        )
        self._pipeline = self.device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader, "entry_point": "main"},
        )

    def _create_buffers(self):
        out_size = self._target * 4
        self._out_buf = self.device.create_buffer(
            size=out_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        self._staging = self.device.create_buffer(
            size=out_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        self._params_buf = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

    def upload_series(self, data: np.ndarray):
        """Загружает серию на GPU (один раз, при add_series)."""
        arr = np.ascontiguousarray(data, dtype=np.float32)
        nbytes = arr.nbytes
        buf = self.device.create_buffer(
            size=nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.device.queue.write_buffer(buf, 0, arr.tobytes())
        self._series_bufs.append({"buffer": buf, "size": nbytes, "length": len(arr)})

    def upload_all(self, series_list: list[np.ndarray]):
        """Загружает все серии на GPU."""
        for s in series_list:
            self.upload_series(s)

    def downsample_series(self, x_min: int, x_max: int, target: int):
        """Даунсемплинг всех серий (без копирования данных — уже на GPU)."""
        window = x_max - x_min
        out_len = min(target, window)
        if out_len <= 0 or not self._series_bufs:
            return [], float("inf"), float("-inf")

        out_nbytes = out_len * 4
        params = struct.pack("<III", x_min, x_max, out_len)
        self.device.queue.write_buffer(self._params_buf, 0, params)

        slices = []
        y_min, y_max = float("inf"), float("-inf")

        for sb in self._series_bufs:
            bind_group = self.device.create_bind_group(
                layout=self._bind_group_layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": sb["buffer"],
                                                 "offset": 0, "size": sb["size"]}},
                    {"binding": 1, "resource": {"buffer": self._out_buf,
                                                 "offset": 0, "size": out_nbytes}},
                    {"binding": 2, "resource": {"buffer": self._params_buf,
                                                 "offset": 0, "size": 16}},
                ],
            )

            num_wg = max(1, (out_len + 255) // 256)
            encoder = self.device.create_command_encoder()
            pass_ = encoder.begin_compute_pass()
            pass_.set_pipeline(self._pipeline)
            pass_.set_bind_group(0, bind_group)
            pass_.dispatch_workgroups(num_wg)
            pass_.end()
            encoder.copy_buffer_to_buffer(self._out_buf, 0, self._staging, 0,
                                          out_nbytes)
            self.device.queue.submit([encoder.finish()])

            self._staging.map_sync(mode=wgpu.MapMode.READ)
            raw = bytes(self._staging.read_mapped())
            self._staging.unmap()

            result = np.frombuffer(raw, dtype=np.float32).copy()
            if len(result) > 0:
                y_min = min(y_min, float(result.min()))
                y_max = max(y_max, float(result.max()))
            slices.append(result)

        if y_min == float("inf"):
            y_min, y_max = 0.0, 1.0
        return slices, y_min, y_max

    def clear(self):
        for sb in self._series_bufs:
            sb["buffer"].destroy()
        self._series_bufs = []
