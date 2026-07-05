"""Viewport Aggregation Engine — GPU compute поверх NumFast бэкенда.

- Хранит серии на GPU (wgpu storage buffers)
- На запрос (x_min, x_max, width) делает GPU compute downsample
- Отдаёт ~width точек на клиент

Поддерживает два типа серий:
  - "raw": сырой float32 буфер (наследие)
  - "table": бито-упакованные строки из TABLE_REGISTRY (JIT WGSL)
"""

from __future__ import annotations

import struct

import numpy as np
import wgpu

from numfast._core.backend import _wgpu

_WGSL_COMMON_MAIN = """
    let bucket = id.x;
    if (bucket >= params.out_len) { return; }

    let window = params.x_max - params.x_min;
    let bucket_size = window / params.out_len;
    let remainder = window - bucket_size * params.out_len;

    if (bucket_size == 0u) {
        if (bucket < window) {
            output[bucket] = __DIRECT__;
        } else {
            output[bucket] = 0.0;
        }
        return;
    }

    // Распределяем остаток: первые `remainder` корзин получают +1 точку
    let extra = u32(bucket < remainder);
    let start = params.x_min + bucket * bucket_size + min(bucket, remainder);
    let end = min(params.x_max, start + bucket_size + extra);
    if (end <= start) { output[bucket] = 0.0; return; }

    var sum: f32 = 0.0;
    for (var i = start; i < end; i = i + 1u) {
        sum = sum + __LOOP__;
    }
    output[bucket] = sum / f32(end - start);
"""


_WGSL_DOWNSAMPLE = """
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
""" + _WGSL_COMMON_MAIN.replace("__DIRECT__", "input[params.x_min + bucket]").replace("__LOOP__", "input[i]") + """
}
"""


def _generate_viewport_shader(num_parts: int, layout: dict, col_name: str) -> str:
    """JIT-генерирует WGSL шейдер даунсемплинга для бито-упакованных строк."""
    cinfo = layout["columns"][col_name]
    extract_expr, _ = _wgpu._generate_extract_expr(cinfo)

    struct_fields = "\n    ".join(
        f"part{i}: u32," for i in range(num_parts)
    )

    body = _WGSL_COMMON_MAIN.replace(
        "__DIRECT__", "load_val(input[params.x_min + bucket])",
    ).replace(
        "__LOOP__", "load_val(input[i])",
    )

    return f"""
struct Row {{
    {struct_fields}
}};

struct Params {{
    x_min: u32,
    x_max: u32,
    out_len: u32,
    _pad: u32,
}};

@group(0) @binding(0) var<storage, read> input: array<Row>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn load_val(row: Row) -> f32 {{
    return {extract_expr};
}}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{{body}
}}
"""


_SHADER_CACHE: dict[tuple, object] = {}


class ViewportAgg:
    """GPU-акселерированный даунсемплинг viewport.

    Использует единый wgpu device из NumFast бэкенда.
    """

    def __init__(self, max_width=2000):
        _wgpu._ensure_device()
        self._device = _wgpu._DEVICE
        self._max_width = max_width
        self._raw_pipeline = None
        self._bind_group_layout = None
        self._pipeline_layout = None
        self._params_buf = None
        self._out_buf = None
        self._staging = None
        self._series_entries: list[dict] = []
        self._target = max_width

        self._create_shared()
        self._create_raw_pipeline()

    def _create_shared(self):
        d = self._device
        self._bind_group_layout = d.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.storage}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.uniform}},
            ]
        )
        self._pipeline_layout = d.create_pipeline_layout(
            bind_group_layouts=[self._bind_group_layout]
        )

        out_size = self._max_width * 4
        self._out_buf = d.create_buffer(
            size=out_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        self._staging = d.create_buffer(
            size=out_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        self._params_buf = d.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

    def _create_raw_pipeline(self):
        d = self._device
        shader = d.create_shader_module(code=_WGSL_DOWNSAMPLE)
        self._raw_pipeline = d.create_compute_pipeline(
            layout=self._pipeline_layout,
            compute={"module": shader, "entry_point": "main"},
        )

    def _get_or_create_jit_pipeline(self, num_parts, layout, col_name):
        cinfo = layout["columns"][col_name]
        parts_tuple = tuple(tuple(p) for p in cinfo["parts"])
        key = (num_parts, col_name, parts_tuple)
        cached = _SHADER_CACHE.get(key)
        if cached is not None:
            return cached
        src = _generate_viewport_shader(num_parts, layout, col_name)
        module = self._device.create_shader_module(code=src)
        pipeline = self._device.create_compute_pipeline(
            layout=self._pipeline_layout,
            compute={"module": module, "entry_point": "main"},
        )
        _SHADER_CACHE[key] = pipeline
        return pipeline

    def attach_series(self, data: np.ndarray):
        """Загружает сырую float32 серию на GPU."""
        arr = np.ascontiguousarray(data, dtype=np.float32)
        nbytes = arr.nbytes
        buf = self._device.create_buffer(
            size=nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._device.queue.write_buffer(buf, 0, arr.tobytes())
        self._series_entries.append({
            "type": "raw",
            "buffer": buf,
            "size": nbytes,
            "length": len(arr),
        })
        self._target = max(self._target, len(arr))

    def attach_series_via_table(self, table_id: str, col_name: str,
                                 layout: dict, length: int):
        """Загружает бито-упакованные строки из TABLE_REGISTRY на GPU."""
        from _core.kernel import TABLE_REGISTRY

        tbl = TABLE_REGISTRY.get(table_id)
        if tbl is None:
            raise ValueError(f"Table {table_id} not found in registry")

        rows = tbl["rows"]
        num_parts = tbl["num_parts"]

        flat = np.asarray(rows, dtype=np.uint32).ravel()
        nbytes = flat.nbytes

        buf = self._device.create_buffer(
            size=nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._device.queue.write_buffer(buf, 0, flat.tobytes())

        pipeline = self._get_or_create_jit_pipeline(num_parts, layout, col_name)

        self._series_entries.append({
            "type": "table",
            "buffer": buf,
            "size": nbytes,
            "length": length,
            "pipeline": pipeline,
        })
        self._target = max(self._target, length)

    def aggregate(self, x_min: int, x_max: int, target: int) -> list[np.ndarray]:
        if not self._series_entries:
            return []

        window = x_max - x_min
        if window <= 0:
            return [np.array([], dtype=np.float32) for _ in self._series_entries]

        out_len = max(1, min(target, window))
        if out_len > self._max_width:
            out_len = self._max_width

        out_nbytes = out_len * 4
        params = struct.pack("<III", x_min, x_max, out_len)
        self._device.queue.write_buffer(self._params_buf, 0, params)

        results = []
        for se in self._series_entries:
            pipeline = se.get("pipeline", self._raw_pipeline)
            bg = self._device.create_bind_group(
                layout=self._bind_group_layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": se["buffer"],
                                                 "offset": 0, "size": se["size"]}},
                    {"binding": 1, "resource": {"buffer": self._out_buf,
                                                 "offset": 0, "size": out_nbytes}},
                    {"binding": 2, "resource": {"buffer": self._params_buf,
                                                 "offset": 0, "size": 16}},
                ],
            )

            num_wg = max(1, (out_len + 255) // 256)
            encoder = self._device.create_command_encoder()
            pass_ = encoder.begin_compute_pass()
            pass_.set_pipeline(pipeline)
            pass_.set_bind_group(0, bg)
            pass_.dispatch_workgroups(num_wg)
            pass_.end()
            encoder.copy_buffer_to_buffer(self._out_buf, 0, self._staging, 0, out_nbytes)
            self._device.queue.submit([encoder.finish()])

            promise = self._staging.map_async("read")
            if promise is not None:
                promise.sync_wait()
            raw = self._staging.read_mapped(0, out_nbytes)
            self._staging.unmap()

            results.append(np.frombuffer(raw, dtype=np.float32).copy())

        return results

    def clear(self):
        for se in self._series_entries:
            se["buffer"].destroy()
        self._series_entries = []
