# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#
# ⚠️ LEGACY — Old GPU compute layer.
# Superseded by: Runtime._lib.Drivers.WebGPU + Compute.Core.*
# Still used by: Stats, Tables, Series, Vis (import directly).
# New code should NOT import from here. Use Runtime + operations API instead.

import math
import numpy as np

PRIORITY = 20

_ADAPTER = None
_DEVICE = None
AVAILABLE = False


def _probe():
    global _ADAPTER, AVAILABLE
    try:
        import wgpu
        _ADAPTER = wgpu.gpu.request_adapter_sync()
        AVAILABLE = True
    except Exception:
        AVAILABLE = False


_MOMENTS_SHADER_SRC = """
struct Params {
    valid_elements: u32,
    scale: f32,
    offset: f32,
    min_int: f32,
    is_compressed: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> params: Params;

fn load_val(idx: u32) -> f32 {
    let raw = input[idx];
    if (params.is_compressed == 1u) {
        return (raw - params.min_int) * params.scale + params.offset;
    }
    return raw;
}

var<workgroup> shmem_sum: array<f32, 256>;
var<workgroup> shmem_sum_sq: array<f32, 256>;
var<workgroup> shmem_min: array<f32, 256>;
var<workgroup> shmem_max: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let idx = gid.x;
    let valid = params.valid_elements;

    if idx < valid {
        let val = load_val(idx);
        shmem_sum[lid.x] = val;
        shmem_sum_sq[lid.x] = val * val;
        shmem_min[lid.x] = val;
        shmem_max[lid.x] = val;
    } else {
        shmem_sum[lid.x] = 0.0;
        shmem_sum_sq[lid.x] = 0.0;
        shmem_min[lid.x] = 340282346638528859811704183484516925440.0;
        shmem_max[lid.x] = -340282346638528859811704183484516925440.0;
    }

    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) {
            shmem_sum[lid.x] = shmem_sum[lid.x] + shmem_sum[lid.x + stride];
            shmem_sum_sq[lid.x] = shmem_sum_sq[lid.x] + shmem_sum_sq[lid.x + stride];
            shmem_min[lid.x] = min(shmem_min[lid.x], shmem_min[lid.x + stride]);
            shmem_max[lid.x] = max(shmem_max[lid.x], shmem_max[lid.x + stride]);
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        let wg_offset = wg_id.x * 5u;
        output[wg_offset + 0u] = shmem_sum[0];
        output[wg_offset + 1u] = shmem_sum_sq[0];
        output[wg_offset + 2u] = shmem_min[0];
        output[wg_offset + 3u] = shmem_max[0];
        output[wg_offset + 4u] = 0.0;
    }
}
"""


def _probe_light():
    global _ADAPTER, AVAILABLE
    try:
        import wgpu
        _ADAPTER = wgpu.gpu.request_adapter_sync()
        AVAILABLE = True
    except Exception:
        AVAILABLE = False


_probe_light()


def _ensure_device():
    global _ADAPTER, _DEVICE, AVAILABLE
    if _DEVICE is not None:
        return
    try:
        import wgpu
        _ADAPTER = wgpu.gpu.request_adapter_sync()
        _DEVICE = _ADAPTER.request_device_sync()
        AVAILABLE = True
    except Exception:
        AVAILABLE = False


def get_xp():
    import numpy as np
    return np


def get_device():
    _ensure_device()
    return _DEVICE


def device_info() -> dict:
    _ensure_device()
    if _ADAPTER is None:
        return {"backend": "wgpu", "available": False}
    info = _ADAPTER.info
    return {
        "backend": "wgpu",
        "vendor": info.get("vendor", ""),
        "device": info.get("device", ""),
        "adapter_type": info.get("adapter_type", ""),
        "backend_type": info.get("backend_type", ""),
        "available": True,
    }


def _run_compute(device, shader_src: str, input_data: np.ndarray,
                 chunk_info_bytes: bytes, output_size: int,
                 num_workgroups: int) -> np.ndarray:
    import wgpu

    input_nbytes = input_data.nbytes
    output_buf_size = output_size
    info_size = len(chunk_info_bytes)

    input_buf = device.create_buffer(
        size=input_nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    device.queue.write_buffer(input_buf, 0, input_data.tobytes(), 0, input_nbytes)

    output_buf = device.create_buffer(
        size=output_buf_size,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )

    info_buf = device.create_buffer(
        size=info_size,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    device.queue.write_buffer(info_buf, 0, chunk_info_bytes, 0, info_size)

    shader = device.create_shader_module(code=shader_src)
    pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": shader, "entry_point": "main"},
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": input_buf, "offset": 0, "size": input_nbytes}},
            {"binding": 1, "resource": {"buffer": output_buf, "offset": 0, "size": output_buf_size}},
            {"binding": 2, "resource": {"buffer": info_buf, "offset": 0, "size": info_size}},
        ],
    )

    encoder = device.create_command_encoder()
    pass_ = encoder.begin_compute_pass()
    pass_.set_pipeline(pipeline)
    pass_.set_bind_group(0, bind_group)
    pass_.dispatch_workgroups(num_workgroups, 1, 1)
    pass_.end()
    device.queue.submit([encoder.finish()])

    staging = device.create_buffer(
        size=output_buf_size,
        usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
    )

    encoder2 = device.create_command_encoder()
    encoder2.copy_buffer_to_buffer(output_buf, 0, staging, 0, output_buf_size)
    device.queue.submit([encoder2.finish()])

    promise = staging.map_async("read")
    if promise is not None:
        promise.sync_wait()
    raw = staging.read_mapped(0, output_buf_size)
    staging.unmap()
    return np.frombuffer(raw, dtype=np.float32)


def _run_compute_shader(device, shader_src: str, buffers: list,
                        num_workgroups: int, output_size: int) -> np.ndarray:
    """Run a WGSL compute shader with arbitrary buffer bindings.

    Args:
        device: wgpu device
        shader_src: WGSL source code
        buffers: list of (data_bytes, usage_flags, is_output) tuples
                 where `is_output` means the buffer needs COPY_SRC for staging readback
        num_workgroups: number of workgroups to dispatch
        output_size: total bytes to read back from the last buffer marked as output

    Returns:
        numpy array of uint8 (reshape/cast as needed)
    """
    import wgpu

    gpu_buffers = []
    for i, (data_bytes, usage, is_output) in enumerate(buffers):
        buf = device.create_buffer(size=len(data_bytes), usage=usage)
        if data_bytes:
            device.queue.write_buffer(buf, 0, data_bytes, 0, len(data_bytes))
        gpu_buffers.append(buf)

    shader = device.create_shader_module(code=shader_src)
    pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": shader, "entry_point": "main"},
    )

    entries = []
    for i, buf in enumerate(gpu_buffers):
        entries.append({
            "binding": i,
            "resource": {"buffer": buf, "offset": 0, "size": buf.size},
        })

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=entries,
    )

    encoder = device.create_command_encoder()
    pass_ = encoder.begin_compute_pass()
    pass_.set_pipeline(pipeline)
    pass_.set_bind_group(0, bind_group)
    pass_.dispatch_workgroups(num_workgroups, 1, 1)
    pass_.end()
    device.queue.submit([encoder.finish()])

    # Find the output buffer and copy to staging
    output_idx = None
    for i, (_, _, is_out) in enumerate(buffers):
        if is_out:
            output_idx = i
            break
    if output_idx is None:
        raise ValueError("No output buffer marked in buffers list")

    staging = device.create_buffer(
        size=output_size,
        usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
    )

    encoder2 = device.create_command_encoder()
    encoder2.copy_buffer_to_buffer(gpu_buffers[output_idx], 0, staging, 0, output_size)
    device.queue.submit([encoder2.finish()])

    promise = staging.map_async("read")
    if promise is not None:
        promise.sync_wait()
    raw = staging.read_mapped(0, output_size)
    staging.unmap()
    return np.frombuffer(raw, dtype=np.uint8)


def compute_moments_wgpu(chunk_buf: list,
                          valid_elements: int,
                          compression: dict | None = None) -> dict:
    _ensure_device()
    if not AVAILABLE:
        raise RuntimeError("WebGPU backend is not available")

    import wgpu

    is_scaled = compression and compression.get("type") == "scaled"
    arr = np.array(chunk_buf, dtype=np.float32)

    if is_scaled:
        scale = float(compression["scale"])
        offset = float(compression["offset"])
        min_int = float(compression.get("min_int", 0.0))
        is_compressed = 1
    else:
        scale = 1.0
        offset = 0.0
        min_int = 0.0
        is_compressed = 0

    shader_src = _MOMENTS_SHADER_SRC

    info_bytes = (
        np.array([valid_elements], dtype=np.uint32).tobytes() +
        np.array([scale], dtype=np.float32).tobytes() +
        np.array([offset], dtype=np.float32).tobytes() +
        np.array([min_int], dtype=np.float32).tobytes() +
        np.array([is_compressed], dtype=np.uint32).tobytes() +
        b'\x00' * 12  # pad to 32 bytes for storage buffer alignment
    )

    n = len(arr)
    num_workgroups = max(1, (valid_elements + 255) // 256)

    output_size = num_workgroups * 5 * 4  # 5 f32 per workgroup

    result = _run_compute(
        _DEVICE, shader_src, arr,
        info_bytes, output_size, num_workgroups,
    )

    partials = result.reshape(num_workgroups, 5)
    total_sum = float(np.sum(partials[:, 0]))
    total_sum_sq = float(np.sum(partials[:, 1]))
    total_min = float(np.min(partials[:, 2]))
    total_max = float(np.max(partials[:, 3]))

    return {
        "count": valid_elements,
        "sum": total_sum,
        "min": total_min,
        "max": total_max,
        "sum_sq": total_sum_sq,
    }


_NORMALIZE_SHADER_SRC = """
struct NormalizeParams {
    min_val: f32,
    max_val: f32,
    count: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> params: NormalizeParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count { return; }
    let range = params.max_val - params.min_val;
    if (range == 0.0) {
        output[idx] = 0.0;
    } else {
        output[idx] = (input[idx] - params.min_val) / range;
    }
}
"""


def normalize_wgpu(data: list[float], min_val: float, max_val: float) -> list[float]:
    _ensure_device()
    if not AVAILABLE:
        raise RuntimeError("WebGPU backend is not available")
    import wgpu

    arr = np.array(data, dtype=np.float32)
    n = len(arr)
    num_workgroups = max(1, math.ceil(n / 256))

    input_bytes = arr.tobytes()
    output_size = n * 4

    params = np.array([min_val, max_val, n], dtype=np.float32)
    info_bytes = params.tobytes()

    buffers = [
        (input_bytes, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST, False),
        (b"\x00" * output_size, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST, True),
        (info_bytes, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST, False),
    ]

    raw = _run_compute_shader(_DEVICE, _NORMALIZE_SHADER_SRC, buffers, num_workgroups, output_size)
    result = np.frombuffer(raw, dtype=np.float32).copy()
    return result.tolist()


_ROLLING_SHADER_SRC = """
struct RollingParams {
    window: u32,
    count: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> params: RollingParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_idx = gid.x;
    let w = params.window;
    let n = params.count;
    let num_windows = n - w + 1u;
    if out_idx >= num_windows { return; }
    for (var j = 0u; j < w; j = j + 1u) {
        output[out_idx * w + j] = input[out_idx + j];
    }
}
"""


def rolling_wgpu(data: list[float], window: int) -> list[float]:
    _ensure_device()
    if not AVAILABLE:
        raise RuntimeError("WebGPU backend is not available")
    import wgpu

    if window > len(data):
        window = len(data)

    arr = np.array(data, dtype=np.float32)
    n = len(arr)
    num_windows = n - window + 1
    num_workgroups = max(1, math.ceil(num_windows / 256))

    input_bytes = arr.tobytes()
    output_size = num_windows * window * 4

    params = np.array([window, n], dtype=np.uint32)
    info_bytes = params.tobytes()

    buffers = [
        (input_bytes, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST, False),
        (b"\x00" * output_size, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST, True),
        (info_bytes, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST, False),
    ]

    raw = _run_compute_shader(_DEVICE, _ROLLING_SHADER_SRC, buffers, num_workgroups, output_size)
    result = np.frombuffer(raw, dtype=np.float32).copy()
    return result.tolist()


def _wgsl_sign_extend(expr: str, bit_width: int) -> str:
    if bit_width >= 32:
        return expr
    shift = 32 - bit_width
    return f"i32(({expr}) << {shift}u) >> {shift}u"


def _generate_extract_expr(cinfo: dict) -> tuple[str, str]:
    """Generate WGSL expression to extract a column value from a Row.

    Returns (wgsl_expr, intermediate_type) where intermediate_type
    is "f32", "u32", or "i32".
    """
    parts = cinfo["parts"]
    dtype = cinfo["dtype"]
    full_size = cinfo["size"]

    if len(parts) == 1:
        _, part_offset, chunk_size = parts[0]
        mask = (1 << chunk_size) - 1
        if chunk_size == 32:
            expr = f"row.part{parts[0][0]}"
        else:
            expr = f"(row.part{parts[0][0]} >> {part_offset}u) & {mask}u"
    else:
        exprs = []
        shift = 0
        for part_idx, part_offset, chunk_size in reversed(parts):
            mask = (1 << chunk_size) - 1
            if chunk_size == 32:
                sub = f"row.part{part_idx}"
            else:
                sub = f"(row.part{part_idx} >> {part_offset}u) & {mask}u"
            if shift > 0:
                sub = f"(({sub}) << {shift}u)"
            shift += chunk_size
            exprs.append(sub)
        expr = " | ".join(exprs)

    if dtype == "float32":
        if cinfo.get("meta", {}).get("_is_compressed_float"):
            meta = cinfo["meta"]
            scale = meta["scale"]
            offset = meta["offset"]
            return (f"f32({expr}) * {scale}f + {offset}f", "f32")
        return f"bitcast<f32>({expr})", "f32"
    elif dtype == "scaled":
        meta = cinfo["meta"]
        scale = meta["scale"]
        offset = meta["offset"]
        min_int = meta["min_int"]
        bit_width = meta.get("bit_width", full_size)
        signed = _wgsl_sign_extend(expr, bit_width)
        return (f"(f32({signed}) - {min_int}f) * {scale}f + {offset}f", "f32")
    elif dtype == "enum":
        return f"f32({expr})", "f32"
    return expr, "u32"


def generate_column_moments_shader(layout: dict, col_name: str) -> str:
    """Generate a WGSL compute shader for moments on a container column.

    Produces a complete shader with Row struct, load_val() using
    JIT bit extraction, and parallel workgroup reduction.

    Args:
        layout: layout dict from compute_layout()
        col_name: which column to extract

    Returns:
        WGSL shader source string
    """
    num_parts = layout["num_parts"]
    cinfo = layout["columns"][col_name]
    extract_expr, _ = _generate_extract_expr(cinfo)

    struct_fields = "\n".join(
        f"    part{i}: u32," for i in range(num_parts)
    )

    shader = f"""
struct Row {{
{struct_fields}
}};

struct Params {{
    valid_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
}};

@group(0) @binding(0) var<storage, read> input: array<Row>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> params: Params;

fn load_val(row: Row) -> f32 {{
    return {extract_expr};
}}

var<workgroup> shmem_sum: array<f32, 256>;
var<workgroup> shmem_sum_sq: array<f32, 256>;
var<workgroup> shmem_min: array<f32, 256>;
var<workgroup> shmem_max: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {{
    let idx = gid.x;
    let valid = params.valid_elements;

    if idx < valid {{
        let val = load_val(input[idx]);
        shmem_sum[lid.x] = val;
        shmem_sum_sq[lid.x] = val * val;
        shmem_min[lid.x] = val;
        shmem_max[lid.x] = val;
    }} else {{
        shmem_sum[lid.x] = 0.0;
        shmem_sum_sq[lid.x] = 0.0;
        shmem_min[lid.x] = 340282346638528859811704183484516925440.0;
        shmem_max[lid.x] = -340282346638528859811704183484516925440.0;
    }}

    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {{
        if (lid.x < stride) {{
            shmem_sum[lid.x] = shmem_sum[lid.x] + shmem_sum[lid.x + stride];
            shmem_sum_sq[lid.x] = shmem_sum_sq[lid.x] + shmem_sum_sq[lid.x + stride];
            shmem_min[lid.x] = min(shmem_min[lid.x], shmem_min[lid.x + stride]);
            shmem_max[lid.x] = max(shmem_max[lid.x], shmem_max[lid.x + stride]);
        }}
        workgroupBarrier();
    }}

    if (lid.x == 0u) {{
        let wg_offset = wg_id.x * 5u;
        output[wg_offset + 0u] = shmem_sum[0];
        output[wg_offset + 1u] = shmem_sum_sq[0];
        output[wg_offset + 2u] = shmem_min[0];
        output[wg_offset + 3u] = shmem_max[0];
        output[wg_offset + 4u] = 0.0;
    }}
}}
"""
    return shader


def generate_i64_moments_shader(num_parts: int) -> str:
    """Generate WGSL compute shader for i64 column moments.

    Accumulates sum as u32 pair with carry for exact 64-bit integer precision.
    Compares min/max with i64 signed comparison logic.
    Output: 6 x f32 per workgroup (sum_lo, sum_hi, min_lo, min_hi, max_lo, max_hi).
    """
    struct_fields = "\n".join(f"    part{i}: u32," for i in range(num_parts))
    return f"""
struct Row {{
{struct_fields}
}};

struct Params {{
    valid_elements: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
    _pad3: u32, _pad4: u32, _pad5: u32, _pad6: u32,
}};

@group(0) @binding(0) var<storage, read> input: array<Row>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> params: Params;

fn is_less_i64(lo_a: u32, hi_a: u32, lo_b: u32, hi_b: u32) -> bool {{
    let hi_a_s = i32(hi_a);
    let hi_b_s = i32(hi_b);
    if (hi_a_s < hi_b_s) {{ return true; }}
    if (hi_a_s > hi_b_s) {{ return false; }}
    return lo_a < lo_b;
}}

var<workgroup> shmem_sum_lo: array<u32, 256>;
var<workgroup> shmem_sum_hi: array<u32, 256>;
var<workgroup> shmem_min_lo: array<u32, 256>;
var<workgroup> shmem_min_hi: array<u32, 256>;
var<workgroup> shmem_max_lo: array<u32, 256>;
var<workgroup> shmem_max_hi: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {{
    let idx = gid.x;
    let valid = params.valid_elements;

    if idx < valid {{
        let row = input[idx];
        let val_lo = row.part1;
        let val_hi = row.part0;

        shmem_sum_lo[lid.x] = val_lo;
        shmem_sum_hi[lid.x] = val_hi;
        shmem_min_lo[lid.x] = val_lo;
        shmem_min_hi[lid.x] = val_hi;
        shmem_max_lo[lid.x] = val_lo;
        shmem_max_hi[lid.x] = val_hi;
    }} else {{
        shmem_sum_lo[lid.x] = 0u;
        shmem_sum_hi[lid.x] = 0u;
        shmem_min_lo[lid.x] = 0xFFFFFFFFu;
        shmem_min_hi[lid.x] = 0x7FFFFFFFu;
        shmem_max_lo[lid.x] = 0u;
        shmem_max_hi[lid.x] = 0x80000000u;
    }}

    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {{
        if (lid.x < stride) {{
            let new_lo = shmem_sum_lo[lid.x] + shmem_sum_lo[lid.x + stride];
            let carry = u32(new_lo < shmem_sum_lo[lid.x]);
            shmem_sum_lo[lid.x] = new_lo;
            shmem_sum_hi[lid.x] = shmem_sum_hi[lid.x] + shmem_sum_hi[lid.x + stride] + carry;

            if (is_less_i64(shmem_min_lo[lid.x + stride], shmem_min_hi[lid.x + stride],
                            shmem_min_lo[lid.x], shmem_min_hi[lid.x])) {{
                shmem_min_lo[lid.x] = shmem_min_lo[lid.x + stride];
                shmem_min_hi[lid.x] = shmem_min_hi[lid.x + stride];
            }}

            if (is_less_i64(shmem_max_lo[lid.x], shmem_max_hi[lid.x],
                            shmem_max_lo[lid.x + stride], shmem_max_hi[lid.x + stride])) {{
                shmem_max_lo[lid.x] = shmem_max_lo[lid.x + stride];
                shmem_max_hi[lid.x] = shmem_max_hi[lid.x + stride];
            }}
        }}
        workgroupBarrier();
    }}

    if (lid.x == 0u) {{
        let wg_offset = wg_id.x * 6u;
        output[wg_offset + 0u] = bitcast<f32>(shmem_sum_lo[0]);
        output[wg_offset + 1u] = bitcast<f32>(shmem_sum_hi[0]);
        output[wg_offset + 2u] = bitcast<f32>(shmem_min_lo[0]);
        output[wg_offset + 3u] = bitcast<f32>(shmem_min_hi[0]);
        output[wg_offset + 4u] = bitcast<f32>(shmem_max_lo[0]);
        output[wg_offset + 5u] = bitcast<f32>(shmem_max_hi[0]);
    }}
}}
"""


# ── Element-wise GPU operations ──────────────────────────────────────

_ARANGE_SRC = """
struct Count { n: u32, _pad0: u32, _pad1: u32, _pad2: u32, };

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> cnt: Count;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= cnt.n) { return; }
    output[idx] = f32(idx);
}
"""

_MUL_SCALAR_SRC = """
struct Params { n: u32, _pad0: u32, scalar: f32, _pad1: u32, };

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.n) { return; }
    output[idx] = input[idx] * params.scalar;
}
"""


def wgpu_arange(n: int) -> np.ndarray:
    """GPU-generated float32 range [0, n)."""
    _ensure_device()
    if not AVAILABLE:
        raise RuntimeError("WebGPU backend is not available")
    import wgpu

    output_size = n * 4
    info_bytes = np.array([n, 0, 0, 0], dtype=np.uint32).tobytes()

    buffers = [
        (b"\x00" * output_size, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST, True),
        (info_bytes, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST, False),
    ]

    num_wg = max(1, (n + 255) // 256)
    raw = _run_compute_shader(_DEVICE, _ARANGE_SRC, buffers, num_wg, output_size)
    return np.frombuffer(raw, dtype=np.float32).copy()


def wgpu_mul_scalar(data: np.ndarray, scalar: float) -> np.ndarray:
    """Multiply array by scalar on GPU."""
    _ensure_device()
    if not AVAILABLE:
        raise RuntimeError("WebGPU backend is not available")
    import struct
    import wgpu

    n = len(data)
    output_size = n * 4
    params = struct.pack("<IIfI", n, 0, scalar, 0)

    buffers = [
        (data.tobytes(), wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST, False),
        (b"\x00" * output_size, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST, True),
        (params, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST, False),
    ]

    num_wg = max(1, (n + 255) // 256)
    raw = _run_compute_shader(_DEVICE, _MUL_SCALAR_SRC, buffers, num_wg, output_size)
    return np.frombuffer(raw, dtype=np.float32).copy()


def _reinterpret_f32_as_u32(val: float) -> int:
    import struct
    return struct.unpack("I", struct.pack("f", val))[0]


def _i64_from_u32_pair(lo: int, hi: int) -> int:
    val = (hi << 32) | lo
    if val >= (1 << 63):
        val -= (1 << 64)
    return val


def compute_column_moments_wgpu(row_parts: list[list[int]],
                                num_parts: int,
                                layout: dict,
                                col_name: str,
                                valid_elements: int) -> dict:
    """Compute moments for a container column via JIT WGSL shader.

    Args:
        row_parts: flat list of u32 row data [p0, p1, ..., p0, p1, ...]
        num_parts: parts per row
        layout: layout dict from compute_layout()
        col_name: which column
        valid_elements: number of valid rows

    Returns:
        moments dict with count, sum, min, max, sum_sq
    """
    _ensure_device()
    if not AVAILABLE:
        raise RuntimeError("WebGPU backend is not available")
    import wgpu

    cinfo = layout["columns"][col_name]
    is_i64 = cinfo.get("dtype") == "int64"

    if is_i64:
        shader_src = generate_i64_moments_shader(num_parts)
        outputs_per_wg = 6
    else:
        shader_src = generate_column_moments_shader(layout, col_name)
        outputs_per_wg = 5

    num_rows = valid_elements
    num_workgroups = max(1, (num_rows + 255) // 256)

    # Build flat u32 array: interleave row parts
    flat = np.array(row_parts, dtype=np.uint32)
    input_bytes = flat.tobytes()
    output_size = num_workgroups * outputs_per_wg * 4

    info_bytes = (
        np.array([num_rows], dtype=np.uint32).tobytes() +
        b'\x00' * 28  # pad to 32 bytes for Params struct
    )

    buffers = [
        (input_bytes, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST, False),
        (b"\x00" * output_size, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST, True),
        (info_bytes, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST, False),
    ]

    raw = _run_compute_shader(_DEVICE, shader_src, buffers, num_workgroups, output_size)
    result = np.frombuffer(raw, dtype=np.float32).copy()
    result = result.reshape(num_workgroups, outputs_per_wg)

    if is_i64:
        # Reconstruct i64 values from u32 pairs stored as f32 bits
        # Intra-workgroup carry handled in WGSL. Inter-workgroup carry
        # must be propagated here: lo overflow → hi increment.
        sum_lo = sum(_reinterpret_f32_as_u32(result[i, 0]) for i in range(num_workgroups))
        sum_hi = sum(_reinterpret_f32_as_u32(result[i, 1]) for i in range(num_workgroups))
        sum_hi += sum_lo >> 32
        total_sum = _i64_from_u32_pair(sum_lo & 0xFFFFFFFF, sum_hi & 0xFFFFFFFF)

        min_lo = min(_reinterpret_f32_as_u32(result[i, 2]) for i in range(num_workgroups))
        min_hi = min(_reinterpret_f32_as_u32(result[i, 3]) for i in range(num_workgroups))
        # Find which workgroup has the minimum i64 value
        tot_min_lo = None
        tot_min_hi = None
        for i in range(num_workgroups):
            lo = _reinterpret_f32_as_u32(result[i, 2])
            hi = _reinterpret_f32_as_u32(result[i, 3])
            if tot_min_lo is None:
                tot_min_lo, tot_min_hi = lo, hi
            else:
                # i64 comparison: compare hi as i32, then lo as u32
                hi_s, tot_hi_s = (hi & 0xFFFFFFFF), (tot_min_hi & 0xFFFFFFFF)
                if hi_s >= (1 << 31): hi_s -= (1 << 32)
                if tot_hi_s >= (1 << 31): tot_hi_s -= (1 << 32)
                if hi_s < tot_hi_s or (hi_s == tot_hi_s and lo < tot_min_lo):
                    tot_min_lo, tot_min_hi = lo, hi
        total_min = _i64_from_u32_pair(tot_min_lo, tot_min_hi)

        max_lo = max(_reinterpret_f32_as_u32(result[i, 4]) for i in range(num_workgroups))
        max_hi = max(_reinterpret_f32_as_u32(result[i, 5]) for i in range(num_workgroups))
        tot_max_lo = None
        tot_max_hi = None
        for i in range(num_workgroups):
            lo = _reinterpret_f32_as_u32(result[i, 4])
            hi = _reinterpret_f32_as_u32(result[i, 5])
            if tot_max_lo is None:
                tot_max_lo, tot_max_hi = lo, hi
            else:
                hi_s, tot_hi_s = (hi & 0xFFFFFFFF), (tot_max_hi & 0xFFFFFFFF)
                if hi_s >= (1 << 31): hi_s -= (1 << 32)
                if tot_hi_s >= (1 << 31): tot_hi_s -= (1 << 32)
                if hi_s > tot_hi_s or (hi_s == tot_hi_s and lo > tot_max_lo):
                    tot_max_lo, tot_max_hi = lo, hi
        total_max = _i64_from_u32_pair(tot_max_lo, tot_max_hi)

        return {
            "count": num_rows,
            "sum": total_sum,
            "min": total_min,
            "max": total_max,
            "sum_sq": 0,
        }

    total_sum = float(np.sum(result[:, 0]))
    total_sum_sq = float(np.sum(result[:, 1]))
    total_min = float(np.min(result[:, 2]))
    total_max = float(np.max(result[:, 3]))

    return {
        "count": num_rows,
        "sum": total_sum,
        "min": total_min,
        "max": total_max,
        "sum_sq": total_sum_sq,
    }


def _generate_raw_extract_expr(cinfo: dict) -> str:
    """Generate WGSL expression to extract raw u32 bits for a column."""
    parts = cinfo["parts"]
    if len(parts) == 1:
        _, part_offset, chunk_size = parts[0]
        mask = (1 << chunk_size) - 1
        if chunk_size == 32:
            return f"(old_buf[row_off + {parts[0][0]}u])"
        return f"(old_buf[row_off + {parts[0][0]}u] >> {part_offset}u) & {mask}u"
    else:
        exprs = []
        shift = 0
        for part_idx, part_offset, chunk_size in reversed(parts):
            mask = (1 << chunk_size) - 1
            if chunk_size == 32:
                sub = f"old_buf[row_off + {part_idx}u]"
            else:
                sub = f"(old_buf[row_off + {part_idx}u] >> {part_offset}u) & {mask}u"
            if shift > 0:
                sub = f"(({sub}) << {shift}u)"
            shift += chunk_size
            exprs.append(sub)
        return " | ".join(exprs)


def _generate_transform_shader(old_num_parts: int, new_num_parts: int,
                                 old_layout: dict, new_layout: dict,
                                 compressed_col: str) -> str:
    """Generate a WGSL compute shader that transforms an old table layout
    into a new tighter layout, compressing one float32 column.

    The shader reads old rows as flat array<u32>, extracts each column
    (applying compression to the target column), and packs into new rows.

    Args:
        old_num_parts: parts per row in old layout
        new_num_parts: parts per row in new layout
        old_layout: layout dict of the old table
        new_layout: layout dict of the new table
        compressed_col: name of the column being compressed

    Returns:
        WGSL shader source string
    """
    # Generate per-column extraction + pack lines
    col_lines = []
    for col in new_layout["schema"]:
        name = col["name"]
        old_cinfo = old_layout["columns"][name]
        new_cinfo = new_layout["columns"][name]
        meta = new_cinfo.get("meta", {})

        if name == compressed_col:
            raw_expr = _generate_raw_extract_expr(old_cinfo)
            scale = meta["scale"]
            offset = meta["offset"]
            bit_width = meta["bit_width"]
            max_val = (1 << bit_width) - 1
            val_expr = (
                "let _v = bitcast<f32>(" + raw_expr + ");\n"
                f"    let _cv = min(u32(round((_v - {offset}f) / {scale}f)), {max_val}u);"
            )
        else:
            raw_expr = _generate_raw_extract_expr(old_cinfo)
            val_expr = f"let _cv = {raw_expr};"

        size = new_cinfo["size"]
        parts = new_cinfo["parts"]
        pack_lines = []
        bits_before = 0
        for part_idx, part_offset, chunk_size in parts:
            mask = (1 << chunk_size) - 1
            shift = size - bits_before - chunk_size
            if shift > 0:
                pv = f"(_cv >> {shift}u)"
            elif shift == 0:
                pv = "_cv"
            else:
                pv = f"(_cv << {-shift}u)"
            pv = f"({pv}) & {mask}u"
            pack_lines.append(
                f"    new_buf[row_off_new + {part_idx}u] |= {pv} << {part_offset}u;"
            )
            bits_before += chunk_size

        col_block = val_expr + "\n" + "\n".join(pack_lines)
        col_lines.append(col_block)

    all_cols = "\n\n".join(col_lines)

    shader = f"""
struct Params {{
    num_rows: u32,
    _pad: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
    _pad7: u32,
}};

const OLD_PARTS = {old_num_parts}u;
const NEW_PARTS = {new_num_parts}u;

@group(0) @binding(0) var<storage, read> old_buf: array<u32>;
@group(0) @binding(1) var<storage, read_write> new_buf: array<u32>;
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx >= params.num_rows {{ return; }}
    let row_off = idx * OLD_PARTS;
    let row_off_new = idx * NEW_PARTS;

{all_cols}
}}
"""
    return shader


def run_transform_wgpu(old_rows: list[list[int]], old_num_parts: int,
                       new_num_parts: int, num_rows: int,
                       old_layout: dict, new_layout: dict,
                       compressed_col: str) -> np.ndarray:
    """Run a JIT transform shader to compress a column on GPU.

    Returns new flat u32 array for the tighter table.
    """
    _ensure_device()
    if not AVAILABLE:
        raise RuntimeError("WebGPU backend is not available")
    import wgpu

    shader_src = _generate_transform_shader(old_num_parts, new_num_parts,
                                            old_layout, new_layout,
                                            compressed_col)

    old_flat = np.array(old_rows, dtype=np.uint32).flatten()
    old_bytes = old_flat.tobytes()
    old_size = len(old_flat) * 4

    new_size = num_rows * new_num_parts * 4
    new_bytes = b"\x00" * new_size

    info_bytes = (
        np.array([num_rows], dtype=np.uint32).tobytes() +
        b'\x00' * 28
    )

    num_workgroups = max(1, (num_rows + 255) // 256)

    buffers = [
        (old_bytes, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST, False),
        (new_bytes, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST, True),
        (info_bytes, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST, False),
    ]

    raw = _run_compute_shader(_DEVICE, shader_src, buffers, num_workgroups, new_size)
    result = np.frombuffer(raw, dtype=np.uint32).copy()
    result = result.reshape(num_rows, new_num_parts)
    return result
