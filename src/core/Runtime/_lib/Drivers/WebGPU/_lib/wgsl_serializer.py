"""WebGPU Serializer — BlockView <-> bytes for GPU.

Supports float32 and uint32 data transfer.
"""

import struct

import numpy as np


def blockview_to_bytes(bv, dtype="float") -> bytes:
    """Read BlockView and pack to bytes.

    Args:
        bv: BlockView
        dtype: "float" (float32) or "uint" (uint32) or "int" (int32)

    Returns:
        bytes for wgpu buffer upload
    """
    n = bv.length()
    raw = bv._raw[:n]
    if dtype in ("uint", "uint32"):
        return raw.astype(np.uint32, copy=False).tobytes()
    if dtype in ("int", "int32"):
        return raw.astype(np.int32, copy=False).tobytes()
    # float32 (default)
    if raw.dtype != np.float32:
        return raw.astype(np.float32, copy=False).tobytes()
    return raw.tobytes()


def bytes_to_blockview(data: bytes, bv, dtype="float"):
    """Write bytes from wgpu buffer into BlockView.

    Args:
        data: bytes from GPU read_buffer
        bv: BlockView for writing
        dtype: "float" (float32) or "uint" (uint32) or "int" (int32)
    """
    elem_size = 4  # both f32 and u32/i32 are 4 bytes
    count = min(bv.length(), len(data) // elem_size)
    if count <= 0:
        return
    raw = bv._raw
    if dtype in ("uint", "uint32"):
        vals = np.frombuffer(data, dtype=np.uint32, count=count)
    elif dtype in ("int", "int32"):
        vals = np.frombuffer(data, dtype=np.int32, count=count)
    else:
        vals = np.frombuffer(data, dtype=np.float32, count=count)
    if vals.dtype != raw.dtype:
        vals = vals.astype(raw.dtype)
    raw[:count] = vals


def pack_uniforms(uniforms: dict) -> bytes:
    """Упаковать uniform-параметры в bytes для WGSL uniform buffer.

    int -> u32 (<I), float -> f32 (<f), bool -> u32
    Выравнивание: 16 байт (WGSL struct minimum alignment).
    Это поддерживает vec4<u32> (fused) и f32 uniforms (legacy kernels).
    """
    packed = b''
    for k, v in uniforms.items():
        if isinstance(v, bool):
            packed += struct.pack('<I', int(v))
        elif isinstance(v, int):
            # WGSL vec4<u32> expects unsigned int bits
            packed += struct.pack('<I', v & 0xFFFFFFFF)
        else:
            packed += struct.pack('<f', float(v))
    # Pad to 16-byte alignment
    pad = (16 - (len(packed) % 16)) % 16
    if pad:
        packed += b'\x00' * pad
    return packed
