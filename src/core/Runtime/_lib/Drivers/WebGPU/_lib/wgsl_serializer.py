"""WebGPU Serializer — BlockView <-> bytes for GPU.

Supports float32 and uint32 data transfer.
"""

import struct


def blockview_to_bytes(bv, dtype="float") -> bytes:
    """Read BlockView and pack to bytes.

    Args:
        bv: BlockView
        dtype: "float" (float32) or "uint" (uint32)

    Returns:
        bytes for wgpu buffer upload
    """
    n = bv.length()
    if dtype == "uint":
        return struct.pack(f'{n}I', *[int(bv.read(i)) for i in range(n)])
    # float32 (default)
    return struct.pack(f'{n}f', *[bv.read(i) for i in range(n)])


def bytes_to_blockview(data: bytes, bv, dtype="float"):
    """Write bytes from wgpu buffer into BlockView.

    Args:
        data: bytes from GPU read_buffer
        bv: BlockView for writing
        dtype: "float" (float32) or "uint" (uint32)
    """
    n = bv.length()
    elem_size = 4  # both f32 and u32 are 4 bytes
    count = min(n, len(data) // elem_size)
    if dtype == "uint":
        fmt = f'{count}I'
        values = struct.unpack(fmt, data[:count * elem_size])
        for i, v in enumerate(values):
            bv.write(i, float(v))
    else:
        fmt = f'{count}f'
        values = struct.unpack(fmt, data[:count * elem_size])
        for i, v in enumerate(values):
            bv.write(i, float(v))


def pack_uniforms(uniforms: dict) -> bytes:
    """Упаковать uniform-параметры в bytes для WGSL uniform buffer.

    Все значения пакуются как float32 — WGSL uniform structs в этом проекте
    используют f32 для всех полей.
    Выравнивание: 16 байт (WGSL struct minimum alignment).
    """
    packed = b''
    for k, v in uniforms.items():
        packed += struct.pack('<f', float(v))
    # Pad to 16-byte alignment
    pad = (16 - (len(packed) % 16)) % 16
    if pad:
        packed += b'\x00' * pad
    return packed
