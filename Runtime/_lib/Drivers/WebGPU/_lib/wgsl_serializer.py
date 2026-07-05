"""WebGPU Serializer — преобразование BlockView ↔ bytes для GPU.

Без numpy. Только struct и базовые типы Python.

Каждый Driver имеет свой Serializer:
  - CpuSerializer: BlockView ↔ numpy array (эталон, no-op)
  - WebGpuSerializer: BlockView ↔ bytes (для wgpu buffer upload/download)
  - CudaSerializer: BlockView ↔ CUDA memory (будущее)
"""

import struct
from typing import Union
from ....mod_iface import BlockView


def blockview_to_bytes(bv: BlockView) -> bytes:
    """Прочитать BlockView и упаковать в bytes (float32).

    Args:
        bv: BlockView с float-данными

    Returns:
        bytes: бинарное представление для загрузки в wgpu буфер
    """
    n = bv.length()
    # Упаковываем каждый float через read()
    return struct.pack(f'{n}f', *[bv.read(i) for i in range(n)])


def bytes_to_blockview(data: bytes, bv: BlockView):
    """Записать bytes из wgpu буфера в BlockView.

    Args:
        data: bytes из GPU read_buffer
        bv: BlockView для записи
    """
    n = bv.length()
    count = min(n, len(data) // 4)
    fmt = f'{count}f'
    values = struct.unpack(fmt, data[:count * 4])
    for i, v in enumerate(values):
        bv.write(i, float(v))


def pack_uniforms(uniforms: dict) -> bytes:
    """Упаковать uniform-параметры в bytes для WGSL uniform buffer.

    Все значения пакуются последовательно:
    - bool/int → int32 (4 байта)
    - float → float32 (4 байта)
    - выравнивание: 16 байт (WGSL struct minimum alignment)
    """
    packed = b''
    for k, v in uniforms.items():
        if isinstance(v, bool):
            packed += struct.pack('<i', int(v))
        elif isinstance(v, int):
            packed += struct.pack('<i', v)
        else:
            packed += struct.pack('<f', v)
    # Pad to 16-byte alignment
    pad = (16 - (len(packed) % 16)) % 16
    if pad:
        packed += b'\x00' * pad
    return packed
