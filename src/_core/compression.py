# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

import math


def _f32(val: float) -> float:
    return float(val)


def pack_scaled(data: list[float], target_dtype: str = "auto") -> tuple[list[int], dict]:
    """Pack float data into int8 or int16 with scale/offset.

    Args:
        data: input float values
        target_dtype: "int8", "int16", or "auto" (picks smallest)

    Returns:
        (packed_values, metadata) where metadata = {
            "type": "scaled",
            "scale": float,
            "offset": float,
            "dtype": "int8" | "int16",
            "min": float,
            "max": float,
        }
    """
    if not data:
        return [], {"type": "scaled", "scale": 1.0, "offset": 0.0, "min_int": -128.0, "dtype": "int8", "min": 0.0, "max": 0.0}

    mn = min(data)
    mx = max(data)
    data_range = mx - mn

    if data_range == 0.0:
        return [0] * len(data), {"type": "scaled", "scale": 1.0, "offset": _f32(mn), "min_int": 0.0, "dtype": "int8", "min": _f32(mn), "max": _f32(mx)}

    ranges = {"int8": (-128.0, 127.0), "int16": (-32768.0, 32767.0)}

    if target_dtype == "auto":
        min_8, max_8 = ranges["int8"]
        target_dtype = "int8" if data_range <= (max_8 - min_8) else "int16"

    min_int, max_int = ranges[target_dtype]
    full_span = max_int - min_int
    scale = data_range / full_span
    offset = _f32(mn)
    packed = [round((v - offset) / scale + min_int) for v in data]

    return packed, {
        "type": "scaled",
        "scale": _f32(scale),
        "offset": offset,
        "min_int": float(min_int),
        "dtype": target_dtype,
        "min": _f32(mn),
        "max": _f32(mx),
    }


def unpack_scaled(packed: list[int], metadata: dict) -> list[float]:
    """Restore floats from scaled int data."""
    scale = metadata["scale"]
    offset = metadata["offset"]
    min_int = metadata["min_int"]
    return [(float(v) - min_int) * scale + offset for v in packed]


def pack_enum(data: list[str | None]) -> tuple[list[int], dict]:
    """Pack string categories into compact integer enum.

    Slot assignment:
      0..1       — first two valid values
      2          — NULL (always reserved, never overlaps with values)
      3..N+2     — remaining valid values

    This guarantees NULL always maps to 2 regardless of vocabulary size.

    Args:
        data: list of strings or None

    Returns:
        (packed_values, metadata)
    """
    unique = sorted(set(d for d in data if d is not None))
    null_slot = 2
    slot_map = {}
    next_slot = 0
    for val in unique:
        if next_slot == null_slot:
            next_slot += 1
        slot_map[val] = next_slot
        next_slot += 1

    vocab = {slot_map[val]: val for val in unique}
    bit_width = max(2, math.ceil(math.log2(next_slot + 1)))

    packed = []
    for item in data:
        if item is None:
            packed.append(null_slot)
        else:
            packed.append(slot_map[item])

    return packed, {
        "type": "enum",
        "vocab": vocab,
        "null_slot": null_slot,
        "bit_width": bit_width,
        "dtype": "int8",
    }


def unpack_enum(packed: list[int], metadata: dict) -> list[str | None]:
    """Restore strings from packed enum values."""
    vocab = metadata["vocab"]
    null_slot = metadata["null_slot"]
    result = []
    for val in packed:
        if val == null_slot:
            result.append(None)
        else:
            result.append(vocab.get(val, f"UNKNOWN_{val}"))
    return result
