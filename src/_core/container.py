# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

import math

CONTAINER_BIT_WIDTH = 32


def _resolve_schema(schema: list[dict]) -> list[dict]:
    """Normalize schema: expand float32+compression into effective column defs."""
    resolved = []
    for col in schema:
        c = dict(col)
        comp = col.get("compression")
        if col["dtype"] == "float32" and comp and comp.get("scaled"):
            c["bit_width"] = comp["bits"]
            c["_is_compressed_float"] = True
        elif col["dtype"] == "int64" and col.get("bits"):
            c["bit_width"] = col["bits"]
        resolved.append(c)
    return resolved


def compute_layout(schema: list[dict]) -> dict:
    """Compute row struct layout from column schema.

    Each column packs sequentially without gaps. Columns may cross
    primitive (u32) boundaries. Returns layout dict used for both
    CPU pack/unpack and JIT WGSL shader generation.

    Args:
        schema: list of column dicts with keys:
            name (str), dtype (str), bit_width (int, optional, default 32 for float32),
            scale/offset/min_int (for scaled), vocab/null_slot (for enum),
            compression (dict, optional for float32)

    Returns:
        layout dict:
            num_parts (int)
            columns (dict): name -> {bit_offset, size, parts, dtype, meta}
    """
    bits_per_part = CONTAINER_BIT_WIDTH
    columns = {}
    next_bit = 0
    resolved = _resolve_schema(schema)

    for col in resolved:
        name = col["name"]
        dtype = col["dtype"]
        size = col.get("bit_width", 32)

        parts = []
        remaining = size
        offset_in_row = next_bit
        while remaining > 0:
            part_idx = next_bit // bits_per_part
            part_start = part_idx * bits_per_part
            part_offset = next_bit - part_start
            available = bits_per_part - part_offset
            chunk = min(remaining, available)
            parts.append((part_idx, part_offset, chunk))
            next_bit += chunk
            remaining -= chunk

        meta = {}
        if col.get("_is_compressed_float"):
            comp = col["compression"]
            meta = {"scale": comp["scale"], "offset": comp["offset"],
                    "bit_width": comp["bits"], "_is_compressed_float": True}
        elif dtype == "scaled":
            meta = {"scale": col["scale"], "offset": col["offset"],
                    "min_int": col.get("min_int", 0.0),
                    "bit_width": col.get("bit_width", 16)}
        elif dtype == "enum":
            meta = {"vocab": col.get("vocab", {}),
                    "null_slot": col.get("null_slot", 2)}

        columns[name] = {
            "bit_offset": offset_in_row,
            "size": size,
            "parts": parts,
            "dtype": dtype,
            "meta": meta,
        }

    total_bits = next_bit
    num_parts = (total_bits + bits_per_part - 1) // bits_per_part

    return {
        "num_parts": num_parts,
        "columns": columns,
        "schema": resolved,
    }


def pack_rows(schema: list[dict], data: dict) -> dict:
    """Pack columnar data into container row parts.

    Args:
        schema: column definitions
        data: {col_name: [values]}

    Returns:
        {"rows": [[p0, p1, ...], ...], "num_rows": N, "num_parts": P,
         "schema": cleaned_schema, "layout": layout}
    """
    layout = compute_layout(schema)
    num_parts = layout["num_parts"]
    col_names = [c["name"] for c in layout["schema"]]
    num_rows = len(data[col_names[0]])

    cleaned_schema = []
    rows = []
    for ri in range(num_rows):
        row_parts = [0] * num_parts
        for col in layout["schema"]:
            name = col["name"]
            dtype = col["dtype"]
            val = data[name][ri]
            cinfo = layout["columns"][name]

            if col.get("_is_compressed_float"):
                comp = col["compression"]
                raw = _pack_compressed_float(val, comp["scale"], comp["offset"], comp["bits"])
            elif dtype == "float32":
                raw = _f32_to_u32_bits(float(val))
            elif dtype == "scaled":
                raw = _pack_scaled(val, col)
            elif dtype == "enum":
                raw = _pack_enum(val, col)
            elif dtype == "int64":
                raw = int(val)
            else:
                raise ValueError(f"Unknown dtype: {dtype}")

            bits_left = cinfo["size"]
            for part_idx, part_offset, chunk_size in cinfo["parts"]:
                mask = (1 << chunk_size) - 1
                chunk = (raw >> (bits_left - chunk_size)) & mask
                row_parts[part_idx] = (row_parts[part_idx] | (chunk << part_offset)) & 0xFFFFFFFF
                bits_left -= chunk_size

            if ri == 0:
                col_for_schema = dict(col)
                col_for_schema.pop("_is_compressed_float", None)
                cleaned_schema.append(col_for_schema)

        rows.append(row_parts)

    return {
        "rows": rows,
        "num_rows": num_rows,
        "num_parts": num_parts,
        "schema": cleaned_schema,
        "layout": layout,
    }


def pack_rows_np(schema: list[dict], data: dict) -> dict:
    """Vectorized pack_rows via NumPy. Same interface as pack_rows().

    data: dict of {col_name: numpy int64 array}
    Returns: same format as pack_rows(), but rows is a numpy uint32 array (N, num_parts)
    """
    import numpy as np

    layout = compute_layout(schema)
    num_parts = layout["num_parts"]
    col_names = [c["name"] for c in layout["schema"]]
    num_rows = len(data[col_names[0]])

    rows_np = np.zeros((num_rows, num_parts), dtype=np.uint32)

    cleaned_schema = []

    for col in layout["schema"]:
        name = col["name"]
        dtype = col["dtype"]
        cinfo = layout["columns"][name]
        vals = np.asarray(data[name], dtype=np.int64)

        if col.get("_is_compressed_float"):
            comp = col["compression"]
            scale = comp["scale"]
            offset = comp["offset"]
            bit_width = comp["bits"]
            raw_vals = np.round((vals.astype(np.float64) - offset) / scale).astype(np.int64)
            max_val = (1 << bit_width) - 1
            raw_vals = np.clip(raw_vals, 0, max_val)
        elif dtype == "float32":
            raise NotImplementedError("float32 not supported in vectorized pack")
        elif dtype == "scaled":
            scale = col.get("scale", 1.0)
            offset = col.get("offset", 0.0)
            min_int = col.get("min_int", 0.0)
            bit_width = col.get("bit_width", 16)
            raw_vals = np.round((vals.astype(np.float64) - offset) / scale + min_int).astype(np.int64)
            mask_val = (1 << bit_width) - 1
            raw_vals = raw_vals & mask_val
        elif dtype == "enum":
            raise NotImplementedError("enum not supported in vectorized pack")
        elif dtype == "int64":
            raw_vals = vals.copy()
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

        bits_left = cinfo["size"]
        for part_idx, part_offset, chunk_size in cinfo["parts"]:
            mask = np.uint32((1 << chunk_size) - 1)
            chunk = ((raw_vals >> (bits_left - chunk_size)) & mask).astype(np.uint32)
            rows_np[:, part_idx] |= (chunk << np.uint32(part_offset))
            bits_left -= chunk_size

        cleaned_schema.append(col)

    return {
        "rows": rows_np,
        "num_rows": num_rows,
        "num_parts": num_parts,
        "schema": cleaned_schema,
        "layout": layout,
    }


def pack_rows_to_flat(schema, data):
    """Pack and flatten to 1D uint32 array for GPU transfer."""
    result = pack_rows_np(schema, data)
    flat = result["rows"].reshape(-1)
    return flat, result


def extract_column(rows, col_name: str, layout: dict) -> list:
    """Extract a single column from container rows (CPU fallback).

    Returns Python list of native values (float, str, or None).
    """
    cinfo = layout["columns"][col_name]
    dtype = cinfo["dtype"]
    meta = cinfo["meta"]
    result = []

    for row in rows:
        raw = 0
        bits_collected = 0
        for part_idx, part_offset, chunk_size in cinfo["parts"]:
            mask = (1 << chunk_size) - 1
            chunk = (int(row[part_idx]) >> part_offset) & mask
            raw = (raw << chunk_size) | chunk
            bits_collected += chunk_size

        if dtype == "float32":
            if meta.get("_is_compressed_float"):
                result.append(_unpack_compressed_float(raw, meta))
            else:
                result.append(_u32_bits_to_f32(raw))
        elif dtype == "scaled":
            result.append(_unpack_scaled(raw, meta))
        elif dtype == "enum":
            result.append(_unpack_enum(raw, meta))
        elif dtype == "int64":
            if raw >= (1 << 63):
                raw -= (1 << 64)
            result.append(raw)
        else:
            result.append(float(raw))

    return result


def _pack_compressed_float(val: float, scale: float, offset: float, bit_width: int) -> int:
    raw_int = round((val - offset) / scale)
    max_val = (1 << bit_width) - 1
    return max(0, min(max_val, int(raw_int)))


def _unpack_compressed_float(raw: int, meta: dict) -> float:
    scale = meta["scale"]
    offset = meta["offset"]
    return float(raw) * scale + offset


def _f32_to_u32_bits(val: float) -> int:
    import struct
    return struct.unpack("I", struct.pack("f", val))[0]


def _u32_bits_to_f32(bits: int) -> float:
    import struct
    return struct.unpack("f", struct.pack("I", bits & 0xFFFFFFFF))[0]


def _pack_scaled(val: float, col: dict) -> int:
    scale = col["scale"]
    offset = col["offset"]
    min_int = col.get("min_int", 0.0)
    bit_width = col.get("bit_width", 16)
    raw_int = round((val - offset) / scale + min_int)
    return int(raw_int) & ((1 << bit_width) - 1)


def _unpack_scaled(raw: int, meta: dict) -> float:
    scale = meta["scale"]
    offset = meta["offset"]
    min_int = meta["min_int"]
    bit_width = meta.get("bit_width", 16)
    if bit_width < 32:
        sign_bit = 1 << (bit_width - 1)
        if raw & sign_bit:
            raw -= (1 << bit_width)
    return (float(raw) - min_int) * scale + offset


def _pack_enum(val: str | None, col: dict) -> int:
    vocab = col.get("vocab", {})
    null_slot = col.get("null_slot", 2)
    if val is None:
        return null_slot
    for slot, label in vocab.items():
        if label == val:
            return slot
    return 0


def _unpack_enum(raw: int, meta: dict) -> str | None:
    vocab = meta.get("vocab", {})
    null_slot = meta.get("null_slot", 2)
    if raw == null_slot:
        return None
    return vocab.get(raw, f"UNKNOWN_{raw}")
