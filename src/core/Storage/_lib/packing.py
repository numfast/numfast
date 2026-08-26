# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Packing — canonical CPU oracle (exact bit-for-bit) copied from src/_core/container.py.

Single source: src/_core/container.py is KEEP, this module is exact copy for
Storage Extension. No duplicate path, one canonical implementation.

Copied functions: compute_layout, pack_rows, pack_rows_np, extract_column, _bits_needed
+ helpers: _pack_*, _unpack_*, _f32_to_u32_bits, _u32_bits_to_f32
"""

import math

CONTAINER_BIT_WIDTH = 32


def _bits_needed(n: int) -> int:
    """Bits needed to represent n distinct values."""
    if n <= 1:
        return 1
    bits = 1
    while (1 << bits) < n:
        bits += 1
    return bits


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
    if len(col_names) == 0:
        return {"rows": [], "num_rows": 0, "num_parts": num_parts, "schema": [], "layout": layout}
    # handle N=0 empty
    first_len = len(data.get(col_names[0], []))
    num_rows = first_len
    if num_rows == 0:
        return {"rows": [], "num_rows": 0, "num_parts": num_parts, "schema": [], "layout": layout}

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
    if len(col_names) == 0:
        return {"rows": np.zeros((0, num_parts), dtype=np.uint32), "num_rows": 0, "num_parts": num_parts, "schema": [], "layout": layout}
    num_rows = len(data[col_names[0]])
    if num_rows == 0:
        return {"rows": np.zeros((0, num_parts), dtype=np.uint32), "num_rows": 0, "num_parts": num_parts, "schema": [], "layout": layout}

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


# --- WGSL shader generator (packed bit-layout, canonical PackingPlan scheme) ---

_INPUT_VARS = {
    'low': 'lo',
    'd_open': 'dO',
    'd_high': 'dH',
    'd_close': 'dC',
    'buy_vol': 'bv',
    'sell_vol': 'sv',
}


def _bits_to_mask(n: int) -> int:
    if n >= 32:
        return 0xFFFFFFFF
    return (1 << n) - 1


def generate_pack_shader(layout: dict) -> str:
    """Generate WGSL compute shader that packs OHLCV into PackingPlan bits.

    Args:
        layout: layout dict from compute_layout() (or pack_rows result)
    Returns:
        WGSL shader source string.
    Adapted for PackingPlan: uses 6 inputs, Config low_offset,n, workgroup 1024,
    parts mask/shift, out[i*num_parts+pi]=p_pi
    """
    cols = layout['columns']
    num_parts = layout['num_parts']
    sorted_names = ['low', 'd_open', 'd_high', 'd_close', 'buy_vol', 'sell_vol']
    lines = []
    lines.append('struct Config { low_offset: i32, n: u32, }')
    lines.append('@group(0) @binding(0) var<storage, read> low: array<i32>;')
    lines.append('@group(0) @binding(1) var<storage, read> d_open: array<i32>;')
    lines.append('@group(0) @binding(2) var<storage, read> d_high: array<i32>;')
    lines.append('@group(0) @binding(3) var<storage, read> d_close: array<i32>;')
    lines.append('@group(0) @binding(4) var<storage, read> buy_vol: array<u32>;')
    lines.append('@group(0) @binding(5) var<storage, read> sell_vol: array<u32>;')
    lines.append('@group(0) @binding(6) var<uniform> config: Config;')
    lines.append(f'@group(0) @binding(7) var<storage, read_write> out: array<u32>;')
    lines.append('')
    lines.append('@compute @workgroup_size(1024)')
    lines.append('fn main(@builtin(global_invocation_id) gid: vec3<u32>) {')
    lines.append('    let i = gid.x;')
    lines.append('    if (i >= config.n) { return; }')
    lines.append('')
    lines.append('    let lo = u32(low[i] - config.low_offset);')
    lines.append('    let dO = u32(d_open[i]);')
    lines.append('    let dH = u32(d_high[i]);')
    lines.append('    let dC = u32(d_close[i]);')
    lines.append('    let bv = buy_vol[i];')
    lines.append('    let sv = sell_vol[i];')
    lines.append('')
    for pi in range(num_parts):
        lines.append(f'    var p{pi}: u32 = 0u;')
    lines.append('')
    for col_name in sorted_names:
        if col_name not in cols:
            continue
        cinfo = cols[col_name]
        v = _INPUT_VARS[col_name]
        total_bits = cinfo['size']
        remaining = total_bits
        for part_idx, part_offset, chunk_size in cinfo['parts']:
            remaining -= chunk_size
            if chunk_size == 0:
                continue
            if remaining > 0:
                shift_right = remaining
                mask = _bits_to_mask(chunk_size)
                expr = f'(({v} >> {shift_right}u) & 0x{mask:X}u)'
            else:
                mask = _bits_to_mask(chunk_size)
                if mask == 0xFFFFFFFF:
                    expr = v
                else:
                    expr = f'({v} & 0x{mask:X}u)'
            if part_offset > 0:
                expr = f'({expr} << {part_offset}u)'
            lines.append(f'    p{part_idx} |= {expr};')
    lines.append('')
    for pi in range(num_parts):
        lines.append(f'    out[i * {num_parts}u + {pi}u] = p{pi};')
    lines.append('}')
    return '\n'.join(lines)


def pack_on_cpu_for_verify(data: dict, layout: dict) -> list:
    """Simulate GPU packing on CPU to verify correctness."""
    cols = layout['columns']
    num_parts = layout['num_parts']
    n = len(data[list(data.keys())[0]]) if len(data) > 0 and len(list(data.values())[0]) > 0 else 0
    if n == 0:
        return []
    sorted_names = ['low', 'd_open', 'd_high', 'd_close', 'buy_vol', 'sell_vol']
    result = []
    for i in range(n):
        parts = [0] * num_parts
        for col_name in sorted_names:
            if col_name not in cols:
                continue
            cinfo = cols[col_name]
            val = int(data[col_name][i])
            total_bits = cinfo['size']
            remaining = total_bits
            for part_idx, part_offset, chunk_size in cinfo['parts']:
                remaining -= chunk_size
                if chunk_size == 0:
                    continue
                if remaining > 0:
                    chunk = (val >> remaining) & _bits_to_mask(chunk_size)
                else:
                    chunk = val & _bits_to_mask(chunk_size)
                parts[part_idx] |= (chunk << part_offset) & 0xFFFFFFFF
        result.append(parts)
    return result


# --- Legacy PackingPlan API (KEEP for Table/Column compatibility, pre-S110) ---
# Original Storage/_lib/packing.py — ColumnLayout, PackingPlan, compute_packing_plan
# Required for Storage.table/column/table tests and nf.table alias.

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ColumnLayout:
    """Physical layout of one column within a packed row (legacy)."""
    name: str
    bit_offset: int
    bit_width: int
    scale: float = 1.0
    offset: float = 0.0
    signed: bool = False
    categories: Optional[list] = None
    parts: tuple = ()


@dataclass(frozen=True)
class PackingPlan:
    """Immutable plan describing bit layout of all columns (legacy)."""
    columns: tuple
    num_parts: int
    bytes_per_row: int

    def get_layout(self, name: str) -> Optional[ColumnLayout]:
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def __repr__(self):
        parts = []
        for c in self.columns:
            parts.append(f"  {c.name}: offset={c.bit_offset} bits={c.bit_width} "
                         f"parts={len(c.parts)} scale={c.scale} offset={c.offset}")
        return f"PackingPlan({self.num_parts} parts, {self.bytes_per_row}B/row):\n" + "\n".join(parts)


def compute_packing_plan(columns):
    """Compute bit layout with cross-boundary split (legacy).

    Args:
        columns: list of Column objects

    Returns:
        PackingPlan describing the bit layout
    """
    PART_BITS = 32
    layouts = []
    next_bit = 0

    for col in columns:
        bw = col._bit_width
        remaining = bw
        parts = []
        bit_start = next_bit

        while remaining > 0:
            part_idx = next_bit // PART_BITS
            part_offset = next_bit % PART_BITS
            available = PART_BITS - part_offset
            chunk = min(remaining, available)
            parts.append((part_idx, part_offset, chunk))
            next_bit += chunk
            remaining -= chunk

        layouts.append(ColumnLayout(
            name=col.name,
            bit_offset=bit_start,
            bit_width=bw,
            scale=col.scale,
            offset=col.offset,
            signed=(col.dtype.name in ("SCALED", "INT8", "INT16")),
            categories=col.categories if col.dtype.name == "ENUM" else None,
            parts=tuple(parts),
        ))

    num_parts = (next_bit + PART_BITS - 1) // PART_BITS
    bytes_per_row = num_parts * 4
    return PackingPlan(columns=tuple(layouts), num_parts=num_parts, bytes_per_row=bytes_per_row)
