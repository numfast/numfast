# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from _core.backend import get_active_name


def _container_table_create(schema: list[dict], data: dict) -> dict:
    """Create a container table with bit-packed row storage.

    Args:
        schema: list of column dicts (name, dtype, bit_width, scale, offset, etc.)
        data: {col_name: [values]}

    Returns:
        table dict with _table_id and metadata
    """
    from _core.container import pack_rows
    packed = pack_rows(schema, data)
    from _core import kernel as _kernel
    tid, _ = _kernel._register_table(packed["rows"], packed["num_parts"],
                                     packed["schema"], packed["layout"],
                                     created_by="table")
    tbl = _kernel._ProxyDict(
        _table_id=tid,
        _num_rows=packed["num_rows"],
        _num_parts=packed["num_parts"],
        _schema=packed["schema"],
        _layout=packed["layout"],
    )
    return tbl


def _get_column(table: dict, col_name: str) -> dict:
    """Return a lightweight column proxy dict for a container table.

    The proxy contains no data — only references for JIT WGSL extraction.
    Stats functions detect this proxy and dispatch to the JIT shader.
    """
    if col_name not in table.get("_layout", {}).get("columns", {}):
        raise KeyError(f"Column '{col_name}' not found in table schema")
    cinfo = table["_layout"]["columns"][col_name]
    from _core import kernel as _kernel
    col = _kernel._ProxyDict(
        _table_id=table["_table_id"],
        _col_name=col_name,
        _col_schema=cinfo,
        _layout=table["_layout"],
        _length=table["_num_rows"],
    )
    return col


def _compress_column(table: dict, col_name: str,
                     target_bits: int, scale: float, offset: float) -> dict:
    """Dynamically compress a raw float32 column to N-bit scaled in a new table.

    Creates a new table with the target column compressed. The old table
    is not modified. On wgpu backend, uses a JIT transform shader on GPU.

    Args:
        table: existing container table dict
        col_name: column to compress (must be raw float32)
        target_bits: target bit width (e.g., 12)
        scale: quantization scale factor
        offset: quantization offset

    Returns:
        new container table dict with compressed column
    """
    from _core import kernel as _kernel
    entry = _kernel._get_table_entry(table["_table_id"])
    if entry is None:
        raise ValueError("Table not found in kernel")
    old_layout = entry["layout"]
    old_rows = entry["rows"]
    old_num_parts = entry["num_parts"]
    num_rows = entry["num_rows"]

    # Build new schema with compression on the target column
    new_schema = []
    for col in old_layout["schema"]:
        c = dict(col)
        if c["name"] == col_name:
            c["compression"] = {"scaled": True, "bits": target_bits,
                                "scale": scale, "offset": offset}
        new_schema.append(c)

    if get_active_name() == "wgpu":
        from _core.backend._wgpu import run_transform_wgpu
        from _core.container import compute_layout
        new_layout = compute_layout(new_schema)
        new_num_parts = new_layout["num_parts"]
        result_arr = run_transform_wgpu(
            old_rows, old_num_parts, new_num_parts, num_rows,
            old_layout, new_layout, col_name,
        )
        new_rows = result_arr.tolist()
        tid, _ = _kernel._register_table(new_rows, new_num_parts,
                                          new_layout["schema"], new_layout)
        return {
            "_table_id": tid,
            "_num_rows": num_rows,
            "_num_parts": new_num_parts,
            "_schema": new_layout["schema"],
            "_layout": new_layout,
        }

    # CPU fallback
    from _core.container import extract_column, pack_rows
    col_vals = extract_column(old_rows, col_name, old_layout)
    all_data = {}
    for col in old_layout["schema"]:
        name = col["name"]
        if name == col_name:
            all_data[name] = col_vals
        else:
            all_data[name] = extract_column(old_rows, name, old_layout)
    packed = pack_rows(new_schema, all_data)
    tid, _ = _kernel._register_table(packed["rows"], packed["num_parts"],
                                     packed["schema"], packed["layout"])
    return {
        "_table_id": tid,
        "_num_rows": packed["num_rows"],
        "_num_parts": packed["num_parts"],
        "_schema": packed["schema"],
        "_layout": packed["layout"],
    }


def _column_view(table: dict, col_name: str):
    """Return a ColumnView for zero-copy index access to a column.

    Reads values directly from the bit-packed container via mask+shift,
    without extracting all values to a Python list.

    Args:
        table: container table dict (from container_table_create)
        col_name: column name

    Returns:
        ColumnView instance
    """
    from _core import kernel as _kernel
    from _core.column_view import ColumnView
    entry = _kernel._get_table_entry(table["_table_id"])
    if entry is None:
        raise ValueError("Table not found in kernel")
    if col_name not in entry["layout"]["columns"]:
        raise KeyError(f"Column '{col_name}' not found")
    return ColumnView(entry["rows"], col_name, entry["layout"])


def _container_rows(table_id: str) -> tuple[list[list[int]], int, int] | None:
    """Retrieve container row data from kernel.

    Returns (row_parts, num_parts, num_rows) or None.
    """
    from _core import kernel as _kernel
    entry = _kernel._get_table_entry(table_id)
    if entry is None:
        return None
    return entry["rows"], entry["num_parts"], entry["num_rows"]



