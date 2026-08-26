# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Storage Extension — P0-S dzst ALL rows. Builder entry point."""

# Legacy table/column API (KEEP for nf.table compatibility)
def table(data, columns=None):
    from Storage._lib.table import Table as _Table
    if columns is None and isinstance(data, dict):
        from Storage._lib.column import Column, ColumnType
        columns = [Column(name, ColumnType.INT32) for name in data.keys()]
    return _Table(columns, data)


def column(name, dtype="float"):
    from Storage._lib.column import Column as _Column
    return _Column(name, dtype)


def compute_plan(columns):
    from Storage._lib.packing import compute_packing_plan as _plan
    return _plan(columns)


def optimize_table(tbl):
    from Storage._lib.optimize import optimize as _opt
    return _opt(tbl)


def column_layout(plan, col_name):
    from Storage._lib.packing import ColumnLayout as _CL
    return _CL(plan, col_name)


def packing_plan(columns):
    from Storage._lib.packing import PackingPlan as _PP
    return _PP(columns)


def Table(data, columns=None):
    from Storage._lib.table import Table as _T
    if columns is None and isinstance(data, dict):
        from Storage._lib.column import Column, ColumnType
        columns = [Column(name, ColumnType.INT32) for name in data.keys()]
    return _T(columns, data)


def Column(name, dtype="float"):
    from Storage._lib.column import Column as _C
    return _C(name, dtype)


def ColumnType():
    from Storage._lib.column import ColumnType as _CT
    return _CT


def PackingPlan(columns):
    from Storage._lib.packing import PackingPlan as _PP
    return _PP(columns)


def ColumnLayout(plan, col_name):
    from Storage._lib.packing import ColumnLayout as _CL
    return _CL(plan, col_name)


# P0-S storage alias — exposed via nf.storage (callable) and kernel.storage dict
def storage():
    """Return canonical storage oracle dict (compute_layout etc)."""
    from Storage._lib.packing import compute_layout, pack_rows, pack_rows_np, extract_column
    return {
        "compute_layout": compute_layout,
        "pack_rows": pack_rows,
        "pack_rows_np": pack_rows_np,
        "extract_column": extract_column,
    }


def setup(kernel):
    kernel.metadata.setdefault("Storage", {})
    kernel.metadata["Storage"]["version"] = "0.1.0"
    kernel.metadata["Storage"]["description"] = "P0-S dzst ALL rows"
    kernel.metadata["Storage"]["types"] = ["storage"]
    kernel.metadata["Storage"]["alias"] = "storage"
    # expose canonical CPU oracle via kernel.storage for direct access
    try:
        from Storage._lib.packing import compute_layout, pack_rows, pack_rows_np, extract_column
        kernel.storage = {
            "compute_layout": compute_layout,
            "pack_rows": pack_rows,
            "pack_rows_np": pack_rows_np,
            "extract_column": extract_column,
        }
    except Exception:
        pass
