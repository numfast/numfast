# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only


def container_table(schema: list[dict], data: dict) -> dict:
    from Tables._lib.tables_lib import _container_table_create
    return _container_table_create(schema, data)


def get_column(table: dict, name: str) -> dict:
    from Tables._lib.tables_lib import _get_column
    return _get_column(table, name)


def compress_column(table: dict, col_name: str,
                    target_bits: int, scale: float, offset: float) -> dict:
    from Tables._lib.tables_lib import _compress_column
    return _compress_column(table, col_name, target_bits, scale, offset)


def column_view(table: dict, col_name: str):
    """Return a zero-copy ColumnView for index access to a column.

    Reads values directly from the bit-packed container via mask+shift,
    without extracting all values to a Python list.

    Args:
        table: container table dict
        col_name: column name

    Returns:
        ColumnView instance
    """
    from Tables._lib.tables_lib import _column_view
    return _column_view(table, col_name)
