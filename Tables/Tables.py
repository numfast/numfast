# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from Tables._lib.tables_lib import _to_dataframe, _merge, _aggregate
from Tables._lib.tables_lib import _container_table_create, _get_column, _compress_column

def container_table(schema: list[dict], data: dict) -> dict:
    return _container_table_create(schema, data)

def get_column(table: dict, name: str) -> dict:
    return _get_column(table, name)

def compress_column(table: dict, col_name: str,
                    target_bits: int, scale: float, offset: float) -> dict:
    return _compress_column(table, col_name, target_bits, scale, offset)

def table_main(data: dict) -> dict:
    return _to_dataframe(data)

def merge_tables(left: dict, right: dict, on: str | None = None) -> dict:
    left_df = _to_dataframe(left)
    right_df = _to_dataframe(right)
    return _merge(left_df, right_df, on)

def aggregate_table(data: dict, group_by: str, agg: str = "sum") -> dict:
    df = _to_dataframe(data)
    return _aggregate(df, group_by, agg)
