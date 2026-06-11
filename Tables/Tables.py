from Tables._lib.tables_lib import _to_dataframe, _merge, _aggregate

def table_main(data: dict) -> dict:
    return _to_dataframe(data)

def merge_tables(left: dict, right: dict, on: str | None = None) -> dict:
    left_df = _to_dataframe(left)
    right_df = _to_dataframe(right)
    return _merge(left_df, right_df, on)

def aggregate_table(data: dict, group_by: str, agg: str = "sum") -> dict:
    df = _to_dataframe(data)
    return _aggregate(df, group_by, agg)
