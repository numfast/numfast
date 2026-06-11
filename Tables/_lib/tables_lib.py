try:
    import cupy as xp
except ImportError:
    import numpy as xp


_AGG_FUNCS = {
    "sum": xp.sum,
    "mean": xp.mean,
    "min": xp.min,
    "max": xp.max,
    "std": xp.std,
    "var": xp.var,
}


def _to_dataframe(data: dict) -> dict[str, xp.ndarray]:
    return {key: xp.array(values, dtype=xp.float64) for key, values in data.items()}


def _merge(left: dict, right: dict, on: str | None = None) -> dict:
    left_keys = left[on]
    right_keys = right[on]
    matches = left_keys[:, None] == right_keys[None, :]
    left_idx, right_idx = xp.where(matches)

    result = {}
    for key, arr in left.items():
        result[key] = arr[left_idx]
    for key, arr in right.items():
        if key != on:
            result[key] = arr[right_idx]
    return result


def _aggregate(df: dict, group_by: str, agg: str = "sum") -> dict:
    groups = xp.unique(df[group_by])
    agg_func = _AGG_FUNCS.get(agg, xp.sum)

    other_cols = [k for k in df if k != group_by]
    result = {group_by: groups}
    for col in other_cols:
        col_vals = xp.empty(len(groups), dtype=df[col].dtype)
        for i, g in enumerate(groups):
            mask = df[group_by] == g
            col_vals[i] = agg_func(df[col][mask])
        result[col] = col_vals

    return result
