# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from _core.backend import get_xp, get_active_name

xp = get_xp()

MOMENTS: dict[int, dict] = {}


def _compute(data) -> dict:
    arr = xp.asarray(data, dtype=xp.float64)

    n = arr.size
    if n == 0:
        return {"count": 0, "sum": 0.0, "min": float("nan"), "max": float("nan"), "sum_sq": 0.0}

    sm = float(xp.sum(arr))
    mn = float(xp.min(arr))
    mx = float(xp.max(arr))
    sum_sq = float(xp.sum(arr ** 2))

    return {
        "count": n,
        "sum": sm,
        "min": mn,
        "max": mx,
        "sum_sq": sum_sq,
    }


def _compute_chunk(chunk: dict) -> dict:
    valid = chunk["valid_elements"]
    if valid == 0:
        return {"count": 0, "sum": 0.0, "min": float("nan"), "max": float("nan"), "sum_sq": 0.0}
    compression = chunk.get("compression")
    if get_active_name() == "wgpu":
        from _core.backend._wgpu import compute_moments_wgpu
        return compute_moments_wgpu(chunk["buf"], valid, compression)
    buf = chunk["buf"][:valid]
    if compression and compression.get("type") == "scaled":
        from _core.compression import unpack_scaled
        buf = unpack_scaled(buf, compression)
    return _compute(buf)


def _merge_moments(a: dict, b: dict) -> dict:
    if a["count"] == 0:
        return dict(b)
    if b["count"] == 0:
        return dict(a)
    return {
        "count": a["count"] + b["count"],
        "sum": a["sum"] + b["sum"],
        "min": a["min"] if a["min"] <= b["min"] else b["min"],
        "max": a["max"] if a["max"] >= b["max"] else b["max"],
        "sum_sq": a["sum_sq"] + b["sum_sq"],
    }


def _compute_via_dispatcher(data) -> dict:
    # NumericSeries (новый class-based API)
    if hasattr(data, 'kind') and hasattr(data, '_proxy'):
        if data.kind.is_numeric and data._proxy is not None:
            return _compute_column(data._proxy)
    if isinstance(data, dict) and data.get("_is_hidden_series"):
        return _compute_column(data)
    if isinstance(data, dict) and "_col_name" in data:
        return _compute_column(data)
    if isinstance(data, dict) and "_series_id" in data:
        from _core.kernel import get_chunks
        chunks = get_chunks(data["_series_id"])
        if chunks is None or len(chunks) == 0:
            return {"count": 0, "sum": 0.0, "min": float("nan"), "max": float("nan"), "sum_sq": 0.0}
        acc = None
        for c in chunks:
            partial = _compute_chunk(c)
            if acc is None:
                acc = partial
            else:
                acc = _merge_moments(acc, partial)
        return acc
    return _compute(data)


def _compute_column(col_proxy: dict) -> dict:
    from _core.kernel import _get_table_entry

    # Validate proxy before use
    if hasattr(col_proxy, "_validate"):
        col_proxy._validate()

    entry = _get_table_entry(col_proxy["_table_id"])
    if entry is None:
        return {"count": 0, "sum": 0.0, "min": float("nan"), "max": float("nan"), "sum_sq": 0.0}
    layout = col_proxy["_layout"]
    col_name = col_proxy["_col_name"]
    num_rows = entry["num_rows"]

    if get_active_name() == "wgpu":
        from _core.backend._wgpu import compute_column_moments_wgpu
        return compute_column_moments_wgpu(entry["rows"], entry["num_parts"],
                                           layout, col_name, num_rows)

    from _core.container import extract_column
    vals = extract_column(entry["rows"], col_name, layout)
    return _compute(vals)


def _mean(m: dict) -> float:
    if m["count"] == 0:
        return float("nan")
    return m["sum"] / m["count"]


def _var(m: dict, ddof: int = 0) -> float:
    n = m["count"]
    if n == 0:
        return float("nan")
    if n <= ddof:
        return 0.0
    mean_val = _mean(m)
    return (m["sum_sq"] / n - mean_val * mean_val) * n / (n - ddof)


def _std(m: dict, ddof: int = 0) -> float:
    return _var(m, ddof) ** 0.5


def _get_moments(data) -> dict:
    if isinstance(data, dict):
        tid = data.get("_table_id") or data.get("_series_id")
        cname = data.get("_col_name")
        cschema = data.get("_col_schema")
        gen = getattr(data, "_generation_id", 0)
        key = (tid, gen, cname, data.get("_length"), repr(cschema))
    elif hasattr(data, 'kind') and hasattr(data, '_proxy'):
        p = data._proxy
        tid = p.get("_table_id") or p.get("_series_id")
        cname = p.get("_col_name")
        cschema = p.get("_col_schema")
        gen = getattr(p, "_generation_id", 0)
        key = (tid, gen, cname, p.get("_length"), repr(cschema))
    else:
        # Plain data (list, np.array) — skip cache to avoid id-reuse bugs
        return _compute_via_dispatcher(data)
    m = MOMENTS.get(key)
    if m is not None:
        return m
    m = _compute_via_dispatcher(data)
    MOMENTS[key] = m
    return m


def _from_cache(key: int) -> dict | None:
    return MOMENTS.get(key)


def _clear_cache():
    MOMENTS.clear()
