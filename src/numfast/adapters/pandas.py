# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Pandas adapter: DataFrame <-> Table, Series <-> Series (boundary only).

Nullable (Int64/boolean/string/Float64) NA -> validity sidecar + zero fill;
round-trip restores NA via nullable ExtensionDtypes. Column names and order
preserved. No execution here: translation + Series/Table construction only.
"""

import numpy as np

try:
    import pandas as pd
    _PD_ERR = None
except ImportError as e:
    pd = None
    _PD_ERR = e

from .._lib.series import Series
from .._lib.table import Table
from .numpy import _ingest_1d, _ingest_text, _text_series


def _require_pd():
    if pd is None:
        raise ImportError(
            "from_pandas/to_pandas needs pandas: pip install numfast[pandas]. "
            f"Import failed: {_PD_ERR}"
        ) from None


def _normalize_name(col, seen):
    name = str(col)
    if name in seen:
        raise ValueError(
            f"from_pandas: duplicate column name '{name}'. "
            "Fix: df.columns = [...] unique names first."
        )
    return name


def _ingest_ea(kernel, col, name):
    """Nullable ExtensionArray column -> (values, logical, validity)."""
    a = kernel.alias
    mask = np.ascontiguousarray(col.isna().to_numpy(dtype=bool))
    dtype = col.dtype
    if isinstance(dtype, pd.StringDtype) or dtype == object:
        vals = [None if m else v for m, v in zip(mask.tolist(), col.tolist())]
        enc = a["dictionary_encode"](vals)
        return _text_series(
            kernel, name, enc, len(col),
            validity=np.ascontiguousarray(~mask) if bool(mask.any()) else None)
    kind = dtype.kind
    if kind in "iu":
        data = np.asarray(col.to_numpy(dtype=np.float64, na_value=0))
        a["check_int32_range"](data[~mask] if bool((~mask).any()) else data,
                               f"from_pandas column '{name}'")
        data = np.ascontiguousarray(data).astype(np.int32)
        return Series(kernel, name, data, "int32",
                      validity=None if not bool(mask.any()) else ~mask)
    if kind == "f":
        data = np.ascontiguousarray(col.to_numpy(
            dtype=np.float32 if dtype == np.dtype(np.float32) else np.float64,
            na_value=np.nan))
        nan = np.isnan(data)
        eff = mask | nan
        data = data.copy()
        data[eff] = 0.0
        logical = "float32" if data.dtype == np.dtype(np.float32) else "float64"
        return Series(kernel, name, data, logical,
                      validity=None if not bool(eff.any()) else ~eff)
    if kind == "b":
        data = np.ascontiguousarray(col.to_numpy(dtype=bool, na_value=False))
        return Series(kernel, name, data, "bool",
                      validity=None if not bool(mask.any()) else ~mask)
    raise ValueError(
        f"from_pandas: column '{name}' dtype {dtype} unsupported. "
        "Fix: convert to numeric/bool/string first."
    )


def _ingest_plain(kernel, col, name):
    arr = col.to_numpy()
    if arr.dtype == object or str(arr.dtype.kind) == "U":
        mask = col.isna().to_numpy(dtype=bool)
        if bool(np.asarray(mask).any()):
            vals = [None if m else v for m, v in zip(mask.tolist(), col.tolist())]
            a = kernel.alias
            enc = a["dictionary_encode"](vals)
            return _text_series(
                kernel, name, enc, len(col),
                validity=np.ascontiguousarray(~np.asarray(mask, dtype=bool)))
        return _ingest_text(kernel, np.asanyarray(col.tolist()), name)
    data, logical, det, shared = _ingest_1d(kernel, arr, name)
    s = Series(kernel, name, data, logical, validity=det)
    s._shared = shared
    return s


def from_pandas(kernel, df, names=None):
    """DataFrame -> Table, Series -> Series. `names` overrides columns."""
    _require_pd()
    single = isinstance(df, pd.Series)
    if single:
        df = df.to_frame()
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"from_pandas: {type(df).__name__} is not DataFrame/Series. "
            "Fix: pass df or df['col']."
        )
    cols = list(df.columns if names is None else names)
    if names is not None and len(names) != len(df.columns):
        raise ValueError(
            f"from_pandas: {len(df.columns)} columns but {len(names)} names. "
            "Fix: pass one name per column."
        )
    out, seen = {}, set()
    for src, want in zip(df.columns, cols):
        name = _normalize_name(want, seen)
        seen.add(name)
        col = df[src]
        if isinstance(col.dtype, pd.api.extensions.ExtensionDtype):
            out[name] = _ingest_ea(kernel, col, name)
        else:
            out[name] = _ingest_plain(kernel, col, name)
    if single:
        return next(iter(out.values()))
    return Table(kernel, out)


def _to_series_pd(s):
    vals, name, valid = s.to_numpy(), s.name, s.validity
    if s.dtype == "text":
        arr = np.asarray([None if v is None else v for v in vals.tolist()],
                         dtype=object)
        out = pd.Series(pd.array(arr, dtype="string"), name=name)
        return out
    if valid is None:
        return pd.Series(vals, name=name)
    inv = ~valid
    if s.dtype == "int32":
        arr = pd.array(vals, dtype="int32").astype("Int32")
        arr[inv] = pd.NA
        return pd.Series(arr, name=name)
    if s.dtype in ("float32", "float64"):
        arr = pd.array(vals, dtype=s.dtype).astype(
            "Float32" if s.dtype == "float32" else "Float64")
        arr[inv] = pd.NA
        return pd.Series(arr, name=name)
    if s.dtype == "bool":
        arr = pd.array(vals, dtype="bool").astype("boolean")
        arr[inv] = pd.NA
        return pd.Series(arr, name=name)
    raise ValueError(f"to_pandas: Series dtype {s.dtype!r} unsupported.")


def to_pandas(obj):
    """Series -> pd.Series, Table -> pd.DataFrame (names/order preserved)."""
    _require_pd()
    if isinstance(obj, Series):
        return _to_series_pd(obj)
    if isinstance(obj, Table):
        return pd.DataFrame({n: _to_series_pd(obj.column(n)) for n in obj.names})
    raise ValueError(
        f"to_pandas: {type(obj).__name__} is not Series/Table. "
        "Fix: pass nf.from_pandas output."
    )


__all__ = ["from_pandas", "to_pandas"]
