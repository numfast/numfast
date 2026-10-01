# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""PyArrow adapter: pa.Table <-> Table, Array/ChunkedArray <-> Series.

Optional dependency: without pyarrow every entry raises a controlled
ImportError (pip install numfast[arrow]); the NumPy CPU path never needs it.
Arrow nulls -> validity sidecar; int64 narrows via check_int32_range (exact
int64 probe, never float64 round-trip); int64 logical stays int64 via
ir_series(dtype='int64') for int64-capable ops.
Timestamps/dates/null-type have no Series mapping yet -> explicit reject.
"""

import numpy as np

try:
    import pyarrow as pa
    _PA_ERR = None
except ImportError as e:
    pa = None
    _PA_ERR = e

from .._lib.series import Series
from .._lib.table import Table
from .numpy import _text_series


def _require_pa():
    if pa is None:
        raise ImportError(
            "from_arrow/to_arrow needs pyarrow: pip install numfast[arrow]. "
            "Without it the NumPy CPU path keeps working. "
            f"Import failed: {_PA_ERR}"
        ) from None


def _unpack_validity(buf, n):
    """Arrow validity bitmap → bool array (True=valid). Zero-copy where possible."""
    if buf is None:
        return np.ones(n, dtype=bool)
    raw = np.frombuffer(buf, dtype=np.uint8)
    bits = np.unpackbits(raw, bitorder='little')
    return np.ascontiguousarray(bits[:n].astype(bool))


def _zero_copy_data(combined):
    """Arrow combined Array → numpy data array with nulls zero-filled.

    Zero-copy buffer access for int32/int64/float32/float64/bool.
    Null positions are filled with 0 (semantic contract).
    """
    bufs = combined.buffers()
    n = len(combined)
    t = combined.type
    data_buf = bufs[1]
    if pa.types.is_int32(t):
        data = np.ascontiguousarray(np.frombuffer(data_buf, dtype=np.int32))
    elif pa.types.is_int64(t):
        raw = np.frombuffer(data_buf, dtype=np.int64)
        data = np.ascontiguousarray(raw.astype(np.int32))
    elif pa.types.is_float32(t):
        data = np.ascontiguousarray(np.frombuffer(data_buf, dtype=np.float32))
    elif pa.types.is_float64(t):
        data = np.ascontiguousarray(np.frombuffer(data_buf, dtype=np.float64))
    elif pa.types.is_boolean(t):
        raw = np.frombuffer(data_buf, dtype=np.uint8)
        data = np.ascontiguousarray(
            np.unpackbits(raw, bitorder='little')[:n].astype(bool))
    else:
        raise ValueError(f"_zero_copy_data: unsupported type {t}")
    # Zero-fill null positions (Arrow stores garbage there)
    valid = _unpack_validity(bufs[0], n)
    if not valid.all():
        data = data.copy()
        data[~valid] = 0
    return data


def _ingest_column(kernel, col, name):
    """pa ChunkedArray/Array -> Series (nulls -> validity, never sentinels)."""
    _require_pa()
    a = kernel.alias
    if isinstance(col, pa.Array):
        col = pa.chunked_array([col])
    if not isinstance(col, pa.ChunkedArray):
        raise ValueError(
            f"from_arrow: column '{name}' is {type(col).__name__}, not "
            "Array/ChunkedArray. Fix: pass pa.table/pa.array columns."
        )
    t = col.type
    # Combine chunks for zero-copy buffer access
    combined = col.combine_chunks()
    # --- validity (shared: numeric + text paths) ---
    valid = _unpack_validity(combined.buffers()[0], len(combined))
    has_nulls = not valid.all()
    validity = None if not has_nulls else valid
    # --- numeric zero-copy path ---
    if pa.types.is_int32(t):
        data = _zero_copy_data(combined)
        return Series(kernel, name, data, "int32", validity=validity)
    if pa.types.is_int64(t):
        # Invariant #1: the range check must see the PRE-narrow int64 array.
        # _zero_copy_data narrows int64->int32 unconditionally, and
        # check_int32_range short-circuits on int32 (Core/_lib/dtypes.py), so
        # checking after the astype is a guaranteed no-op -- read the same
        # zero-copy int64 buffer and check it first, then narrow.
        raw = np.frombuffer(combined.buffers()[1], dtype=np.int64)
        query = raw[valid] if has_nulls else raw
        a["check_int32_range"](query, f"from_arrow column '{name}'")
        data = _zero_copy_data(combined)
        return Series(kernel, name, data, "int32", validity=validity)
    for check, logical in (
            (pa.types.is_float32, "float32"),
            (pa.types.is_float64, "float64")):
        if check(t):
            data = _zero_copy_data(combined)
            return Series(kernel, name, data, logical, validity=validity)
    if pa.types.is_boolean(t):
        data = _zero_copy_data(combined)
        return Series(kernel, name, data, "bool", validity=validity)
    # --- text path: Arrow-native encode (skip to_pylist + obj array) ---
    if (pa.types.is_string(t) or pa.types.is_large_string(t)
            or pa.types.is_dictionary(t)):
        if pa.types.is_dictionary(t):
            arr_col = col.cast(pa.string()).combine_chunks()
        else:
            arr_col = combined
        # Arrow buffers are already in hand, so go straight to the Arrow C++
        # dense encode: dedup, sorted rank and body are all C++ calls, no row
        # reaches Python. Same sorted-unique envelope as dictionary_encode
        # (verified bit-identical: codes, values, body, validity, metadata).
        enc = a["dict_encode_arrow"](arr_col)
        return _text_series(kernel, name, enc, len(col), validity=validity)
    raise ValueError(
        f"from_arrow: column '{name}' type {t} has no Series mapping "
        "(timestamps/dates/null/binary/nested unsupported). "
        "Fix: cast to int32/float32/float64/bool/string first."
    )


def from_arrow(kernel, table, names=None):
    """pa.Table -> Table, pa Array/ChunkedArray -> Series."""
    _require_pa()
    if isinstance(table, (pa.Array, pa.ChunkedArray)):
        return _ingest_column(kernel, table, "v" if names is None else names)
    if not isinstance(table, pa.Table):
        raise ValueError(
            f"from_arrow: {type(table).__name__} is not pa.Table/Array. "
            "Fix: pass pa.table(...) or pa.array(...)."
        )
    want = list(table.schema.names if names is None else names)
    if len(want) != table.num_columns:
        raise ValueError(
            f"from_arrow: {table.num_columns} columns but {len(want)} names. "
            "Fix: pass one name per column."
        )
    if len(set(want)) != len(want):
        raise ValueError(
            f"from_arrow: duplicate column names {want}. "
            "Fix: rename to unique names first."
        )
    cols = {}
    for src, name in zip(table.schema.names, want):
        cols[name] = _ingest_column(kernel, table.column(src), name)
    return Table(kernel, cols)


def _to_array_pa(s):
    vals, valid = s.to_numpy(), s.validity
    mask = None if valid is None else ~valid
    if s.dtype == "text":
        return pa.array(vals.tolist(), type=pa.string(), mask=mask)
    types = {"int32": pa.int32(), "int64": pa.int64(), "float32": pa.float32(),
             "float64": pa.float64(), "bool": pa.bool_()}
    try:
        typ = types[s.dtype]
    except KeyError:
        raise ValueError(
            f"to_arrow: Series dtype {s.dtype!r} unsupported."
        ) from None
    if s.dtype == "float32":
        vals = np.ascontiguousarray(vals, dtype=np.float32)
    return pa.array(vals, type=typ, mask=mask)


def to_arrow(obj):
    """Series -> pa.Array, Table -> pa.Table (names/order preserved)."""
    _require_pa()
    if isinstance(obj, Series):
        return _to_array_pa(obj)
    if isinstance(obj, Table):
        return pa.table({n: _to_array_pa(obj.column(n)) for n in obj.names})
    raise ValueError(
        f"to_arrow: {type(obj).__name__} is not Series/Table. "
        "Fix: pass nf.from_arrow output."
    )


__all__ = ["from_arrow", "to_arrow"]
