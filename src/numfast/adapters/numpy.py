# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""NumPy adapter: 1D -> Series, 2D -> Table (and back).

Dtype contract (Core single source): int32 logical via from_numpy (small
ints narrow to int32 only after check_int32_range, OverflowError, never
silent wrap); int64 input stays int64 logical (generic int64 path, zero-copy
if contiguous; uint64 range-checked to int64, overflow raises); float32
compute; float64 explicit; bool stays bool. int64 raw via from_numpy / float16 / complex /
datetime64 are rejected with what+fix (use Schema scaled-int / date paths).

Zero-copy: contiguous int32/float32/float64/bool arrays without missing
values are kept BY REFERENCE (no copy). Copy happens (explicitly) when:
non-contiguous input, int64 narrowing, or missing values (NaN/None/NA ->
validity sidecar + zero fill). to_numpy() copies (caller owns the result).
"""

import numpy as np

from .._lib.series import Series
from .._lib.table import Table

_LOGICAL_OF_KIND = {"i": "int32", "u": "int32", "f": None, "b": "bool"}


def _logical_for(arr, kernel):
    kind = arr.dtype.kind
    if kind == "f":
        if arr.dtype == np.dtype(np.float32):
            return "float32"
        if arr.dtype == np.dtype(np.float64):
            return "float64"
        raise ValueError(
            f"from_numpy: float width {arr.dtype} unsupported: use float32/float64. "
            "Fix: arr.astype(np.float32)."
        )
    if kind in "iu":
        if arr.dtype == np.dtype(np.int64):
            return "int64"
        return "int32"
    if kind == "b":
        return "bool"
    raise ValueError(
        f"from_numpy: dtype {arr.dtype} (kind '{kind}') has no Series mapping. "
        "Fix: pass int32/float32/float64/bool/str."
    )


def _is_text_array(arr):
    if arr.dtype.kind == "U":
        return True
    if arr.dtype == object and arr.size:
        try:
            sample = arr.reshape(-1)
            nonnull = sample != None  # noqa: E711 -- vectorized C loop
            if not bool(np.asarray(nonnull).any()):
                return True  # all-None: text column, every row invalid
            probe = sample[np.asarray(nonnull, dtype=bool)][:8]
            return all(isinstance(v, str) for v in probe.tolist())
        except (TypeError, ValueError):
            return False
    return False


def _text_series(kernel, name, enc, n, validity=None):
    """Text Series from a dictionary_encode result.

    All-null edge: the engine emits an empty dictionary (D=0) with zero
    codes; decode range-checks every code, so an empty sidecar would raise.
    Every row is invalid there, therefore any placeholder decodes to None:
    substitute a single empty string (adapter-side, engine untouched).
    """
    values = list(enc["values"])
    if n > 0 and not values:
        values = [""]
    s = Series(kernel, name, enc["codes"], "text", validity=validity,
               sidecar={"values": values})
    s._shared = False
    return s


def _ingest_text(kernel, arr, name):
    """String column -> text Series via Dictionary (Schema/Storage layer)."""
    a = kernel.alias
    flat = (arr.reshape(-1) if arr.dtype.kind == "U"
            else np.asarray(arr, dtype=object).reshape(-1))
    # Hand the flat carrier to the Arrow C++ encode as is. The old route did
    # flat.tolist() (N Python str, or an extra N-element list comprehension for
    # the object lane) purely to give dictionary_encode something to walk in
    # Python -- the encode never needed a Python row. Measured at N=2,000,000:
    # 1745/1522 ms through tolist(), ~350 ms straight into the C++ lane.
    enc = a["dict_encode_arrow"](flat)
    n = int(flat.size)
    valid = enc["validity"]
    return _text_series(
        kernel, name, enc, n,
        validity=None if valid is None else np.ascontiguousarray(
            np.asarray(valid, dtype=bool)))


def _ingest_1d(kernel, arr, name):
    """(values, logical, validity, shared) for one flat column."""
    a = kernel.alias
    arr = np.asanyarray(arr)
    if arr.ndim != 1:
        raise ValueError(
            f"from_numpy: column '{name}' must be rank-1, got shape {arr.shape}. "
            "Fix: pass a flat array (2D -> table path)."
        )
    if isinstance(arr, np.ma.MaskedArray):
        data, validity = np.asarray(arr.data), ~np.asarray(arr.mask)
        shared = False
    else:
        data, validity, shared = arr, None, True
    kind = data.dtype.kind
    if kind == "f":
        nan = np.isnan(data)
        if bool(nan.any()):
            validity = (~nan) if validity is None else (validity & ~nan)
            data = data.copy()
            data[nan] = 0.0
            shared = False
    elif kind in "iu" and data.dtype != np.dtype(np.int32):
        if data.dtype == np.dtype(np.int64):
            pass  # int64 logical path: keep width, contiguity handled below
        elif data.dtype == np.dtype(np.uint64):
            hi = int(data.max()) if data.size else 0
            if hi > 2 ** 63 - 1:
                raise OverflowError(
                    f"from_numpy column '{name}': uint64->int64 out of range "
                    f"(max {hi} exceeds int64 max {2 ** 63 - 1}). Fix: pass values "
                    "within int64 range."
                )
            data = np.ascontiguousarray(data, dtype=np.int64)
            shared = False
        else:
            a["check_int32_range"](data, f"from_numpy column '{name}'")
            data = np.ascontiguousarray(data, dtype=np.int32)
            shared = False  # narrowing cast always copies
    elif kind == "b" and data.dtype != np.dtype(bool):
        data = np.ascontiguousarray(data, dtype=bool)
        shared = False
    if not data.flags["C_CONTIGUOUS"]:
        data = np.ascontiguousarray(data)
        shared = False
    logical = _logical_for(data, kernel)
    if validity is not None:
        validity = np.ascontiguousarray(np.asarray(validity, dtype=bool))
    return data, logical, validity, shared


def from_numpy(kernel, arr, name="v", names=None, validity=None):
    """1D -> Series, 2D (n, m) -> Table. `validity` overrides auto detection."""
    arr = np.asanyarray(arr)
    if arr.ndim == 1:
        if _is_text_array(arr) and validity is None:
            return _ingest_text(kernel, arr, name)
        data, logical, det, shared = _ingest_1d(kernel, arr, name)
        if validity is not None:
            det = np.ascontiguousarray(np.asarray(validity, dtype=bool))
            shared = False
        s = Series(kernel, name, data, logical, validity=det)
        s._shared = shared
        return s
    if arr.ndim == 2:
        n, m = arr.shape
        names = list(names) if names is not None else [f"col_{i}" for i in range(m)]
        if len(names) != m:
            raise ValueError(
                f"from_numpy: {m} columns but {len(names)} names. "
                "Fix: pass one name per column."
            )
        cols = {}
        for i, nm in enumerate(names):
            col_arr = arr[:, i]
            if _is_text_array(np.asanyarray(col_arr)) and validity is None:
                cols[nm] = _ingest_text(kernel, col_arr, nm)
                continue
            data, logical, det, shared = _ingest_1d(kernel, col_arr, nm)
            if validity is not None:
                det = np.ascontiguousarray(np.asarray(validity, dtype=bool))
            col = Series(kernel, nm, data, logical, validity=det)
            col._shared = False  # column slice always copies
            cols[nm] = col
        _ = n
        return Table(kernel, cols)
    raise ValueError(
        f"from_numpy: rank {arr.ndim} unsupported (only 1D Series / 2D Table). "
        "Fix: reshape or select a column."
    )


def to_numpy(obj):
    if isinstance(obj, Series):
        return obj.to_numpy()
    if isinstance(obj, Table):
        return obj.to_numpy()
    raise ValueError(
        f"to_numpy: {type(obj).__name__} is not Series/Table. "
        "Fix: pass nf.from_numpy output."
    )


__all__ = ["from_numpy", "to_numpy"]
