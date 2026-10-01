# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Resident typed columns: encode-once raw -> physical int32 (generic).

One bulk pass per column at preparation time (pattern strings via
encode_pattern, plain columns via series); afterwards every query takes
int32 codes directly — no per-query string parse. Not query-specific:
no aggregation is computed here, only the typed physical representation.

TEXT keeps codes + validity + the NATIVE dictionary body (utf8_data bytes
+ offsets int32[D+1], Dictionary<T> dtype "text") by reference -- no
list[str] is held resident. The [str] values[D] live only as a D-scale
boundary/adaptor materialization (dictionary_decode, Series sidecar).
int64 (explicit encoding "dictionary" hint only) keeps codes int32[N] +
validity + the int64[D] sorted-unique LUT (dtype "int64") by reference.
Without the hint int64 columns stay series (no auto-churn).

Zero-copy: returned arrays are the driver buffers by reference; queries
must pass them into ir_series (kept by reference, viewed, never copied).
Dependencies (cpu_execute) are injected via Builder alias, never imported.
"""

import numpy as np


def _is_text_column(values, dtype):
    """Generic TEXT probe (column-agnostic): explicit dtype, <U, or all str|None."""
    if dtype == "dictionary":
        return True
    if isinstance(values, np.ndarray):
        if values.dtype.kind == "U":
            return True
        if values.dtype == object and values.size:
            try:
                nonnull = np.asarray(values.reshape(-1) != None, dtype=bool)  # noqa: E711
            except TypeError:
                return False
            if not bool(nonnull.any()):
                return False
            s = values.reshape(-1)[nonnull]
            try:
                return bool(np.asarray(s == s.astype("U"), dtype=bool).all())
            except (TypeError, ValueError):
                return False
        return False
    if isinstance(values, (list, tuple)):
        return bool(values) and all(v is None or isinstance(v, str) for v in values)
    return False


def _is_int64_column(values):
    """int64 probe (explicit-hint only): int ndarray (bool excluded) or
    all int|None list/tuple. Floats/strs/bools -> False (series path)."""
    import numbers as _numbers

    def _is_int(v):
        return isinstance(v, _numbers.Integral) and not isinstance(
            v, (bool, np.bool_))
    if isinstance(values, np.ndarray):
        return values.dtype.kind in "iu"
    if isinstance(values, (list, tuple)):
        return bool(values) and all(v is None or _is_int(v) for v in values) \
            and any(_is_int(v) for v in values)
    return False


def _is_date_column(values, dtype):
    """Generic DATE probe: explicit dtype, datetime64, or all date|datetime|None."""
    import datetime as _dt

    if dtype == "date":
        return True
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.datetime64):
            return True
        return False
    if isinstance(values, (list, tuple)):
        return bool(values) and all(v is None or isinstance(v, (_dt.date, _dt.datetime))
                                    for v in values)
    return False


def resident_prepare_impl(columns, cpu_execute, dictionary_encode=None, date_encode=None):
    """prepare({name: {values, prefix?|dtype?|encoding?}}) -> {name: resident column}.

    prefix set -> pattern column (prefix+int strings -> int32 codes +
    validity + pattern sidecar).     TEXT (dtype 'dictionary', <U, or all str|None, prefix None) -> dictionary
    column (sorted-unique codes int32 + validity + NATIVE body sidecar
    utf8_data/offsets, dtype "text", deterministic, NULL explicit). int64
    (encoding 'dictionary' hint + int column, prefix None) -> dictionary
    column (sorted-unique codes int32 + validity + int64[D] LUT sidecar,
    dtype "int64", deterministic, NULL explicit). encoding 'auto'/'series'
    or absent keeps the legacy behavior (TEXT auto-detect, else series).
    DATE (dtype 'date', datetime64, or all date|datetime|None) -> date column
    (int32 epoch seconds + validity, encoding date-epoch-s-v1). Otherwise
    typed series as-is (int32 codes view, no copy when dtype matches).
    """
    nodes = []
    kinds = {}
    resident = {}
    for name, col in columns.items():
        prefix = col.get("prefix")
        if prefix:
            if not isinstance(prefix, str) or not prefix:
                raise ValueError(
                    f"resident_prepare: prefix for '{name}' must be a non-empty str. "
                    "Fix: pass prefix like 'id'. See specs/delta-4-pattern-strings.md"
                )
            nodes.append({
                "op": "encode_pattern", "inputs": [],
                "params": {"values": col["values"], "prefix": prefix}, "out": name,
            })
            kinds[name] = "pattern"
        elif _is_text_column(col.get("values"), col.get("dtype")) \
                or (col.get("encoding") == "dictionary"
                    and _is_int64_column(col.get("values"))):
            if dictionary_encode is None:
                raise ValueError(
                    f"resident_prepare: TEXT column '{name}' needs dictionary_encode. "
                    "Fix: build kernel from full.toml (Schema depends Dictionary). "
                    "See specs/05-storage-encoding.md"
                )
            enc = dictionary_encode(col["values"], col.get("validity"))
            kinds[name] = "dictionary"
            # Bypass cpu_execute: codes are already the resident int32 buffer
            # (kept by reference, never copied); Driver consumes them via
            # ir_series zero-copy like every other resident column.
            # The dictionary body (native utf8_data + offsets, dtype "text",
            # or int64[D] LUT, dtype "int64") is kept by reference too; no
            # values[D] list stays resident.
            resident[name] = {
                "codes": enc["codes"],
                "validity": enc["validity"],
                "dictionary": enc["dictionary"],
                "dtype": enc.get("dtype", "text"),
                "kind": "dictionary",
                "physical": "int32",
                "n": int(enc["codes"].size),
                "encoding": enc["metadata"]["encoding"],
            }
        elif _is_date_column(col.get("values"), col.get("dtype")):
            if date_encode is None:
                raise ValueError(
                    f"resident_prepare: DATE column '{name}' needs date_encode. "
                    "Fix: build kernel from full.toml (Schema provides date_encode). "
                    "See specs/05-storage-encoding.md"
                )
            enc = date_encode(col["values"], col.get("validity"))
            kinds[name] = "date"
            resident[name] = {
                "codes": enc["codes"],
                "validity": enc["validity"],
                "kind": "date",
                "physical": "int32",
                "n": int(enc["codes"].size),
                "encoding": enc["metadata"]["encoding"],
            }
        else:
            _dt = col.get("dtype")
            if _dt is None:
                _v = col.get("values")
                if isinstance(_v, np.ndarray) and _v.dtype.kind == "f":
                    _dt = str(_v.dtype)  # float32 stays float32, float64 stays float64
                else:
                    _dt = "int32"
            nodes.append({
                "op": "series", "inputs": [],
                "params": {"values": col["values"],
                           "dtype": _dt}, "out": name,
            })
            kinds[name] = "series"
    bufs = cpu_execute(nodes)
    for name in columns:
        if name in resident:
            continue  # dictionary path: buffer already resident, no copy
        codes = bufs[name]
        if not isinstance(codes, np.ndarray):
            codes = np.asarray(codes)
        resident[name] = {
            "codes": codes,
            "validity": bufs.get(name + "#validity"),
            "pattern": bufs.get(name + "#pattern"),
            "kind": kinds[name],
            "physical": str(codes.dtype),
            "n": int(codes.size),
        }
    return resident
