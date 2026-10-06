# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""NFS-staging persist/load: table dict -> .npz file -> table dict (spec 09).

STAGING format only (npz-staging-v0, observable in metadata). Canonical
dzst/block-codec comes later and is NOT implemented here.
Table = {column: {"values": list, "dtype": logical-name}}.
numpy dtypes derived from Core canonical names (no private imports).
"""

import json

import numpy as np

_FORMAT = "npz-staging-v0"
_META_KEY = "__meta__"


def _fallback_err(what, fix="", doc=""):
    msg = str(what)
    if fix:
        msg += f" Fix: {fix}."
    if doc:
        msg += f" See {doc}"
    return ValueError(msg)


def persist_impl(table, path, canonical_dtype, format_error=None, check_int32_range=None):
    """persist(table, path) -> metadata {path, format, n, columns[{name,dtype,n}]}.

    Invariant #1: int32 columns are range-checked before the narrowing
    materialize (OverflowError, never a silent wrap). The checker arrives via
    kernel.alias (Core, injected by the entry); None keeps the legacy coerce
    path for standalone use only. int64 columns stay int64 (no check) for
    int64-capable ops; execution codes stay int32.
    """
    err = format_error or _fallback_err
    if not isinstance(table, dict) or not table:
        raise err(
            f"persist: table must be non-empty dict, got {type(table).__name__}",
            fix="pass {column: {'values': [...], 'dtype': 'int32'}}",
        )
    payload = {}
    columns = []
    n = None
    for name, col in table.items():
        try:
            values, dtype = col["values"], col["dtype"]
        except (TypeError, KeyError):
            raise err(
                f"persist: column '{name}' must be {{'values', 'dtype'}}",
                fix="pass {'values': [...], 'dtype': 'int32'}",
            ) from None
        info = canonical_dtype(dtype)  # int64 allowed (stays int64); scaled-int via Schema
        if isinstance(info, dict) and info.get("kind") == "enum":
            raise err(
                f"persist: packed enum '{dtype}' not in npz-staging-v0 "
                "(bool masks persist as bool columns)",
                fix="unpack to uint8 codes or persist as bool",
            )
        if info["logical"] == "int32" and check_int32_range is not None:
            check_int32_range(values, f"persist column '{name}'")
        arr = np.asarray(list(values), dtype=np.dtype(info["logical"]))
        if n is None:
            n = arr.size
        elif arr.size != n:
            raise err(
                f"persist: column '{name}' length {arr.size} != {n}",
                fix="pass equal-length columns",
            )
        payload[name] = arr
        columns.append({"name": name, "dtype": info["logical"], "n": int(arr.size)})
    payload[_META_KEY] = np.asarray(json.dumps({"format": _FORMAT, "columns": columns}))
    np.savez(str(path), **payload)
    return {"path": str(path), "format": _FORMAT, "n": int(n or 0), "columns": columns}


def load_impl(path, canonical_dtype, format_error=None):
    """load(path) -> table {column: {'values', 'dtype'}}; missing file -> ValueError."""
    err = format_error or _fallback_err
    try:
        with np.load(str(path), allow_pickle=False) as z:
            names = [k for k in z.files if k != _META_KEY]
            table = {}
            for name in names:
                arr = z[name]
                table[name] = {"values": arr.tolist(), "dtype": str(arr.dtype)}
    except FileNotFoundError:
        raise err(
            f"load: file not found '{path}'",
            fix="persist the table first via persist_table(table, path)",
        ) from None
    for col in table.values():
        canonical_dtype(col["dtype"])
    return table
