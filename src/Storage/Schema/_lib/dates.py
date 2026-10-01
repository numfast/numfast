# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""DATE as int32 epoch seconds (single physical variant, fixed as encoding).

date-epoch-s-v1: DATE/DATETIME -> int32 seconds since 1970-01-01T00:00:00Z.
date inputs normalize to midnight; naive datetimes are UTC (documented, no
tz database, no calendar tables). Core sees plain int32 scalars: existing
compare/filter/groupby/reduce work unchanged, zero-copy.

Bounds: int32 seconds cover 1901-12-13..2038-01-19; outside -> explicit
error (never silent wrap). NULL: None -> invalid (codes placeholder 0 +
validity False). Decode is display-only boundary restore.
"""

import datetime as _dt

import numpy as np

ENCODING = "date-epoch-s-v1"
_INT32_MIN = -(2 ** 31)
_INT32_MAX = 2 ** 31 - 1


def _err(format_error, what, fix, doc="specs/05-storage-encoding.md"):
    if format_error is not None:
        raise format_error(what, fix=fix, doc=doc)
    raise ValueError(f"{what} Fix: {fix}. See {doc}")


def _to_epoch_s(values, format_error):
    """Column -> (epoch int64[N] seconds, nonnull bool[N]). Bulk, C-speed."""
    if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.datetime64):
        arr = values.reshape(-1)
        nonnull = ~np.isnat(arr)
        ep = np.zeros(arr.size, dtype=np.int64)
        ep[nonnull] = arr[nonnull].astype("datetime64[s]").astype(np.int64)
        return ep, nonnull
    if not isinstance(values, (np.ndarray, list, tuple)):
        try:
            values = list(values)
        except TypeError:
            _err(format_error, "date_encode needs DATE/DATETIME column, got "
                 f"{type(values).__name__}.", "pass date/datetime/datetime64/None rows")
    arr = np.asarray(values, dtype=object).reshape(-1)
    n = arr.size
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    try:
        nonnull = np.asarray(arr != None, dtype=bool)  # noqa: E711 -- C loop
    except TypeError as e:
        _err(format_error, f"date_encode needs DATE/DATETIME column ({e}).",
             "pass date/datetime/datetime64/None rows")
    ep = np.zeros(n, dtype=np.int64)
    if bool(nonnull.any()):
        s = arr[nonnull]
        flat = s.tolist()
        for v in flat:
            if not isinstance(v, (_dt.date, _dt.datetime, np.datetime64)):
                _err(format_error, "date_encode needs DATE/DATETIME column, got "
                     f"{type(v).__name__} value {v!r}.",
                     "pass date/datetime/datetime64/None rows")
            if isinstance(v, _dt.datetime) and v.tzinfo is not None:
                _err(format_error, f"date_encode rejects tz-aware {v!r} (no tz database).",
                     "pass naive datetimes (UTC) or None")
        try:
            conv = np.asarray(flat, dtype="datetime64[s]")
        except (TypeError, ValueError) as e:
            _err(format_error, f"date_encode cannot parse datetime values ({e}).",
                 "pass date/datetime/datetime64/None rows")
        if bool(np.isnat(conv).any()):
            _err(format_error, "date_encode got NaT datetime.",
                 "pass concrete date/datetime or None")
        ep[nonnull] = conv.astype(np.int64)
    return ep, nonnull


def date_encode_impl(values, validity=None, format_error=None):
    """encode([date|datetime|datetime64|None]*N) -> {codes int32, validity, metadata}."""
    ep, nonnull = _to_epoch_s(values, format_error)
    n = ep.size
    if validity is not None:
        vb = np.asarray(validity, dtype=bool).reshape(-1)
        if vb.size != n:
            _err(format_error, f"date_encode validity size {vb.size} != values {n}.",
                 "pass validity matching values length",
                 doc="specs/delta-3-null-contract.md")
    else:
        vb = None
    if n and (bool((ep[nonnull] < _INT32_MIN).any()) or bool((ep[nonnull] > _INT32_MAX).any())):
        bad = int(ep[nonnull][np.nonzero((ep[nonnull] < _INT32_MIN) |
                                         (ep[nonnull] > _INT32_MAX))[0][0]])
        _err(format_error, f"date_encode epoch seconds {bad} exceed int32 "
             "(range 1901-12-13..2038-01-19).", "use a wider physical dtype path")
    codes = np.zeros(n, dtype=np.int32)
    codes[nonnull] = ep[nonnull].astype(np.int32)
    eff = (nonnull & vb) if vb is not None else nonnull
    if vb is not None:
        codes[~vb] = 0
    eff = np.ascontiguousarray(eff)
    return {"codes": np.ascontiguousarray(codes),
            "validity": None if bool(np.all(eff)) else eff,
            "metadata": {"encoding": ENCODING, "n": int(n),
                         "nulls": int(n - np.count_nonzero(eff))}}


def date_decode_impl(codes, validity=None, format_error=None):
    """Display-only boundary restore: int32 epoch seconds -> [datetime|None]."""
    c = np.asarray(codes, dtype=np.int64).reshape(-1)
    if validity is not None:
        v = np.asarray(validity, dtype=bool).reshape(-1)
        if v.size != c.size:
            _err(format_error, f"date_decode validity size {v.size} != codes {c.size}.",
                 "pass validity matching codes length",
                 doc="specs/delta-3-null-contract.md")
    else:
        v = None
    out = []
    for i, k in enumerate(c.tolist()):
        if v is not None and not bool(v[i]):
            out.append(None)
        else:
            out.append(_dt.datetime(1970, 1, 1) + _dt.timedelta(seconds=int(k)))
    return out
