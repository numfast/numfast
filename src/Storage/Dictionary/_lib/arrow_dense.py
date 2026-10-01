# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""Generic Arrow C++ dense-dictionary encode for a flat TEXT column.

Contract
--------
dict_encode_arrow(values, validity=None, order="sorted")

  values : rank-1 flat column of str | None. Accepted carriers: python
           list, tuple, object ndarray, numpy <U ndarray, pyarrow
           Array, pyarrow ChunkedArray. Every other carrier (dict
           envelope, 2-D ndarray, non-string scalars, bytes, numbers) is
           an explicit error -- never a silent coercion.
  order  : "sorted"      -- code == lexicographic rank of the value
                             (the dictionary_encode contract: code order
                             == value order, so min(codes) == min(value),
                             ORDER BY over codes needs no rank LUT).
             "first_seen" -- code == rank of first appearance. Same value
                             set, same D, different numbering. For callers
                             that use codes ONLY as group keys and never
                             compare codes as values; NOT interchangeable
                             with the sorted contract.
  result : the dictionary_encode envelope --
           {"codes": int32[N], "values": [str]*D,
            "dictionary": DictionaryBody (native UTF-8 body),
            "dtype": "text", "validity": bool[N] | None,
            "metadata": {encoding, d, n, sorted, nulls}}

NULL contract (DELTA-3, identical to dictionary_encode): a None row is
invalid, gets the code-0 placeholder, and never appears as a dictionary
value. `validity` is None when every row is valid. The value SET is built
from the column itself; the `validity` sidecar only masks codes, exactly
like dictionary_encode.

Why this is a separate op rather than a change to dictionary_encode
--------------------------------------------------------------------
The encode cost of a high-cardinality TEXT column was dominated by
per-row Python work: one Python-level dedup pass over every row plus a
``sorted()`` of D strings, a Python dict of D entries and a Python list
remap of D entries. Here every one of those is a single C++ call over D
or N:

  * one bulk transcode of the carrier into a pyarrow string Array;
  * ``pc.dictionary_encode`` -- the dedup, entirely in C++;
  * ``pc.sort_indices`` on the D-entry dictionary -- the sorted rank, in
    C++, instead of ``sorted()`` + dict + list comprehension;
  * one gather of the permutation over the N indices -- the codes;
  * ``take`` on the D-entry dictionary -- the sorted body, in C++;
  * the native UTF-8 body read straight out of the Arrow buffers.

No row ever reaches Python. That is the whole difference, and it is a
property of the column (D), not of any query.

Deliberately NOT here
---------------------
  * no regex / substring matching (Arrow has no vectorized Python-regex
    equivalent, and the two regex dialects disagree on ``$`` -- Python
    matches before a trailing newline, RE2 means end-of-text);
  * no int64 dictionaries (numeric dictionaries are not text);
  * no change to dictionary_encode's own behaviour -- it keeps its
    carrier set, its int64 lane and its error messages.
"""

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    _HAS_PA = True
except ImportError:  # pragma: no cover -- Arrow-less install
    pa = pc = None
    _HAS_PA = False

from _lib.dictionary import BODY_FORMAT, ENCODING, TEXT_DTYPE, DictionaryBody, _body_build

ORDERS = ("sorted", "first_seen")


def _err(format_error, what, fix, doc="specs/05-storage-encoding.md"):
    if format_error is not None:
        raise format_error(what, fix=fix, doc=doc)
    raise ValueError(f"{what} Fix: {fix}. See {doc}")


def _reject(what, fix, format_error=None):
    _err(format_error, what, fix)


def _carrier(values, op, format_error=None):
    """Any accepted flat TEXT carrier -> one pyarrow string Array, or raise.

    The Arrow type is INFERRED, not forced: inference is both cheaper than
    a forced conversion and the type check itself. bytes / mixed
    bytes+str come back as ``binary`` and are rejected here instead of
    being silently decoded; numbers fail inference outright.
    """
    if not _HAS_PA:
        _reject(f"{op}: pyarrow is required for the bulk dense-dictionary "
                "encode (pip install pyarrow).",
                "install pyarrow, or use dictionary_encode on an Arrow-less "
                "install", format_error)
    if isinstance(values, dict):
        _reject(f"{op}: got a dict carrier (an encode envelope or a body "
                "dict), expected a flat string column.",
                "pass the flat str column, or pass the dictionary to a "
                "dictionary-domain op", format_error)
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            _reject(f"{op} needs a rank-1 column, got shape {values.shape}.",
                    "flatten the column before encoding it", format_error)
        if values.dtype.kind not in "UO":
            _reject(f"{op} is a TEXT encode, got a {values.dtype} ndarray.",
                    "pass a <U / object string column", format_error)
        seq = values
    elif isinstance(values, pa.ChunkedArray):
        seq = values
    elif isinstance(values, pa.Array):
        seq = values
    elif isinstance(values, (list, tuple)):
        seq = list(values)
    elif isinstance(values, (str, bytes)):
        _reject(f"{op}: got a single {type(values).__name__}, expected a "
                "flat string column.", "pass a column, not a scalar",
                format_error)
    else:
        try:
            seq = list(values)
        except TypeError:
            _reject(f"{op}: needs a string column, got "
                    f"{type(values).__name__}.", "pass a list[str|None] column",
                    format_error)
    try:
        arr = seq.combine_chunks() if isinstance(seq, pa.ChunkedArray) else seq
        arr = pa.array(arr) if not isinstance(arr, pa.Array) else arr
    except Exception as e:  # noqa: BLE001 -- every Arrow failure is a type error here
        _reject(f"{op} needs a string column ({e}).",
                "pass str or None per row, never raw numbers", format_error)
    if pa.types.is_null(arr.type):
        arr = pa.nulls(len(arr), pa.string())
    elif not pa.types.is_string(arr.type):
        _reject(f"{op} is a TEXT encode, got an Arrow {arr.type} column.",
                "pass a column of str|None", format_error)
    return arr


def _validity(arr, validity, n, op, format_error=None):
    """Arrow null mask merged with an optional validity sidecar -> bool[N]."""
    valid = np.ascontiguousarray(
        ~np.asarray(arr.is_null().to_numpy(zero_copy_only=False), dtype=bool))
    if validity is None:
        return valid
    vb = np.asarray(validity, dtype=bool).reshape(-1)
    if vb.size != n:
        _err(format_error,
             f"{op} validity size {vb.size} != values {n}.",
             "pass validity matching values length",
             doc="specs/delta-3-null-contract.md")
    return np.ascontiguousarray(valid & vb)


def _body(sorted_array):
    """pyarrow string Array -> DictionaryBody, straight from the buffers."""
    bufs = sorted_array.buffers()
    if bufs[1] is None or bufs[2] is None:
        return None
    try:
        off = np.frombuffer(bufs[1], dtype=np.int32)
        if int(off.size) != len(sorted_array) + 1 or int(off[0]) != 0:
            return None
        off = np.ascontiguousarray(off)
        data = bytes(memoryview(bufs[2])[:int(off[-1])])
    except (TypeError, ValueError):
        return None
    if len(data) != int(off[-1]):
        return None
    return DictionaryBody(data, off)


def dict_encode_arrow_impl(values, validity=None, order="sorted", format_error=None):
    """dict_encode_arrow_impl(values, validity, order, format_error) -> envelope.

    See the module docstring for the contract. Every failure mode is an
    explicit error; the Arrow C++ dense dictionary is never a partial
    answer.
    """
    op = "dict_encode_arrow"
    if order not in ORDERS:
        _err(format_error, f"{op} order must be one of {ORDERS}, got {order!r}.",
             "pass order='sorted' (code == value rank) or order='first_seen'")
    arr = _carrier(values, op, format_error)
    n = len(arr)
    valid = _validity(arr, validity, n, op, format_error)
    d = pc.dictionary_encode(arr)
    dd = len(d.dictionary)
    if dd > 2 ** 31 - 1:
        _err(format_error,
             f"{op} cardinality {dd} exceeds int32.", "use a wider code dtype path")
    if dd == 0:
        # every row invalid: no value ever enters the dictionary.
        return {"codes": np.zeros(n, dtype=np.int32), "values": [],
                "dictionary": DictionaryBody(b"", np.zeros(1, dtype=np.int32)),
                "dtype": TEXT_DTYPE, "order": order,
                "validity": None if bool(np.all(valid)) else np.ascontiguousarray(valid),
                "metadata": {"encoding": ENCODING, "d": 0, "n": int(n),
                             "sorted": order == "sorted",
                             "nulls": int(n - int(np.count_nonzero(valid)))}}
    indices = np.ascontiguousarray(
        np.asarray(d.indices.fill_null(0).to_numpy(zero_copy_only=False),
                   dtype=np.int64).reshape(-1))
    if order == "sorted":
        permutation = np.asarray(
            pc.sort_indices(d.dictionary).to_numpy(zero_copy_only=False),
            dtype=np.int64)
        ranks = np.empty(dd, dtype=np.int32)
        ranks[permutation] = np.arange(dd, dtype=np.int32)
        codes = ranks[indices]
        body_array = d.dictionary.take(pa.array(permutation, type=pa.int32()))
    else:
        codes = indices
        body_array = d.dictionary
    codes = np.ascontiguousarray(codes, dtype=np.int32)
    codes[~valid] = 0
    body = _body(body_array)
    if body is None:  # pragma: no cover -- non-standard Arrow buffer layout
        data, offsets = _body_build(body_array.to_pylist(), format_error)
        body = DictionaryBody(data, offsets)
    nulls = int(n - int(np.count_nonzero(valid)))
    return {"codes": codes, "values": body_array.to_pylist(),
            "dictionary": body, "dtype": TEXT_DTYPE, "order": order,
            "validity": None if nulls == 0 else np.ascontiguousarray(valid),
            "metadata": {"encoding": ENCODING, "d": int(dd), "n": int(n),
                         "sorted": order == "sorted", "nulls": nulls,
                         "format": BODY_FORMAT}}


def dict_encode_arrow_metadata_impl(enc, format_error=None):
    """Envelope -> the dictionary metadata view (dict + d + order)."""
    if not isinstance(enc, dict) or "codes" not in enc:
        _err(format_error,
             "dict_encode_arrow_metadata needs an encode envelope with codes.",
             "pass the dict returned by dict_encode_arrow")
    md = dict(enc.get("metadata") or {})
    md.setdefault("encoding", ENCODING)
    md.setdefault("d", len(enc.get("values") or []))
    md.setdefault("sorted", enc.get("order", "sorted") == "sorted")
    return md