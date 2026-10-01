# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""Generic dict-scale REGEXP_REPLACE for a dictionary-encoded TEXT column.

Contract
--------
regex_replace_dict(codes, values, validity, pattern, repl)

  codes    : int32[N] dictionary codes (any carrier coercible to a flat
             int32 array) of a TEXT dictionary.
  values   : the TEXT dictionary carrier -- [str]*D, a DictionaryBody, an
             encode envelope, a pyarrow string Array, or a <U / object
             ndarray -- in the SAME order the codes index.
  validity : bool[N] | None. False marks an invalid (NULL) row.
  pattern  : str, compiled with Python ``re``. First match only
             (``re.sub(..., count=1)``).
  repl     : str replacement template, Python ``re`` backreference syntax.
  result   : the dictionary_encode envelope for the replaced column --
             {"codes": int32[N], "values": [str]*D', "arrow": pyarrow
             string Array (NULL preserved), "dtype": "text",
             "validity": bool[N] | None,
             "metadata": {encoding, d, n, sorted, nulls}}.

Semantics are DuckDB ``regexp_replace`` parity, frozen: replace the
leftmost non-overlapping match only, ``\\1`` backreferences honoured,
empty pattern matches at position 0, case-sensitive, and an invalid row
maps to NULL (DELTA-3) -- never a match, never an error. A bad pattern or
a bad replacement template is an explicit error at call time, never a
silent NULL.

Why dict-scale
--------------
``re.sub`` is a pure function of the string, so N calls are redundant
whenever the column has a dictionary: the same input value must always
produce the same output value. Applying the pattern once per DISTINCT
value and broadcasting the result over the codes turns an N-row Python
regex pass into a D-row one. On a column with a few million rows and a
few million distinct values that is still a win, because the per-row work
in the flat path is not the regex itself but the Python object
materialization around it (one Python str per row, one object-array
write per row) -- and on any column whose D is much smaller than its N it
is the difference between a per-row cost and a per-value cost.

EXACT BY CONSTRUCTION
---------------------
The output row is ``repl_result[dictionary_value_of(codes[i]]]``, and
``repl_result`` is ``re.sub`` of that very value. A per-row ``re.sub`` of
the input string cannot differ, so this is an identity, not an
approximation -- including for ``$`` and trailing newlines, where Python
``$`` matches before a final newline and the result is whatever Python's
``re.sub`` returns.

The output dictionary is deduplicated and sorted (code == lexicographic
rank), because a regex can collapse many input values onto one output
value: two distinct inputs can share one output, so D' <= D.

Why there is no Arrow-vectorized regex here
-------------------------------------------
Arrow's ``pc.replace_substring_regex`` is RE2, not Python ``re``: RE2's
``$`` is end-of-text while Python's matches before a trailing newline, so
the two disagree on exactly the inputs where a column ends in a newline.
Python ``re.sub`` is the contract; a second, subtly different regex
dialect is not. The cost model here is D-scale Python ``re``, which is
where the flat path's N-scale Python ``re`` was going anyway.
"""

import re

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    _HAS_PA = True
except ImportError:  # pragma: no cover -- Arrow-less install
    pa = pc = None
    _HAS_PA = False

# The encode-envelope vocabulary is a wire contract, not an implementation
# detail, so it is spelled out here rather than imported: cross-Extension
# imports are forbidden by the Builder import-guard (Storage/Dictionary is
# a different Extension, and this one must stay independent of it).
ENCODING = "dictionary-sorted-v1"
TEXT_DTYPE = "text"


def _err(format_error, what, fix, doc="specs/05-storage-encoding.md"):
    if format_error is not None:
        raise format_error(what, fix=fix, doc=doc)
    raise ValueError(f"{what} Fix: {fix}. See {doc}")


def _values_as_strings(values, op, format_error=None):
    """TEXT dictionary carrier -> [str]*D, in code order."""
    if values is None:
        _err(format_error, f"{op} needs the dictionary values.",
             "pass values=[str]*D alongside the codes")
    if isinstance(values, dict):
        v = values.get("values")
        if v is None and isinstance(values.get("dictionary"), (list, tuple)):
            v = values["dictionary"]
        if v is None:
            body = values.get("dictionary")
            if body is not None and hasattr(body, "to_list"):
                v = list(body.to_list())
            else:
                v = None
        if v is None:
            _err(format_error,
                 f"{op}: dictionary carrier exposes no values list.",
                 "pass values=[str]*D alongside the codes")
        values = v
    if hasattr(values, "to_list") and not isinstance(values, np.ndarray):
        try:
            values = list(values.to_list())
        except (UnicodeDecodeError, ValueError) as e:
            _err(format_error, f"{op}: dictionary body is not valid UTF-8 ({e}).",
                 "pass a body from a dictionary encode")
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    if isinstance(values, pa.Array):
        values = values.to_pylist()
    elif isinstance(values, np.ndarray):
        if values.dtype.kind not in "UO":
            _err(format_error, f"{op} is a TEXT op, got a {values.dtype} ndarray.",
                 "pass a <U / object string dictionary carrier")
        values = values.reshape(-1).tolist()
    else:
        values = list(values)
    out = []
    for v in values:
        if v is None or isinstance(v, str):
            out.append(v)
        else:
            _err(format_error,
                 f"{op} needs a [str] dictionary, got a "
                 f"{type(v).__name__} value.", "pass [str]*D")
    return out


def regex_replace_dict_impl(codes, values, validity, pattern, repl,
                            format_error=None):
    """regex_replace_dict_impl(...) -> dictionary_encode envelope. See module."""
    op = "regex_replace_dict"
    if not isinstance(pattern, str):
        _err(format_error,
             f"{op} pattern must be str, got {type(pattern).__name__}.",
             "pass a str literal")
    if not isinstance(repl, str):
        _err(format_error,
             f"{op} repl must be str, got {type(repl).__name__}.",
             "pass a str literal")
    try:
        rx = re.compile(pattern)
    except re.error as e:
        _err(format_error, f"{op} bad pattern {pattern!r}: {e}",
             "pass a valid Python-compatible regex")
    ca = np.ascontiguousarray(np.asarray(codes).reshape(-1).astype(np.int64,
                                                              copy=False))
    n = int(ca.size)
    dv = _values_as_strings(values, op, format_error)
    d = len(dv)
    if n and d:
        lo, hi = int(ca.min()), int(ca.max())
        if lo < 0 or hi >= d:
            _err(format_error,
                 f"{op} code {lo if lo < 0 else hi} out of range D={d}.",
                 "pass codes and values from the same dictionary encode")

    valid = np.ones(n, dtype=bool)
    if validity is not None:
        va = np.asarray(validity, dtype=bool).reshape(-1)
        if va.size != n:
            _err(format_error,
                 f"{op} validity size {va.size} != codes {n}.",
                 "pass validity matching codes length",
                 doc="specs/delta-3-null-contract.md")
        valid = np.ascontiguousarray(valid & va)
    if d == 0:
        # No distinct value at all: every row is invalid by definition.
        return {"codes": np.zeros(n, dtype=np.int32), "values": [],
                "arrow": pa.nulls(n, pa.string()) if _HAS_PA else None,
                "dtype": TEXT_DTYPE, "validity": np.zeros(n, dtype=bool),
                "metadata": {"encoding": ENCODING, "d": 0, "n": n,
                             "sorted": True, "nulls": int(n)}}

    # --- D-scale: one re.sub per distinct value (count=1, DuckDB parity) ---
    replaced = []
    for s in dv:
        if s is None:
            replaced.append(None)
            continue
        try:
            replaced.append(rx.sub(repl, s, count=1))
        except re.error as e:
            _err(format_error, f"{op} replacement {repl!r} failed: {e}",
                 "pass a valid Python replacement template")

    # --- D-scale: a regex can collapse many values onto one, so the output
    #     needs its own dedup + sorted rank. C++ on the D replaced values.
    if _HAS_PA:
        arr = pa.array(replaced, type=pa.string())
        dd = pc.dictionary_encode(arr)
        dp = len(dd.dictionary)
        permutation = np.asarray(
            pc.sort_indices(dd.dictionary).to_numpy(zero_copy_only=False),
            dtype=np.int64)
        ranks = np.empty(dp, dtype=np.int32)
        ranks[permutation] = np.arange(dp, dtype=np.int32)
        old_codes = np.asarray(
            dd.indices.fill_null(0).to_numpy(zero_copy_only=False), dtype=np.int64)
        code_to_new = ranks[old_codes]
        new_values_array = dd.dictionary.take(pa.array(permutation, type=pa.int32()))
    else:  # pragma: no cover -- Arrow-less install
        uniq = sorted({s for s in replaced if s is not None})
        rank = {s: i for i, s in enumerate(uniq)}
        code_to_new = np.asarray(
            [rank.get(s, 0) for s in replaced], dtype=np.int32)
        new_values_array = uniq
        dp = len(uniq)

    in_range = (ca >= 0) & (ca < d)
    out_codes = np.zeros(n, dtype=np.int32)
    np.copyto(out_codes, code_to_new[np.where(in_range, ca, 0)], where=in_range)
    valid = np.ascontiguousarray(valid & in_range)
    out_codes[~valid] = 0

    out_values = (new_values_array.to_pylist() if _HAS_PA else list(new_values_array))
    arrow = None
    if _HAS_PA and dp:
        taken = pc.take(new_values_array, pa.array(out_codes, type=pa.int32()))
        if not bool(valid.all()):
            taken = pc.if_else(pa.array(np.ascontiguousarray(valid)),
                               taken, pa.scalar(None, pa.string()))
        arrow = taken
    elif _HAS_PA:
        arrow = pa.nulls(n, pa.string())
    nulls = int(n - int(np.count_nonzero(valid)))
    return {"codes": np.ascontiguousarray(out_codes, dtype=np.int32),
            "values": out_values, "arrow": arrow, "dtype": TEXT_DTYPE,
            "validity": None if nulls == 0 else np.ascontiguousarray(valid),
            "metadata": {"encoding": ENCODING, "d": int(dp), "n": n,
                         "sorted": True, "nulls": nulls}}