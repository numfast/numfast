# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""One-shot: reverse the M4b fix to reconstruct the PRE-FIX dictionary.py.

Used only to prove the new regression tests FAIL against the original code.
Writes m4b/dictionary_ORIG.py; asserts the result is 886 lines and carries
none of the new symbols. Not part of the product.
"""

import sys

FIXED = "C:/Users/Mikech/AppData/Local/Temp/opencode/m4b/dictionary_FIXED.py"
OUT = "C:/Users/Mikech/AppData/Local/Temp/opencode/m4b/dictionary_ORIG.py"

PAIRS = [
    # 1. helpers added for BUG 2
    ('''def _arrow_text_layout(arr):
    """True when ``arr`` is a TEXT/binary Arrow array (the 3-buffer layout).

    Gate for the Arrow-native path: a non-text Arrow array carries fewer
    than three buffers, so indexing ``bufs[2]`` on it is a raw IndexError.
    binary/large_binary stay here on purpose -- their offsets buffer is the
    3-buffer layout _encode_arrow_text already reads.
    """
    if pa is None:
        return False
    try:
        t = arr.type
        return (pa.types.is_string(t) or pa.types.is_large_string(t)
                or pa.types.is_binary(t) or pa.types.is_large_binary(t))
    except Exception:
        return False


def _arrow_dict_text(values):
    """Arrow dictionary-typed carrier -> plain text pa.Array, else None.

    Reuses the ONE canonical normalisation already proven in the Arrow
    adapter (``col.cast(pa.string()).combine_chunks()``): Arrow rewrites
    the index stream into string offsets over the dictionary's own value
    buffer, after which the existing Arrow-native text path owns dedup and
    the sorted-unique re-rank. Non-text dictionaries return None (never a
    silent stringify -- an int64 dictionary must reach the int64 path).
    """
    if pa is None:
        return None
    try:
        t = values.type
        if not pa.types.is_dictionary(t):
            return None
        if not (pa.types.is_string(t.value_type)
                or pa.types.is_large_string(t.value_type)):
            return None
        out = values.cast(pa.string())
        # Array.cast -> StringArray (already single chunk); the Arrow adapter
        # casts a ChunkedArray, whose cast stays chunked and needs merging.
        return out.combine_chunks() if isinstance(out, pa.ChunkedArray) else out
    except Exception:
        return None


def _arrow_cxx_sorted_encode(arr, validity, format_error):''',
     '''def _arrow_cxx_sorted_encode(arr, validity, format_error):'''),
    # 2. dispatch guard for BUG 2
    ('''    # --- Arrow-native path: skip to_pylist + object array + Python dedup ---
    if pa is not None and isinstance(values, (pa.Table, pa.RecordBatch)):
        _err(format_error,
             f"dictionary_encode got {type(values).__name__}, a table of "
             "columns, not a column.",
             "pass one column: table.column('name') or table['name']")
    if pa is not None and isinstance(values, (pa.Array, pa.ChunkedArray)):
        # Dictionary-encoded text carrier: normalize with the canonical Arrow
        # cast, then the plain text path owns the rest. Without this the
        # index stream reaches bufs[2] below as a bare IndexError.
        _dt = _arrow_dict_text(values)
        if _dt is not None:
            return _encode_arrow_text(_dt, validity, format_error)
    if pa is not None and isinstance(values, pa.Array):
        if not _arrow_text_layout(values):
            _err(format_error,
                 f"dictionary_encode needs a TEXT Arrow array, got {values.type}.",
                 "pass pa.string()/pa.large_string() data, a dictionary-encoded "
                 "string column, or a [str|None] / int64 column")
        return _encode_arrow_text(values, validity, format_error)''',
     '''    # --- Arrow-native path: skip to_pylist + object array + Python dedup ---
    if pa is not None and isinstance(values, pa.Array):
        return _encode_arrow_text(values, validity, format_error)'''),
    # 3. _all_null helper + docstring
    ('''def _all_null(codes, d, validity, format_error):
    """D==0 -> [None]*N (every row NULL, exactly).

    DELTA-3 gives an all-NULL column a zero-length dictionary, so no value
    can legally sit at code 0 and every code is the NULL placeholder --
    true with or without the sidecar, since there is nothing to gather. A
    negative code is still an error (corrupt carrier, never a NULL row), and
    a sidecar of the wrong length still fails loudly.
    """
    if codes.size and int(codes.min()) < 0:
        bad = int(codes[np.nonzero(codes < 0)[0][0]])
        _err(format_error, f"dictionary_decode code {bad} out of range D={d}.",
             "pass codes from dictionary_encode of the same dictionary")
    if validity is not None:
        v = np.asarray(validity, dtype=bool).reshape(-1)
        if v.size != codes.size:
            _err(format_error,
                 f"dictionary_decode validity size {v.size} != codes {codes.size}.",
                 "pass validity matching codes length",
                 doc="specs/delta-3-null-contract.md")
    return [None] * int(codes.size)


def dictionary_decode_impl''',
     '''def dictionary_decode_impl'''),
    # 4. docstring contract block
    ('''    never wrap.

    ``validity`` omitted == "every code in ``codes`` is valid" (the encode
    envelope returns ``validity=None`` for an all-valid column): decode is
    then a pure gather over the LUT. It CANNOT restore NULLs, and must not
    pretend to -- under DELTA-3 the NULL placeholder IS code 0, which is
    also the sorted rank of the smallest real value, so NULL and a genuine
    value are indistinguishable once the sidecar is dropped. Pass the
    envelope's ``validity`` to get NULLs back. The one exact exception is
    D=0 (an all-NULL column): no value exists, so every row decodes to None.
    """''',
     '''    never wrap.
    """'''),
    # 5a. int64 D==0 guard
    ('''        d = lut.size
        if d == 0:
            return _all_null(c, d, validity, format_error)
        if c.size and (int(c.min()) < 0 or int(c.max()) >= d):''',
     '''        d = lut.size
        if c.size and (int(c.min()) < 0 or int(c.max()) >= d):'''),
    # 5b. TEXT D==0 guard
    ('''    d = len(vals)
    if d == 0:
        return _all_null(c, d, validity, format_error)
    if c.size and (int(c.min()) < 0 or int(c.max()) >= d):''',
     '''    d = len(vals)
    if c.size and (int(c.min()) < 0 or int(c.max()) >= d):'''),
]


def main():
    src = open(FIXED, encoding="utf-8").read()
    for i, (new, old) in enumerate(PAIRS, 1):
        if src.count(new) != 1:
            print(f"FAIL pair {i}: found {src.count(new)} occurrences")
            return 1
        src = src.replace(new, old)
    for sym in ("_arrow_text_layout", "_arrow_dict_text", "_all_null",
                "TEXT Arrow array", "every code in ``codes`` is valid"):
        if sym in src:
            print(f"FAIL: {sym!r} still present")
            return 1
    n = src.count("\n")
    if n != 886:
        print(f"FAIL: {n} newlines, expected 886")
        return 1
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        fh.write(src)
    print(f"OK -> {OUT} ({len(src.encode('utf-8'))} bytes, {n} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
