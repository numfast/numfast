# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""Vectorized TEXT length primitive (code points, not bytes).

Contract
--------
text_length(values) -> np.int32[N]

  values : 1-D column of str | None. Accepted shapes: python list,
           object ndarray, numpy <U ndarray, pyarrow Array, pyarrow
           ChunkedArray.  Any other carrier is an explicit error.
  result : int32 code-point length per row.  None / null -> 0 and
           never an error (DELTA-3, same rule as the CPU driver TEXT
           ops); empty string -> 0.  Length is Unicode code points, so
           a 2-byte cyrillic char and a 4-byte emoji both count 1.

Why it is vectorized
--------------------
One bulk UTF-8 materialization of the whole column (single C-speed
transcode, Arrow) plus one C-speed length pass over that buffer.  There
is no per-row Python, no object-dtype elementwise comparison, and no
<U ndarray materialization: a <U array is itemsize = 4 bytes x longest
row, so the Referer column (max 2649 chars) would allocate 85.7 GB of
UCS4 before a single length is counted.  Input <U arrays are read
straight out of their UCS4 buffer by the bulk transcode instead.
"""

import numpy as np

try:  # Arrow is the bulk UTF-8 carrier (Storage/Loaders use it too).
    import pyarrow as pa
    import pyarrow.compute as pc
    _HAS_PA = True
except ImportError:  # pragma: no cover -- Arrow-less install
    _HAS_PA = False


def _utf8_carrier(values):
    """Column -> single pyarrow string Array (bulk, no per-row Python)."""
    if not _HAS_PA:
        raise RuntimeError(
            "TextOps.text_length: pyarrow is required for the bulk UTF-8 "
            "carrier (pip install pyarrow).")
    if isinstance(values, dict):
        raise TypeError(
            "TextOps.text_length: got a dict carrier (ENC shape or mapping); "
            "pass the flat str column, or route the ENC codes through the "
            "dictionary length path.")
    if isinstance(values, pa.ChunkedArray):
        return values.combine_chunks()
    if isinstance(values, pa.Array):
        return values
    return pa.array(values, type=pa.string())


def text_length(values):
    """Code-point length of every row of a str|None column -> int32[N]."""
    carrier = _utf8_carrier(values)
    lens = pc.utf8_length(carrier).fill_null(0)
    out = lens.to_numpy(zero_copy_only=False)
    return np.ascontiguousarray(out, dtype=np.int32)
