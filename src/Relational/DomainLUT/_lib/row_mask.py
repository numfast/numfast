# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Row mask from D-scale dictionary LUTs: one gather per predicate.

Contract
--------
    codes_lut_mask(codes, luts, validities=None) -> bool[N]

``codes``      sequence of integer code columns, all of length N;
``luts``       one bool[D_j] table per code column (D_j = that
               dictionary's cardinality);
``validities`` optional per-column bool[N] NULL gates, or None.

    mask[i] = AND_j ( luts[j][codes[j][i]] AND validities[j][i] )

i.e. the row is kept only if every dictionary predicate holds for every
column of that row and no column is NULL. 3VL is exact: an invalid row
never matches, and an empty LUT rejects every row, without either being a
special case.

The all-true LUT
----------------
One gather per column is the cost floor of the shape, and for a predicate
that keeps every code of its dictionary the gather is not a computation
at all -- it is the identity on a bool array. That is a *data fact about
the LUT*, not a fact about the query, and it is checked here as one:
``lt.all()`` reads D bits and costs 0.18-0.65 ms at D ~ 0.8-2.6 M, which is
two to three orders of magnitude below the ~48 ms the 10 M-row gather it
replaces costs. When it holds, ``out &= lt[codes]`` is dropped and only
the validity gate is applied, which is exactly equivalent: ``& True`` is
the identity and every code is already proven in ``[0, D)`` by
``_code_column``, so the gather could not have produced a False.

It is not a fast path for a lucky query: any dictionary whose predicate
selects everything hits it, and any dictionary with a single rejecting
code does not, with the same code and no knowledge of which is which.

Why a gather and not np.isin
----------------------------
A dictionary predicate is a bit per code, so the answer is a D-bit table
and every row already holds its code. ``lut[codes]`` is one indexed read
per column at the memory floor. ``np.isin(codes, allowed)`` instead sorts
the surviving codes and runs N binary searches -- log D dependent misses
per row, on a table far larger than the LUT that answers the same
question. The LUT is the representation that matches the predicate.
"""

import numpy as np


def _err(what, fix):
    raise ValueError(f"codes_lut_mask: {what} Fix: {fix}.")


def _code_column(col, lut, n, j):
    c = np.asarray(col)
    if c.ndim != 1:
        _err(f"codes[{j}] is not 1-D (ndim={c.ndim}).", "pass a flat code column")
    if c.dtype.kind not in "iu":
        _err(f"codes[{j}] has dtype {c.dtype}, expected integer codes.",
             "pass resident int32/int64 codes")
    if c.size != n:
        _err(f"codes[{j}] has {c.size} rows, expected {n}.",
             "all code columns must describe the same rows")
    if c.size == 0:
        return c.astype(np.int64, copy=False)
    if int(c.min()) < 0:
        _err(f"codes[{j}] has a negative code.",
             "dictionary codes are positions in the sorted value list, [0, D)")
    d = int(lut.size)
    if d > 0 and int(c.max()) >= d:
        _err(f"codes[{j}] has a code >= D={d}.",
             "pass codes and the LUT of the same dictionary")
    return c


def codes_lut_mask(codes, luts, validities=None):
    """AND of per-column dictionary LUT gathers and validity gates -> bool[N]."""
    code_cols = list(codes)
    lut_cols = list(luts)
    if not code_cols:
        _err("no code column given.",
             "pass at least one (codes, lut) pair")
    if len(lut_cols) != len(code_cols):
        _err(f"{len(lut_cols)} LUTs for {len(code_cols)} code columns.",
             "pass exactly one LUT per code column")
    if validities is None:
        val_cols = [None] * len(code_cols)
    else:
        val_cols = list(validities)
        if len(val_cols) != len(code_cols):
            _err(f"{len(val_cols)} validity columns for {len(code_cols)} code columns.",
                 "pass exactly one validity per code column (or None)")

    n = int(np.asarray(code_cols[0]).size)
    for j, lut in enumerate(lut_cols):
        lt = np.asarray(lut)
        if lt.ndim != 1 or lt.dtype != np.bool_:
            _err(f"luts[{j}] must be a 1-D bool table, got shape {lt.shape} "
                 f"dtype {lt.dtype}.", "pass the D-scale bool LUT as returned")
    checked = [_code_column(code_cols[j], lut_cols[j], n, j)
               for j in range(len(code_cols))]

    out = np.ones(n, dtype=bool)
    for j, c in enumerate(checked):
        lt = np.asarray(lut_cols[j], dtype=bool)
        if lt.size == 0:
            # empty dictionary: no code can belong to it, so no row matches.
            out[:] = False
            continue
        if not lt.all():
            # not all-true -> the gather is a real per-row predicate
            out &= lt[c]
        # all-true -> the gather is the identity; the validity gate below
        # is the only thing this column can still remove.
        v = val_cols[j]
        if v is not None:
            vb = np.asarray(v, dtype=bool).reshape(-1)
            if vb.size != n:
                _err(f"validities[{j}] has {vb.size} rows, expected {n}.",
                     "pass validity matching codes length")
            out &= vb
    return np.ascontiguousarray(out)
