# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""D-scale TEXT dictionary predicate -> bool[D] lookup table.

Contract
--------
    dict_contains_lut(values, substr)     -> bool[D]   (col LIKE '%substr%')
    dict_not_contains_lut(values, substr) -> bool[D]   (col NOT LIKE '%substr%')
    dict_not_equal_lut(values, other)     -> bool[D]   (col <>  other)
    dict_equal_lut(values, other)         -> bool[D]   (col =   other)

``values`` is any TEXT dictionary carrier accepted by
``carrier.text_dictionary`` (ENC dict, native body, ``[str]``,
``<U``/object ndarray, Arrow Array/ChunkedArray). The result is aligned to
the dictionary code order, i.e. ``lut[codes[i]]`` is the predicate value of
row ``i``'s dictionary entry -- which is the whole point.

Why a LUT and not the "allowed codes" int32 vector
--------------------------------------------------
The row side already has to turn a per-code predicate into a per-row mask.
Two shapes are possible:

  * ``np.isin(codes, allowed)`` -- 10 M binary searches into a sorted
    table of the surviving codes, when the answer is a single D-bit table
    that every row can gather in one indexed read;
  * ``lut[codes]`` -- one gather, one cache-resident table.

For a selective predicate the first shape is a real algorithm and the
second is a table lookup; for a predicate that keeps most of D (which is
what a D-bit table is for) the first shape still pays N log D. The LUT is
returned here so the row side never has to rebuild the table.

The empty needle keeps every entry (LIKE '%%'), matching
``dict_contains``. No per-row Python, no UCS4 materialization, no [str] of
D objects anywhere in either kernel.

Two kernels, one contract
-------------------------
``pc.match_substring`` is the canonical implementation: it walks D
pyarrow strings, and its cost is per *string*, so it is insensitive to how
many bytes those strings hold (measured on the 10 M ClickBench domain:
~310 ns/string, ~373 MB/s).

The body scan is an alternative kernel for the same contract. It never
touches a string object: it finds candidate byte positions across the
whole concatenated UTF-8 body with one vectorized pass, narrows them with
one gather+compare per further needle byte, and maps the survivors to
entry indices with one ``searchsorted``. Its cost is per *byte*, so it wins
by ~2x on a large body of many distinct values and loses on a small one.

Which kernel runs is therefore **measured, not assumed**: ``_scan_wins``
times both on a head of the same body and keeps the byte scan only when
the extrapolated total makes it cheaper by a margin. On any input shape
where it is not, Arrow stays in charge -- the byte scan is a fast path,
never a different answer.

Exactness
---------
The scan is byte containment over the concatenated body, so it must be
told apart from *naive* byte containment in exactly one place: a needle
that straddles the boundary between two dictionary entries is a match in
the flat body and **not** a match in either entry. The scan therefore
rejects any candidate whose extent crosses an entry end, which is the
check ``cands + m <= offsets[ent + 1]`` performs, and ``_scan_wins`` also
requires the head scan to equal the head Arrow result before it is allowed
to take over.

The needle itself is a ``str``. Its UTF-8 encoding is used verbatim, so:

  * a needle that is not valid UTF-8 has no encoding -- it is refused with
    an explicit ``ValueError`` (a ``bytes`` needle is decoded, and a byte
    string that is not valid UTF-8 is refused the same way), never scanned
    and never silently truncated;
  * a needle that is valid UTF-8 can only ever match at a character
    boundary inside a haystack, because UTF-8 is self-synchronizing: no
    lead or continuation byte of a valid sequence occurs in the middle of
    another. So byte containment and code-point containment coincide for
    every ``str`` needle, and multi-byte needles (cyrillic, emoji) need no
    special case;
  * the empty needle is all-true before either kernel runs.

A NULL dictionary value is never a match -- neither for ``contains`` nor
for ``not_contains`` -- which is the 3VL rule ``codes_lut_mask`` relies on.
"""

import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from _lib.carrier import text_dictionary, text_dictionary_array

# Arrow's head pass must be this much cheaper than the scan's whole-body
# pass before the measurement is allowed to pick the scan.
_SCAN_MARGIN = 1.15

# Lane-selection constants. All are shape-free: they describe the sample
# used to compare two kernels, never the data. 16384 entries is where the
# head-sample stop-rate stopped mispredicting the whole-domain rate on
# every needle shape measured over a 2.62 M-value domain.
_HEAD_ENTRIES = 16384
_HEAD_MIN = 64

KERNELS = ("auto", "body", "arrow")


def _err(op, what, fix):
    raise ValueError(f"{op}: {what} Fix: {fix}.")


def _needle_bytes(substr, op):
    """UTF-8 bytes of a ``str`` needle as uint8; refuse non-UTF-8 explicitly."""
    if isinstance(substr, str):
        return np.frombuffer(substr.encode("utf-8"), dtype=np.uint8)
    if isinstance(substr, (bytes, bytearray, memoryview)):
        raw = bytes(substr)
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            _err(op, "the needle is not valid UTF-8 and has no code-point "
                     "reading, so LIKE '%needle%' is undefined for it.",
                 "pass a str, or a byte string that decodes as UTF-8")
        return np.frombuffer(raw, dtype=np.uint8)
    _err(op, f"needs a str needle, got {type(substr).__name__}.",
         "pass a str literal")


def _null_free(lut, arr):
    """A NULL value is never a match -- force it to False, explicitly."""
    n = arr.null_count
    if n:
        lut[arr.is_null().to_numpy(zero_copy_only=False)] = False
    return lut


def _lut(pa_arr):
    return np.ascontiguousarray(
        pa_arr.to_numpy(zero_copy_only=False), dtype=bool).reshape(-1)


# --------------------------------------------------------------------------
# kernel 1 -- byte scan over a native UTF-8 domain body
# --------------------------------------------------------------------------

def _scan_contains_lut(body, needle):
    """``(uint8 body, int32 offsets)`` + needle bytes -> bool[D]."""
    d8, o = body
    d = int(o.size) - 1
    lut = np.zeros(d, dtype=bool)
    m = int(needle.size)
    if d < 1 or m == 0:
        return lut
    n = int(d8.size)
    if m > n:
        return lut
    cands = np.flatnonzero(d8 == needle[0])
    if cands.size == 0:
        return lut
    if m > 1:
        cands = cands[cands <= n - m]
        for j in range(1, m):
            if cands.size == 0:
                return lut
            cands = cands[d8[cands + j] == needle[j]]
    if cands.size == 0:
        return lut
    # A candidate is a match only if its whole extent lives in one entry.
    ent = np.searchsorted(o, cands, side="right")
    ent -= 1
    inside = cands + m <= o[ent + 1]
    if not inside.all():
        ent = ent[inside]
    lut[ent] = True
    return lut


def _time(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def _head(body, k):
    """First ``k`` entries as a valid ``(uint8 body, int32 offsets)`` pair."""
    d8, o = body
    cut = int(o[k])
    return d8[:cut], np.ascontiguousarray(o[: k + 1])


def _head_view(head, k):
    d8, o = head
    return pa.Array.from_buffers(pa.string(), k,
                                 [None, pa.py_buffer(o), pa.py_buffer(d8)])


def _pick_kernel(body, needle, needle_str):
    """``(use_scan, result or None)`` -- the byte scan, or None for Arrow.

    The two kernels are timed on one head of the *same* body and the
    byte scan is kept only if the extrapolated total makes it cheaper by
    ``_SCAN_MARGIN``. Arrow costs per string, the scan costs per body
    byte, so each head time is extrapolated along its own axis. The head
    result is also the correctness witness: the scan is not allowed to
    take over unless it already equals Arrow on that head.

    Below ``_HEAD_MIN`` distinct values there is no per-string work worth
    extrapolating -- both kernels are one call, and the scan's six numpy
    passes do not carry pyarrow's per-call cost -- so the scan is taken
    without spending a measurement that would cost more than the choice.
    """
    d8, o = body
    d = int(o.size) - 1
    if d < 1 or needle.size == 0 or d8.size == 0:
        return False, None
    if d < _HEAD_MIN:
        return True, None
    k = min(d, _HEAD_ENTRIES)
    head = _head(body, k)
    hb = int(head[0].size)
    if hb == 0:
        return False, None
    try:
        view = _head_view(head, k)
        ref = _lut(pc.match_substring(view, needle_str))
        t_arrow = _time(lambda: pc.match_substring(view, needle_str))
        cand = _scan_contains_lut(head, needle)
    except Exception:  # noqa: BLE001 -- Arrow lane owns it
        return False, None
    if not np.array_equal(cand, ref):
        return False, None
    if k == d:
        # the head is the whole domain: this is the answer, not a sample.
        return True, cand
    t_scan = _time(lambda: _scan_contains_lut(head, needle))
    scan_total = t_scan * (int(d8.size) / hb)
    return scan_total * _SCAN_MARGIN < t_arrow * (d / k), None


# --------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------

def _contains(op, values, substr, nb, kernel):
    """``(bool[D] positive LUT, pa.Array)`` for ``col LIKE '%substr%'``."""
    if kernel not in KERNELS:
        _err(op, f"kernel={kernel!r} is not one of {KERNELS}.",
             "pass 'auto', 'body' or 'arrow'")
    arr, body = text_dictionary(values, op)
    d = len(arr)
    if d == 0:
        return np.zeros(0, dtype=bool), arr
    if substr == "":
        return np.ones(d, dtype=bool), arr
    if body is not None:
        use_scan = kernel == "body"
        if not use_scan and kernel == "auto":
            use_scan, scan = _pick_kernel(body, nb, substr)
            if scan is not None:
                return scan, arr
        if use_scan:
            return _scan_contains_lut(body, nb), arr
    elif kernel == "body":
        _err(op, "the body kernel was forced but this carrier has no native "
                 "UTF-8 body (it is a [str] / Arrow / ndarray carrier).",
             "pass kernel='auto' to let the carrier decide")
    return _null_free(_lut(pc.match_substring(arr, substr)), arr), arr


def dict_contains_lut(values, substr, kernel="auto"):
    """``col LIKE '%substr%'`` -> bool[D] (case-sensitive, code-point).

    ``kernel`` picks the implementation for the same contract: ``'auto'``
    measures both and keeps the cheaper, ``'body'`` forces the native
    UTF-8 byte scan (an error if the carrier has no body), ``'arrow'``
    forces ``pc.match_substring``.
    """
    nb = _needle_bytes(substr, "dict_contains_lut")
    return _contains("dict_contains_lut", values, substr, nb, kernel)[0]


def dict_not_contains_lut(values, substr, kernel="auto"):
    """``col NOT LIKE '%substr%'`` -> bool[D] (case-sensitive, code-point).

    Same ``kernel`` argument and same lane selection as ``dict_contains_lut``;
    a NULL value is False here too, never True.
    """
    nb = _needle_bytes(substr, "dict_not_contains_lut")
    pos, arr = _contains("dict_not_contains_lut", values, substr, nb, kernel)
    return _null_free(np.ascontiguousarray(~pos), arr)


def dict_not_equal_lut(values, other):
    """``col <> other`` -> bool[D]."""
    arr = text_dictionary_array(values, "dict_not_equal_lut")
    if len(arr) == 0:
        return np.zeros(0, dtype=bool)
    return _lut(pc.not_equal(arr, other))


def dict_equal_lut(values, other):
    """``col = other`` -> bool[D]."""
    arr = text_dictionary_array(values, "dict_equal_lut")
    if len(arr) == 0:
        return np.zeros(0, dtype=bool)
    return _lut(pc.equal(arr, other))
