# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Dictionary-domain predicates: string work happens on values[D], never on rows.

Contract (row-side stays numeric): dict predicate -> allowed codes (int32)
-> codes membership (numeric, np.isin, C-speed) -> BoolMask[N] -> existing
ir_filter. Dictionary ops are CPU-only by construction (D small, strings
never reach GPU). NULL (3VL): invalid rows never match (mask &= validity).

TEXT inputs accept the boundary [str] list, the native DictionaryBody
(utf8_data + offsets, D-scale lazy materialization), or a body/enc dict.
int64 inputs accept the int64[D] sorted-unique LUT (ndarray, [int] list,
or dtype-"int64" body/enc dict). Semantics are identical for every
carrier; only D entries are ever materialized, never N rows.

Numeric rule: SUM/AVG/arithmetic MUST go LUT->gather->numeric kernel
(lut[codes] via existing ir_gather, then ir_reduce/ir_groupby).
SUM(codes) is meaningless and FORBIDDEN (codes are ranks, not values).
int64 dictionaries are sorted by construction, so code order == value
order: ORDER BY over int64 codes needs no rank LUT. GPU int64 gather is
NOT implemented (separate WGSL task after CPU).
"""

import numbers
from itertools import repeat

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.compute as pc

    _HAS_PA = True
except ImportError:  # verbatim np.char path below stays the contract
    pa, pc, _HAS_PA = None, None, False

TEXT_DTYPE = "text"
INT64_DTYPE = "int64"


def _body_to_list(data, offsets, format_error):
    """Native body -> [str]*D (D-scale boundary materialization, local copy).

    Local copy by design: no cross-module import inside _lib (Builder mount
    owns wiring); decode-side owner is dictionary.py.
    """
    import numpy as _np

    o = _np.ascontiguousarray(_np.asarray(offsets, dtype=_np.int32))
    mv = memoryview(bytes(data))
    try:
        return [bytes(mv[int(a):int(b)]).decode("utf-8")
                for a, b in zip(o[:-1].tolist(), o[1:].tolist())]
    except (UnicodeDecodeError, ValueError) as e:
        if format_error is not None:
            raise format_error("dictionary body is not valid UTF-8 (%s)." % e,
                               fix="pass a body from dictionary_encode",
                               doc="specs/05-storage-encoding.md")
        raise ValueError(f"dictionary body is not valid UTF-8 ({e}). "
                         "Fix: pass a body from dictionary_encode. "
                         "See specs/05-storage-encoding.md")


def _is_int_scalar(v):
    return isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_))


def _coerce_i64(values):
    """int64 LUT carrier -> int64[D] ndarray or None (local copy, no _lib imports)."""
    if isinstance(values, dict):
        if values.get("dtype", TEXT_DTYPE) != INT64_DTYPE:
            return None
        if "dictionary" in values:
            return np.ascontiguousarray(
                np.asarray(values["dictionary"], dtype=np.int64).reshape(-1))
        if isinstance(values.get("values"), (list, tuple, np.ndarray)):
            seq = values["values"] if isinstance(values["values"], (list, tuple)) \
                else values["values"].reshape(-1).tolist()
            if not seq:
                return np.zeros(0, dtype=np.int64)
            if any(not _is_int_scalar(v) for v in seq):
                return None
            return np.ascontiguousarray(np.asarray(list(seq), dtype=np.int64))
        return None
    if isinstance(values, np.ndarray) and values.dtype.kind in "iu":
        return np.ascontiguousarray(values.reshape(-1).astype(np.int64, copy=False))
    if isinstance(values, (list, tuple)) and values \
            and all(_is_int_scalar(v) for v in values):
        return np.ascontiguousarray(np.asarray(list(values), dtype=np.int64))
    return None


def _reject_i64_for_text_op(lut, op, hint, format_error):
    if lut is not None:
        _err(format_error, f"{op} is TEXT-only, got int64 dictionary.",
             hint)


def _err(format_error, what, fix, doc="specs/05-storage-encoding.md"):
    if format_error is not None:
        raise format_error(what, fix=fix, doc=doc)
    raise ValueError(f"{what} Fix: {fix}. See {doc}")


def _u(values, format_error):
    if isinstance(values, dict):
        dt = values.get("dtype", TEXT_DTYPE)
        if dt != TEXT_DTYPE:
            _err(format_error,
                 f"dictionary body dtype {dt!r} is not implemented (TEXT only).",
                 "pass a TEXT dictionary body from dictionary_encode")
        if isinstance(values.get("values"), (list, tuple)):
            vals = list(values["values"])
        elif "utf8_data" in values and "offsets" in values:
            vals = _body_to_list(values["utf8_data"], values["offsets"],
                                 format_error)
        elif "data" in values and "offsets" in values:
            vals = _body_to_list(values["data"], values["offsets"],
                                 format_error)
        else:
            _err(format_error, "dictionary-domain op needs [str] values.",
                 "pass resident['dictionary'] from resident_prepare")
    elif isinstance(values, (list, tuple)):
        vals = list(values)
    elif hasattr(values, "to_list") and hasattr(values, "offsets") \
            and getattr(values, "dtype", TEXT_DTYPE) == TEXT_DTYPE:
        vals = list(values.to_list())
    else:
        _err(format_error, "dictionary-domain op needs [str] values.",
             "pass resident['dictionary'] from resident_prepare")
    if any(not isinstance(s, str) for s in vals):
        _err(format_error, "dictionary-domain op needs [str] values.",
             "pass resident['dictionary'] from resident_prepare")
    return vals


def dict_lookup_impl(values, key, format_error=None):
    """key -> code (int) or None when absent.

    TEXT: O(1)-amortized C scan. int64: searchsorted over the sorted-unique
    LUT (D-scale, C-speed); key must be int (bool rejected).
    """
    lut = _coerce_i64(values)
    if lut is not None:
        if not _is_int_scalar(key):
            _err(format_error, f"dict_lookup key must be int for int64 dictionary, "
                 f"got {type(key).__name__}.", "pass an int literal")
        pos = int(np.searchsorted(lut, int(key), side="left"))
        if pos < lut.size and int(lut[pos]) == int(key):
            return pos
        return None
    vals = _u(values, format_error)
    if not isinstance(key, str):
        _err(format_error, f"dict_lookup key must be str, got {type(key).__name__}.",
             "pass a str literal")
    try:
        return int(vals.index(key))
    except ValueError:
        return None


def dict_range_codes_impl(values, lo, hi, format_error=None):
    """Numeric range 'lo <= col <= hi' -> allowed codes int32 (inclusive).

    searchsorted over the sorted-unique int64 LUT (D-scale, C-speed);
    lo > hi -> empty. TEXT dictionaries -> explicit error (range is
    numeric-only). Row-side 3VL via codes_member_mask as usual.
    """
    lut = _coerce_i64(values)
    if lut is None:
        _err(format_error, "dict_range_codes needs an int64 dictionary.",
             "encode ints via dictionary_encode or use dict_contains/"
             "dict_startswith for TEXT")
    if not _is_int_scalar(lo) or not _is_int_scalar(hi):
        _err(format_error, f"dict_range_codes bounds must be int, got "
             f"{type(lo).__name__}/{type(hi).__name__}.",
             "pass int lo/hi literals")
    if int(lo) > int(hi):
        return np.zeros(0, dtype=np.int32)
    l = int(np.searchsorted(lut, int(lo), side="left"))
    r = int(np.searchsorted(lut, int(hi), side="right"))
    return np.arange(l, r, dtype=np.int32)


def dict_equal_codes_impl(values, key, format_error=None):
    """equality 'col = key' -> allowed codes int32 ([code] or empty)."""
    c = dict_lookup_impl(values, key, format_error)
    return np.array([c] if c is not None else [], dtype=np.int32)


def dict_not_empty_codes_impl(values, format_error=None):
    """`col <> ''` -> allowed codes int32 (all but the '' code, if present)."""
    _reject_i64_for_text_op(_coerce_i64(values), "dict_not_empty_codes",
                            "int64 has no empty string; filter NULLs via validity",
                            format_error)
    vals = _u(values, format_error)
    return np.nonzero(np.array(vals) != "")[0].astype(np.int32)


# NATIVE-ALL: JIT scan kernels + thread-budget helpers deleted (target
# zero). Canonical lane is the zero-copy Arrow bulk in _dict_contains_hit.


def _dict_contains_hit(data, offsets, substr):
    """D-scale contains over native body buffers -> bool[D] or None.

    Zero-copy Arrow bulk over utf8_data+offsets (no Python per-row loop,
    no [str] materialization, no <U padding). Byte search == code-point
    contains for valid UTF-8 (self-synchronizing); bodies arrive valid
    from dictionary_encode.     Lanes: (1) zero-copy Arrow bulk (canonical, C++). NATIVE-ALL retired
    the (2) parallel and (3) single-thread JIT byte-scan lanes (history:
    parallel was ~5.9x over single-thread on D=2.62M/302MB: 570ms ->
    90-121ms at the memcpy floor ~84-101ms; Arrow measured 788.9ms on the
    same D-scale, all lanes chk-exact per FINAL_CLOSE4 H2). Any doubt (no
    Arrow, malformed offsets, matcher error) -> None: the verbatim
    path in dict_contains_impl owns semantics and errors.
    NOTE: no explicit per-call decode-validate pass. The old
    bytes(data).decode("utf-8") re-scan cost ~750ms per 300MB call
    (falsifier 2026-09-20) while Arrow itself is byte-level on invalid
    input (probe: no raise), so the pass only diverted out-of-contract
    invalid buffers to the verbatim path at 40% query cost. Engine
    bodies are valid by construction; invalid hand-crafted buffers now
    get byte-level answers consistent with Arrow semantics.
    """
    try:
        o = np.ascontiguousarray(np.asarray(offsets, dtype=np.int32))
        if o.size == 0 or int(o[0]) != 0:
            return None
        if int(o[-1]) != len(data) or bool((o[1:] < o[:-1]).any()):
            return None
        d = int(o.size) - 1
        if d == 0:
            return np.zeros(0, dtype=bool)
        nb = substr.encode("utf-8")
        if len(nb) == 0:
            return np.ones(d, dtype=bool)
        # NATIVE-ALL: retired JIT scan lanes deleted (target zero). The
        # zero-copy Arrow bulk lane below is canonical (C++, bit-exact vs
        # the retired lanes per FINAL_CLOSE4 H2: all lanes chk-exact).
        if not _HAS_PA:
            return None
        arr = pa.Array.from_buffers(pa.string(), d,
                                    [None, pa.py_buffer(o),
                                     pa.py_buffer(data)])
        m = pc.match_substring(arr, substr).to_numpy(zero_copy_only=False)
        return np.ascontiguousarray(np.asarray(m, dtype=bool))
    except Exception:  # noqa: BLE001 -- verbatim path owns it
        return None


def dict_contains_impl(values, substr, format_error=None):
    """LIKE '%sub%' (case-sensitive) -> allowed codes int32 (C-speed find)."""
    _reject_i64_for_text_op(_coerce_i64(values), "dict_contains",
                            "use dict_range_codes/dict_equal_codes for int64",
                            format_error)
    if isinstance(substr, str):
        _d = _o = None
        if isinstance(values, dict):
            for _dk, _ok in (("utf8_data", "offsets"), ("data", "offsets")):
                if _dk in values and _ok in values:
                    _d, _o = values[_dk], values[_ok]
                    break
        else:
            for _dk, _ok in (("utf8_data", "offsets"), ("data", "offsets")):
                _dv = getattr(values, _dk, None)
                _ov = getattr(values, _ok, None)
                if _dv is not None and _ov is not None:
                    _d, _o = _dv, _ov
                    break
        if _d is not None and _o is not None:
            _hit = _dict_contains_hit(_d, _o, substr)
            if _hit is not None:
                return np.nonzero(_hit)[0].astype(np.int32)
    vals = _u(values, format_error)
    if not isinstance(substr, str):
        _err(format_error, f"dict_contains substr must be str, got {type(substr).__name__}.",
             "pass a str literal")
    if substr == "":
        return np.arange(len(vals), dtype=np.int32)
    hit = np.char.find(np.array(vals, dtype=str), substr) != -1
    return np.nonzero(hit)[0].astype(np.int32)


_NUL = "\x00"
_LEN_LANE_CHUNK = 1 << 17  # 128 KiB of body bytes per pass -> fixed transient


# The <U PADDING CONTRACT (the reason every lane below is gated)
# -----------------------------------------------------------
# np.array(list_of_str, dtype=str) stages each value as UCS-4 followed by NUL
# padding to a SHARED itemsize, and np.char.str_len / np.char.startswith /
# np.argsort read that buffer as a C string. A value's TRAILING NULs are
# therefore padding, not content, and the contract those kernels implement is
#
#     length = len(s.rstrip(NUL))
#     affix  = s.rstrip(NUL).startswith(p.rstrip(NUL))
#     order  = stable sort by s.rstrip(NUL)
#
# NOT len(s) and NOT a byte prefix on the raw body. Proof that the raw value
# IS the padded key exactly when no value ends in a NUL byte -- which is the
# gate every body lane and the list lane below apply, with the verbatim <U
# lane as the fallback whenever the gate says no:
#
#   * s ends in NUL  =>  rstrip(s) != s  =>  the two answers can differ
#     ("ab\0" -> <U length 2, byte length 3; "a\0" byte-min "a" vs padded
#      min "a\0" when "ab" is also present);
#   * s does not end in NUL  =>  rstrip(s) == s, and UTF-8 is order
#     preserving, so byte order == code-point order, so a byte-level
#     pc.starts_with / pc.min_max answers the padded question exactly. For a
#     prefix p that itself ends in no NUL, p is a byte prefix of v iff it is a
#     prefix of rstrip(v) (a non-NUL last character cannot be cut by rstrip),
#     so the value's own trailing NULs never matter.
#
# Exhaustive over every string in {a, NUL}^<=4 for str_len / startswith (35
# prefixes) and over {ab, NUL}^<=4 for argsort, plus the 22-corpus x 3-carrier
# x 4-op x 13-prefix grid and a 400-corpus fuzz; the rule itself is
# pinned by tests/fast/test_ops_dict_ucs4.py.


def _nul_tail_free(vals):
    """True when no value in a [str] list ends in NUL (rstrip is the identity).

    One C-level pass -- map() over the C method str.endswith -- so the gate
    costs no more than a single scan and short-circuits on a NUL corpus.
    """
    return not any(map(str.endswith, vals, repeat(_NUL)))


def _body_buffers(values):
    """(utf8_data, offsets) for any native body carrier, else (None, None).

    Same duck-typed sniff dict_contains_impl uses inline (it keeps its own
    copy: that lane is frozen and out of scope for M7c).
    """
    if isinstance(values, dict):
        for dk, ok in (("utf8_data", "offsets"), ("data", "offsets")):
            if dk in values and ok in values:
                return values[dk], values[ok]
        return None, None
    for dk, ok in (("utf8_data", "offsets"), ("data", "offsets")):
        dv = getattr(values, dk, None)
        ov = getattr(values, ok, None)
        if dv is not None and ov is not None:
            return dv, ov
    return None, None


def _body_ready(data, offsets, need_nul_free=True):
    """Validated int32[D+1] offsets for a body, or None when any doubt remains.

    The offset shape is what both the Arrow buffers and the offset arithmetic
    rely on (starts at 0, ends at len(data), non-decreasing). need_nul_free
    additionally enforces the padding contract above over D bytes, without
    materialising a single string. Any malformed input, missing pyarrow or
    unexpected exception -> None, and the verbatim <U lane owns semantics.
    """
    try:
        o = np.ascontiguousarray(np.asarray(offsets, dtype=np.int32))
        if o.ndim != 1 or o.size == 0 or int(o[0]) != 0:
            return None
        if int(o[-1]) != len(data) or bool((o[1:] < o[:-1]).any()):
            return None
        if need_nul_free:
            last = o[1:] - 1
            keep = last >= o[:-1]          # an empty value has no last byte
            if keep.any() and bool(
                    (np.frombuffer(data, dtype=np.uint8)[last[keep]] == 0).any()):
                return None
        return o
    except (TypeError, ValueError, BufferError, OverflowError):
        return None


def _len_lut_from_body(data, o):
    """Code-point count per value straight from the UTF-8 body -> int32[D].

    count[i] = byte_length[i] - continuation_bytes[i], which for valid UTF-8 is
    the code-point count np.char.str_len reports (a 4-byte emoji is 1, not 4;
    pc.utf8_length would answer 4 and is therefore NOT usable here). No <U
    array, no decode, and the transient mask is a pair of fixed
    _LEN_LANE_CHUNK buffers, so the peak is O(D) whatever maxlen is. The body
    is walked in _LEN_LANE_CHUNK slices, so a single value longer than the
    chunk still works (it becomes a slice of its own).
    """
    d = int(o.size) - 1
    lut = np.empty(d, dtype=np.int32)
    if d == 0:
        return lut
    raw = np.empty(_LEN_LANE_CHUNK + 1, dtype=np.uint8)
    acc = np.empty(_LEN_LANE_CHUNK + 1, dtype=bool)
    lo = 0
    while lo < d:
        base = int(o[lo])
        hi = lo
        while hi < d and (int(o[hi + 1]) - base) <= _LEN_LANE_CHUNK:
            hi += 1
        if hi == lo:
            # one value bigger than the chunk: count it alone, still without a
            # D x maxlen allocation
            n = int(o[lo + 1]) - base
            cont = 0
            for p in range(0, n, _LEN_LANE_CHUNK):
                k = min(_LEN_LANE_CHUNK, n - p)
                blk = np.frombuffer(data, dtype=np.uint8, count=k,
                                    offset=base + p)
                np.bitwise_and(blk, 0xC0, out=raw[:k])
                np.equal(raw[:k], 0x80, out=acc[:k])
                cont += int(np.count_nonzero(acc[:k]))
            lut[lo] = n - cont
            lo += 1
            continue
        n = int(o[hi]) - base
        blk = np.frombuffer(data, dtype=np.uint8, count=n, offset=base)
        np.bitwise_and(blk, 0xC0, out=raw[:n])
        np.equal(raw[:n], 0x80, out=acc[:n])
        acc[n] = False                  # sentinel: an empty last value
        starts = (o[lo:hi] - o[lo]).astype(np.intp)
        cont = np.add.reduceat(acc[:n + 1], starts, dtype=np.int64)
        if hi - lo > 1:
            # starts[k] == starts[k+1] means value k is EMPTY, and reduceat
            # answers it with acc[starts[k]] (the first byte of value k+1), so
            # index k -- not k+1 -- is the one that has to be forced to 0.
            cont[:-1][starts[1:] == starts[:-1]] = 0
        lut[lo:hi] = (o[lo + 1:hi + 1] - o[lo:hi]).astype(np.int32) - cont
        lo = hi
    return lut


def _startswith_from_body(data, o, prefix):
    """startswith over the raw body via zero-copy Arrow -> int32[|hits|].

    The hit mask is read straight out of the Arrow BOOLEAN BITMAP and
    unpacked with np.unpackbits, NOT BooleanArray.to_numpy(zero_copy_only=
    False): that call allocates ~27 MB of one-time, D-independent conversion
    machinery on its first use in a process and Arrow refuses boolean
    zero-copy anyway. Unpacking the bitmap costs one uint8[D] and keeps the
    whole lane O(D).
    """
    if not _HAS_PA:
        return None
    d = int(o.size) - 1
    try:
        arr = pa.Array.from_buffers(pa.string(), d,
                                    [None, pa.py_buffer(o), pa.py_buffer(data)])
        m = pc.starts_with(arr, prefix)
        b = m.buffers()
        if len(b) == 2 and b[0] is None:
            raw = np.frombuffer(b[1], dtype=np.uint8, count=(d + 7) // 8)
            hit = np.unpackbits(raw, bitorder="little")[:d]
        else:                              # unexpected layout: be explicit
            hit = np.asarray(m, dtype=bool)
        return np.nonzero(hit)[0].astype(np.int32)
    except Exception:  # noqa: BLE001 -- verbatim path owns it
        return None


def _min_max_from_body(data, o):
    """MIN/MAX over the raw body via zero-copy Arrow -> (str, str).

    Byte order == code-point order for valid UTF-8 and the gate proved the
    values carry no trailing NUL, so the first minimum / last maximum of the
    byte order is the first minimum / last maximum of the <U padded order. An
    empty array is declined (SQL MIN/MAX over nothing is NULL, which the
    verbatim lane returns as (None, None)).
    """
    if not _HAS_PA or int(o.size) < 2:
        return None
    try:
        arr = pa.Array.from_buffers(pa.string(), int(o.size) - 1,
                                    [None, pa.py_buffer(o), pa.py_buffer(data)])
        mm = pc.min_max(arr).as_py()
        return (mm["min"], mm["max"])
    except Exception:  # noqa: BLE001 -- verbatim path owns it
        return None


def dict_startswith_impl(values, prefix, format_error=None):
    """prefix 'abc%' -> allowed codes int32 (C-speed startswith).

    Three lanes, all bit-identical to the <U staging array they replace
    (padding contract + gate above):
      body  -> zero-copy Arrow byte match, O(D) codes out, no decode;
      list  -> one C-level map(str.startswith) pass, no <U array;
      otherwise (any NUL tail, malformed body, no pyarrow) -> the verbatim
      np.char lane, unchanged, which owns every odd corner.
    """
    _reject_i64_for_text_op(_coerce_i64(values), "dict_startswith",
                            "use dict_range_codes/dict_equal_codes for int64",
                            format_error)
    # gate 1: a needle with a trailing NUL is truncated by the <U compare, so
    # the raw-byte answer is not the staged answer (see the contract above)
    needle_raw = isinstance(prefix, str) and prefix.rstrip(_NUL) == prefix
    if needle_raw:
        _d, _o = _body_buffers(values)
        if _d is not None:
            _so = _body_ready(_d, _o, need_nul_free=False)
            if _so is not None:
                _lane = _startswith_from_body(_d, _so, prefix)
                if _lane is not None:
                    return _lane
    vals = _u(values, format_error)
    if not isinstance(prefix, str):
        _err(format_error, f"dict_startswith prefix must be str, got {type(prefix).__name__}.",
             "pass a str literal")
    if needle_raw and _nul_tail_free(vals):
        hit = list(map(str.startswith, vals, repeat(prefix)))
        return np.nonzero(
            np.fromiter(hit, dtype=bool, count=len(vals)))[0].astype(np.int32)
    hit = np.char.startswith(np.array(vals, dtype=str), prefix)
    return np.nonzero(hit)[0].astype(np.int32)


def dict_len_lut_impl(values, format_error=None):
    """length LUT int32[D]; row_len = lut[codes] via existing gather.

    Three lanes, all bit-identical to np.char.str_len over the <U array:
      body  -> code-point count from the body offsets (bytes - continuation
               bytes), chunked, so neither the UCS4 array nor the [str] list
               is ever built;
      list  -> one C-level map(len) pass, no <U array;
      otherwise -> the verbatim np.char lane, unchanged.
    """
    _reject_i64_for_text_op(_coerce_i64(values), "dict_len_lut",
                            "int64 values need no length LUT; gather the LUT itself",
                            format_error)
    _d, _o = _body_buffers(values)
    if _d is not None:
        _lo = _body_ready(_d, _o)
        if _lo is not None:
            _lane = _len_lut_from_body(_d, _lo)
            if _lane is not None:
                return _lane
    vals = _u(values, format_error)
    if not vals:
        return np.zeros(0, dtype=np.int32)
    if _nul_tail_free(vals):
        return np.fromiter(map(len, vals), dtype=np.int32, count=len(vals))
    return np.char.str_len(np.array(vals, dtype=str)).astype(np.int32)


def dict_ordering_impl(values, format_error=None):
    """Lexicographic order over ANY dictionary order: argsort once (CPU, D).

    Returns {order int32[D], rank_of int32[D]}; row_rank = rank_of[codes]
    via existing gather, then sort/gather over row_rank (all numeric).
    Guarantee: rank order == codepoint-lexicographic (np <U sort == sorted()).
    int64 dictionaries are sorted by construction (code == value rank),
    so ordering is the identity -- use codes directly.

    M7c: this is the ONE predicate of the four that still stages the
    D x maxlen x 4 UCS4 array, and it is deliberate. Ordering is a CONTENT
    question, so no length/offset vector can answer it, and four exact
    replacements were measured against this line and every one is SLOWER at a
    realistic maxlen (D=1e5, maxlen=24: python stable sort by the padded key
    2.05x, Arrow pc.sort_indices 2.66x; at D=20k/maxlen=1024 the python one
    does win, 0.63x, but Arrow loses 4.91x). A fixed-width UCS-4 record sorts
    faster in C than any Python/Arrow path. The cost is memory, not time, and
    it is bounded below by the cheapest exact sort.
    """
    _reject_i64_for_text_op(_coerce_i64(values), "dict_ordering",
                            "int64 code order == value order; sort codes directly",
                            format_error)
    vals = _u(values, format_error)
    if not vals:
        return {"order": np.zeros(0, dtype=np.int32),
                "rank_of": np.zeros(0, dtype=np.int32)}
    order = np.argsort(np.array(vals, dtype=str), kind="stable").astype(np.int32)
    rank_of = np.empty(len(vals), dtype=np.int32)
    rank_of[order] = np.arange(len(vals), dtype=np.int32)
    return {"order": np.ascontiguousarray(order),
            "rank_of": np.ascontiguousarray(rank_of)}


def dict_min_max_impl(values, format_error=None):
    """MIN/MAX = ordered-dictionary boundary.

    TEXT, three lanes, all bit-identical to the <U order's first/last slot:
      body  -> zero-copy Arrow byte min/max, no decode and no <U array;
      list  -> C-level min()/max()/index(), no <U array;
      otherwise (any NUL tail, malformed body, no pyarrow) -> the verbatim
      padded-key lane, unchanged.
    The first minimum and the LAST maximum are the contract, so the tie-break
    is kept explicitly on the list lane. int64: O(1) from the sorted LUT
    borders (lut[0]/lut[-1], D-scale, no scan). Empty dictionary ->
    (None, None) (SQL MIN/MAX over empty = NULL).
    """
    lut = _coerce_i64(values)
    if lut is not None:
        if lut.size == 0:
            return (None, None)
        return (int(lut[0]), int(lut[-1]))
    _d, _o = _body_buffers(values)
    if _d is not None:
        _mo = _body_ready(_d, _o)
        if _mo is not None:
            _lane = _min_max_from_body(_d, _mo)
            if _lane is not None:
                return _lane
    vals = _u(values, format_error)
    if not vals:
        return (None, None)
    if _nul_tail_free(vals):
        hi = max(vals)
        return (min(vals), vals[len(vals) - 1 - vals[::-1].index(hi)])
    keys = [s.rstrip(_NUL) for s in vals]
    return (vals[keys.index(min(keys))],
            vals[len(vals) - 1 - keys[::-1].index(max(keys))])


def codes_member_mask_impl(codes, allowed, validity=None, format_error=None):
    """codes[N] IN allowed -> bool[N] (numeric isin + 3VL validity gate)."""
    c = np.asarray(codes).reshape(-1)
    if not np.issubdtype(c.dtype, np.integer):
        _err(format_error, "codes_member_mask needs integer codes.",
             "pass resident int32 codes")
    a = np.asarray(allowed).reshape(-1)
    mask = np.isin(c, a)
    if validity is not None:
        v = np.asarray(validity, dtype=bool).reshape(-1)
        if v.size != c.size:
            _err(format_error,
                 f"codes_member_mask validity size {v.size} != codes {c.size}.",
                 "pass validity matching codes length",
                 doc="specs/delta-3-null-contract.md")
        mask = mask & v
    return np.ascontiguousarray(mask)


def codes_count_allowed_impl(codes, allowed, validity=None, D=None,
                             format_error=None):
    """Mask-free count of rows whose code is in allowed (exact 3VL).

    count = bincount(codes)[allowed].sum() minus the invalid-code
    correction over codes[~valid] (tiny: 644 rows @10M Q21). No bool[N]
    membership mask: one bincount pass (C-speed) + one isin over the
    invalid rows only. Invalid rows (validity False or code out of
    [0, D)) never match -- identical answer to
    codes_member_mask(...).sum(), verified by the Q21 gates
    (exact/null-invalid/unicode/empty/fuzz/10M-E2E). Generic over
    codes/allowed/validity shapes; no query-specific constants.
    """
    c = np.ascontiguousarray(np.asarray(codes).reshape(-1))
    if not np.issubdtype(c.dtype, np.integer):
        _err(format_error, "codes_count_allowed needs integer codes.",
             "pass resident int32 codes")
    a = np.ascontiguousarray(np.asarray(allowed).reshape(-1))
    if a.size and not np.issubdtype(a.dtype, np.integer):
        _err(format_error, "codes_count_allowed needs integer allowed.",
             "pass allowed codes from dict_contains/dict_equal_codes")
    n = int(c.size)
    if n == 0 or a.size == 0:
        return 0
    if D is None:
        amax = int(np.max(a)) if a.size else -1
        D = amax + 1 if amax >= 0 else 0
    D = int(D)
    if D <= 0:
        return 0
    a_f = np.ascontiguousarray(a[(a >= 0) & (a < D)])
    if a_f.size == 0:
        return 0
    cmin = int(np.min(c)) if n else 0
    cmax = int(np.max(c)) if n else -1
    in_range_all = (cmin >= 0) and (cmax < D)
    if validity is None:
        if in_range_all:
            bc = np.bincount(c, minlength=D)
            return int(np.asarray(bc[a_f].sum()).ravel()[0])
        clipped = np.clip(c, 0, D - 1)
        bc = np.bincount(clipped, minlength=D)
        raw = int(np.asarray(bc[a_f].sum()).ravel()[0])
        bad = (c < 0) | (c >= D)
        if not bool(bad.any()):
            return raw
        corr = int(np.isin(clipped[bad], a_f).sum())
        return int(raw - corr)
    v = np.ascontiguousarray(np.asarray(validity, dtype=bool).reshape(-1))
    if v.size != n:
        _err(format_error,
             f"codes_count_allowed validity size {v.size} != codes {n}.",
             "pass validity matching codes length",
             doc="specs/delta-3-null-contract.md")
    if in_range_all:
        bc = np.bincount(c, minlength=D)
        raw = int(np.asarray(bc[a_f].sum()).ravel()[0])
        if bool(v.all()):
            return raw
        corr = int(np.isin(c[~v], a_f).sum())
        return int(raw - corr)
    clipped = np.clip(c, 0, D - 1)
    bc = np.bincount(clipped, minlength=D)
    raw = int(np.asarray(bc[a_f].sum()).ravel()[0])
    good = v & (c >= 0) & (c < D)
    if bool(good.all()):
        return raw
    corr = int(np.isin(clipped[~good], a_f).sum())
    return int(raw - corr)
