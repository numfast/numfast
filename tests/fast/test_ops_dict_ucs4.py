# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""M7c gate: the dictionary-domain predicates must not stage D x maxlen x 4 UCS4.

Three of the four F4 predicates (dict_len_lut, dict_startswith,
dict_min_max) compute per value at D scale, so they allocate O(D) and never a
D x maxlen x 4 array. This file pins three things at once:

  1. the ALLOCATION -- tracemalloc peak stays O(D), provably far below
     D x maxlen x 4 (the array the old np.array(vals, dtype=str) built);
  2. the NUL CONTRACT -- numpy's <U staging buffer pads each value with NULs
     and np.char.*/np.argsort read it as a C string, so a value's TRAILING
     NULs are padding. That quirk is part of the frozen output and pyarrow's
     byte-level pc.* kernels do NOT reproduce it, so the fix carries the
     quirk instead of silently correcting it. Pinned by brute force over
     every string in {a, NUL}^<=4;
  3. the ORACLE -- the per-value lanes must equal an independent raw-byte
     computation, except where the oracle and the frozen <U contract
     disagree, and there the <U contract wins.
"""

import gc
import itertools
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])
SEED = 42
NUL = "\x00"
# Every string over {a, NUL} up to length 4 -- 121 values, exhaustive.
NUL_ALPHABET = ["".join(p) for n in range(0, 5)
                for p in itertools.product("ab", (NUL,), repeat=n)]


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


def body(values):
    """The ENC dictionary shape: one UTF-8 body + int32 offsets."""
    offs = np.zeros(len(values) + 1, dtype=np.int32)
    offs[1:] = np.cumsum([len(v.encode("utf-8")) for v in values])
    return {"utf8_data": "".join(values).encode("utf-8"), "offsets": offs}


def carriers(kernel, values):
    """Every carrier _u() accepts, so a lane cannot be carrier-specific."""
    from Storage.Dictionary._lib.dictionary import DictionaryBody

    out = {"list": list(values), "body": body(values)}
    if values:
        b = body(values)
        out["DictionaryBody"] = DictionaryBody(b["utf8_data"], b["offsets"])
    return out


def peak_mb(fn):
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    out = fn()
    pk = tracemalloc.get_traced_memory()[1] / 2 ** 20
    tracemalloc.stop()
    del out
    gc.collect()
    return pk


# ---------- (1) the allocation ----------

# D is small and maxlen is huge: the old code reserved D x maxlen x 4 here
# and got nothing for it. The new lanes must stay O(D).
SMALL_D = 512
HUGE_MAXLEN = 4096
HUGELY_WASTEFUL = SMALL_D * HUGE_MAXLEN * 4 / 2 ** 20  # 8.0 MB


def _corpus_512_of_4096():
    rng = np.random.default_rng(SEED)
    pool = "ab\u0430\u4f60\U0001f600"
    vals = ["".join(pool[int(j)] for j in rng.integers(0, len(pool), size=8))
            for _ in range(SMALL_D - 1)]
    vals.append("L" * HUGE_MAXLEN)
    assert max(len(v) for v in vals) == HUGE_MAXLEN
    return vals


@pytest.mark.fast
def test_len_lut_does_not_stage_the_U_bomb(kernel):
    a = kernel.alias
    vals = _corpus_512_of_4096()
    want_mb = len(vals) * HUGE_MAXLEN * 4 / 2 ** 20
    for name, carrier in carriers(kernel, vals).items():
        pk = peak_mb(lambda c=carrier: a["dict_len_lut"](c))
        assert pk < want_mb / 8, (
            f"list/len_lut via {name}: peak {pk:.3f} MB is not O(D); the old "
            f"np.array(vals, dtype=str) staged {want_mb:.1f} MB")


@pytest.mark.fast
def test_startswith_does_not_stage_the_U_bomb(kernel):
    a = kernel.alias
    vals = _corpus_512_of_4096()
    want_mb = len(vals) * HUGE_MAXLEN * 4 / 2 ** 20
    for name, carrier in carriers(kernel, vals).items():
        pk = peak_mb(lambda c=carrier: a["dict_startswith"](c, "L"))
        assert pk < want_mb / 8, (
            f"startswith via {name}: peak {pk:.3f} MB is not O(D)")


@pytest.mark.fast
def test_min_max_does_not_stage_the_U_bomb(kernel):
    a = kernel.alias
    vals = _corpus_512_of_4096()
    want_mb = len(vals) * HUGE_MAXLEN * 4 / 2 ** 20
    for name, carrier in carriers(kernel, vals).items():
        pk = peak_mb(lambda c=carrier: a["dict_min_max"](c))
        assert pk < want_mb / 8, (
            f"min_max via {name}: peak {pk:.3f} MB is not O(D)")


@pytest.mark.fast
def test_len_lut_peak_tracks_D_not_maxlen(kernel):
    """The peak must be flat in maxlen -- that is the whole point of the fix."""
    a = kernel.alias
    d = 4096
    peaks = []
    for maxlen in (8, 64, 1024, 4096):
        rng = np.random.default_rng(SEED)
        vals = ["".join("ab\u0430"[int(j)]
                        for j in rng.integers(0, 3, size=maxlen)) for _ in range(d - 1)]
        vals.append("L" * maxlen)
        peaks.append(peak_mb(lambda v=vals: a["dict_len_lut"](v)))
    # maxlen grew 512x; the peak may not grow with it.
    assert max(peaks) < 4 * min(peaks) + 1.0, f"peak still scales with maxlen: {peaks}"


# ---------- (2) the NUL contract (brute force) ----------

@pytest.mark.fast
def test_nul_suffix_is_padding_not_content():
    """The frozen <U contract: np.char.str_len counts code points BEFORE the
    first NUL run that reaches the end of the value. A value that is nothing
    but NULs has length 0, and 'abc' + NUL has length 3, NOT 4.

    This is numpy's <U staging buffer read as a C string, and it is what
    dict_len_lut has always returned. pyarrow's pc.utf8_length disagrees
    (it returns 1 and 4), which is why the fix reproduces the <U rule
    instead of routing through Arrow.
    """
    for s in NUL_ALPHABET:
        staged = int(np.char.str_len(np.array([s], dtype=str))[0])
        padded = s.rstrip(NUL)
        assert staged == len(padded), f"{s!r}: <U says {staged}"
        # and the Arrow kernel really does disagree, so the two are not
        # interchangeable -- this test is a warning, not a tautology
        if s.endswith(NUL) and s:
            assert len(s) != staged, f"{s!r} unexpectedly agrees with Arrow"


@pytest.mark.fast
def test_len_lut_matches_the_staged_U_rule_on_every_nul_string(kernel):
    a = kernel.alias
    for s in NUL_ALPHABET:
        got = a["dict_len_lut"]([s, "pad"])
        assert got.tolist() == [len(s.rstrip(NUL)), 3], s.encode("unicode_escape")
        got_body = a["dict_len_lut"](body([s, "pad"]))
        assert got_body.tolist() == got.tolist()


@pytest.mark.fast
def test_startswith_matches_the_staged_U_rule_on_every_nul_pair(kernel):
    """np.char.startswith on <U compares the C-string forms, so a needle with
    a trailing NUL is silently truncated -- 'a\\x00' matches 'ab'. Frozen."""
    a = kernel.alias
    for s in NUL_ALPHABET:
        for p in NUL_ALPHABET:
            want = [i for i, v in enumerate([s, "a"])
                    if v.rstrip(NUL).startswith(p.rstrip(NUL))]
            assert a["dict_startswith"]([s, "a"], p).tolist() == want, \
                f"{s!r} / {p!r}"


@pytest.mark.fast
def test_min_max_takes_the_first_min_and_the_last_max(kernel):
    """dict_min_max reads the FIRST and LAST slot of the stable <U order. With
    padded keys, 'bbb' and 'bbb\\x00' are the same key, so the max is whichever
    of the two appears LAST in the dictionary."""
    a = kernel.alias
    for s in NUL_ALPHABET:
        for t in NUL_ALPHABET:
            vals = [s, "m" + t, "a" + t]
            want = (vals[sorted(range(len(vals)),
                                 key=lambda i: vals[i].rstrip(NUL))[0]],
                    vals[sorted(range(len(vals)),
                                 key=lambda i: vals[i].rstrip(NUL))[-1]])
            assert a["dict_min_max"](vals) == want, (s, t)


# ---------- (3) the independent raw-byte oracle ----------

def _oracle_len(data, offs):
    """Code points per value straight from the raw bytes: byte length minus
    the continuation bytes. No numpy <U array, no pyarrow."""
    b = np.frombuffer(data, dtype=np.uint8)
    cont = (b & 0xC0) == 0x80
    csum = np.empty(cont.size + 1, dtype=np.int64)
    csum[0] = 0
    np.cumsum(cont, dtype=np.int64, out=csum[1:])
    o = offs.astype(np.int64)
    return np.asarray(np.diff(o) - (csum[o[1:]] - csum[o[:-1]]), dtype=np.int32)


@pytest.mark.fast
def test_len_lut_equals_the_raw_byte_oracle_whenever_no_value_ends_in_nul(kernel):
    """Without a trailing NUL the <U padding is a no-op, so the frozen output
    and the independent raw-byte oracle must agree exactly."""
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    alpha = "abZ\u0430\u4f60\U0001f600 \t\"\\"
    checked = 0
    for _ in range(300):
        d = int(rng.integers(0, 40))
        vals = ["".join(alpha[int(j)] for j in rng.integers(0, len(alpha),
                                                           size=int(rng.integers(0, 14))))
                for _ in range(d)]
        b = body(vals)
        want = _oracle_len(b["utf8_data"], b["offsets"])
        got = a["dict_len_lut"](b)
        assert got.dtype == np.int32
        assert np.array_equal(got, want), vals
        checked += d
    assert checked > 3000


@pytest.mark.fast
def test_startswith_equals_the_raw_byte_oracle_whenever_no_nul_anywhere(kernel):
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    alpha = "abZ\u0430\u4f60\U0001f600"
    for _ in range(300):
        d = int(rng.integers(0, 40))
        vals = ["".join(alpha[int(j)] for j in rng.integers(0, len(alpha),
                                                           size=int(rng.integers(0, 14))))
                for _ in range(d)]
        b = body(vals)
        data, offs = b["utf8_data"], b["offsets"]
        for pfx in ("", "a", "abZ", "\u0430", "\U0001f600", "zzzz"):
            want = []
            mv = memoryview(data)
            o = offs.tolist()
            pb = pfx.encode("utf-8")
            for i, (x, y) in enumerate(zip(o[:-1], o[1:])):
                if bytes(mv[x:y])[:len(pb)] == pb:
                    want.append(i)
            assert a["dict_startswith"](b, pfx).tolist() == want, (vals, pfx)


@pytest.mark.fast
def test_min_max_equals_python_min_max_without_nul(kernel):
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    alpha = "abZ\u0430\u4f60\U0001f600"
    for _ in range(200):
        d = int(rng.integers(0, 40))
        vals = ["".join(alpha[int(j)] for j in rng.integers(0, len(alpha),
                                                           size=int(rng.integers(0, 12))))
                for _ in range(d)]
        if not vals:
            assert a["dict_min_max"](vals) == (None, None)
            continue
        assert a["dict_min_max"](vals) == (min(vals), max(vals)), vals
        assert a["dict_min_max"](body(vals)) == (min(vals), max(vals)), vals


# ---------- (4) the untouched sibling must not drift ----------

@pytest.mark.fast
def test_dict_ordering_still_reports_the_U_padding_key(kernel):
    """dict_ordering was NOT changed (every exact replacement measured slower).
    Pin the current behaviour so a future edit cannot move it unnoticed."""
    a = kernel.alias
    for s in NUL_ALPHABET:
        for t in NUL_ALPHABET:
            vals = [s, t, "m" + s, t]
            want = sorted(range(len(vals)), key=lambda i: vals[i].rstrip(NUL))
            assert a["dict_ordering"](vals)["order"].tolist() == want, (s, t)


# ---------- (5) shapes, dtypes and the empty contract ----------

@pytest.mark.fast
def test_empty_and_single_value_corpora(kernel):
    a = kernel.alias
    for empty in ([], body([])):
        lut = a["dict_len_lut"](empty)
        assert lut.size == 0 and lut.dtype == np.int32
        sw = a["dict_startswith"](empty, "a")
        assert sw.size == 0 and sw.dtype == np.int32
        assert a["dict_min_max"](empty) == (None, None)
        o = a["dict_ordering"](empty)
        assert o["order"].size == 0 and o["rank_of"].size == 0
    one = ["only"]
    assert a["dict_len_lut"](one).tolist() == [4]
    assert a["dict_len_lut"](body(one)).tolist() == [4]
    assert a["dict_startswith"](one, "on").tolist() == [0]
    assert a["dict_startswith"](one, "").tolist() == [0]
    assert a["dict_startswith"](one, "nope").tolist() == []
    assert a["dict_min_max"](one) == ("only", "only")
    assert a["dict_min_max"](body(one)) == ("only", "only")


@pytest.mark.fast
def test_all_null_column_keeps_the_d0_lane(kernel):
    """A NULL lives on the row side, so an all-NULL column encodes to D=0."""
    import pyarrow as pa

    a = kernel.alias
    enc = a["dictionary_encode"](pa.array([None] * 64, type=pa.string()))
    assert enc["metadata"]["d"] == 0
    for carrier in (list(enc["values"]), enc["dictionary"]):
        assert a["dict_len_lut"](carrier).size == 0
        assert a["dict_min_max"](carrier) == (None, None)


@pytest.mark.fast
def test_mixed_null_column_length_lut_ignores_the_nulls(kernel):
    import pyarrow as pa

    a = kernel.alias
    enc = a["dictionary_encode"](
        pa.array([None if i % 3 == 0 else "v%d" % (i % 7) for i in range(300)],
                 type=pa.string()))
    lut = a["dict_len_lut"](enc["dictionary"])
    assert lut.tolist() == [len(v) for v in enc["values"]]
    assert a["dict_min_max"](enc["dictionary"]) == (min(enc["values"]),
                                                    max(enc["values"]))


@pytest.mark.fast
def test_int64_lane_is_still_rejected_by_the_text_ops(kernel):
    a = kernel.alias
    i64 = np.array([5, 1, 9, 3], dtype=np.int64)
    for op, args in (("dict_len_lut", ()), ("dict_startswith", ("1",)),
                     ("dict_ordering", ())):
        with pytest.raises(ValueError, match="TEXT-only"):
            a[op](i64, *args)
    # dict_min_max is the one int64-aware op; that branch is untouched and
    # still reads the LUT borders, which is only MIN/MAX for a SORTED LUT
    # (the int64 dictionary invariant: code == sorted rank).
    assert a["dict_min_max"](np.sort(i64)) == (1, 9)
    assert a["dict_min_max"]([]) == (None, None)


@pytest.mark.fast
def test_non_str_prefix_still_raises_the_project_error(kernel):
    a = kernel.alias
    with pytest.raises(ValueError, match="prefix must be str"):
        a["dict_startswith"](["abc"], 5)
    with pytest.raises(ValueError, match="prefix must be str"):
        a["dict_startswith"](body(["abc"]), None)


@pytest.mark.fast
def test_startwith_does_not_need_a_body_or_arrow(kernel):
    """The new lanes are pure Python -- no pyarrow, no <U array, no body."""
    a = kernel.alias
    vals = ["abc", "abd", "xyz", "ab"]
    assert a["dict_startswith"](vals, "ab").tolist() == [0, 1, 3]
    assert a["dict_len_lut"](vals).tolist() == [3, 3, 3, 2]
    assert a["dict_min_max"](vals) == ("ab", "xyz")
    assert a["dict_min_max"](body(vals)) == ("ab", "xyz")


@pytest.mark.fast
def test_boundary_4096_lengths_and_multibyte(kernel):
    a = kernel.alias
    for n in (4095, 4096, 4097):
        vals = ["L" * n, "x", "y" * (n - 1)]
        got = a["dict_len_lut"](body(vals))
        assert got.tolist() == [n, 1, n - 1], n
        assert a["dict_startswith"](body(vals), "L" * n).tolist() == [0]
    # a 4096-BYTE value made of 3-byte code points
    vals = ["\u4f60" * 1365, "\u4f60" * 1366]
    assert a["dict_len_lut"](body(vals)).tolist() == [1365, 1366]
    assert a["dict_min_max"](body(vals)) == ("你" * 1365, "你" * 1366)


# ---------- (6) the zero-copy body lanes (M7c second pass) ----------
# The lanes above are pure Python, so a body carrier still paid the D-scale
# decode (_body_to_list) before answering. The body lanes below answer from
# utf8_data+offsets directly. They are gated on the <U padding contract (see
# the contract block in _lib/domain.py): no value may end in a NUL byte for the
# length and MIN/MAX lanes, and the needle may not end in one for startswith.
# Every gate miss falls back to the verbatim <U lane.

def _staged_len(values):
    """The frozen reference: np.char.str_len over the <U staging array."""
    if not values:
        return np.zeros(0, dtype=np.int32)
    return np.char.str_len(np.array(values, dtype=str)).astype(np.int32)


def _staged_order(values):
    return np.argsort(np.array(values, dtype=str), kind="stable")


def _staged_min_max(values):
    if not values:
        return (None, None)
    o = _staged_order(values)
    return (values[int(o[0])], values[int(o[-1])])


def _staged_startswith(values, prefix):
    if not values:
        return np.zeros(0, dtype=np.int32)
    return np.nonzero(np.char.startswith(np.array(values, dtype=str),
                                         prefix))[0].astype(np.int32)


def _chunk():
    from Storage.Dictionary._lib.domain import _LEN_LANE_CHUNK

    return _LEN_LANE_CHUNK


@pytest.mark.fast
def test_body_lane_does_not_materialise_the_string_list(kernel):
    """A body carrier must not decode. D=20000 of ~800 bytes would cost
    ~20 MB of Python str objects plus the 62 MB <U array before any answer;
    the lane answers from the offsets alone."""
    a = kernel.alias
    ch = _chunk()
    rng = np.random.default_rng(SEED)
    pool = "ab" + "а" + "\U0001f600"
    vals = ["".join(pool[int(j)] for j in rng.integers(0, len(pool), size=200))
            for _ in range(19_999)]
    vals.append("L" * ch)                      # force several chunks
    b = body(vals)
    staged_mb = len(vals) * ch * 4 / 2 ** 20
    for op, args in (("dict_len_lut", ()), ("dict_startswith", ("a",)),
                     ("dict_min_max", ())):
        pk = peak_mb(lambda o=op, g=args: a[o](b, *g))
        assert pk < staged_mb / 8, f"{op} on a body peaked {pk:.3f} MB; the " \
            f"decode + <U staging cost {staged_mb:.1f} MB"


@pytest.mark.fast
def test_len_lut_counts_code_points_not_bytes(kernel):
    a = kernel.alias
    vals = ["\U0001f600", "你", "а", "a",
            "\U0010ffff", "z" * 3]
    assert a["dict_len_lut"](body(vals)).tolist() == [1, 1, 1, 1, 1, 3]
    assert a["dict_len_lut"](body(vals)).tolist() == _staged_len(vals).tolist()
    # pc.utf8_length would answer bytes; pin that it is NOT what we return
    assert a["dict_len_lut"](body(vals)).tolist() != [4, 3, 2, 1, 4, 3]


@pytest.mark.fast
def test_len_lut_empty_value_does_not_shift_the_following_value(kernel):
    """np.add.reduceat answers an EMPTY segment with the first byte of the
    NEXT value, so a naive continuation count is charged to the value after
    the hole. Pinned at every position around a multibyte value."""
    a = kernel.alias
    base = ["a", "", "\U0001f600\U0001f600", "", "z", "b"]
    for hole in range(len(base) + 1):
        vals = base[:hole] + [""] + base[hole:]
        assert a["dict_len_lut"](body(vals)).tolist() == _staged_len(vals).tolist()
    for hole in range(5):
        vals = ["", "", "", "", ""]
        vals[hole] = "\U0001f600а你"
        assert a["dict_len_lut"](body(vals)).tolist() == _staged_len(vals).tolist()


@pytest.mark.fast
def test_len_lut_value_bigger_than_the_chunk(kernel):
    """The body is walked in _LEN_LANE_CHUNK slices; a single value larger
    than one slice must still be counted exactly (it becomes its own slice)."""
    a = kernel.alias
    ch = _chunk()
    for mult in (1, 2, 3):
        n = ch * mult
        vals = ["L" * n, "\U0001f600" * (n // 4), "a", ""]
        assert a["dict_len_lut"](body(vals)).tolist() == _staged_len(vals).tolist()
    # a value spanning a slice boundary next to an empty one
    vals = ["", "L" * (ch + 1), "", "а" * (ch // 2), ""]
    assert a["dict_len_lut"](body(vals)).tolist() == _staged_len(vals).tolist()


@pytest.mark.fast
def test_malformed_body_offsets_fall_back_instead_of_raising(kernel):
    """Any doubt about the offsets -> the verbatim <U lane, which owns both
    the answer and the error. The expectation is the staged lane over exactly
    the values those offsets describe, which is what _body_to_list decodes."""
    a = kernel.alias

    def staged_for(data, offs):
        mv = memoryview(data)
        vals = [bytes(mv[int(x):int(y)]).decode("utf-8")
                for x, y in zip(offs[:-1], offs[1:])]
        return _staged_len(vals)

    good = body(["aa", "bb", "cc"])
    for offs in ([0, 2, 4, 5], [1, 3, 5, 7], [2, 4, 6, 8], [0, 2, 4, 4]):
        o = np.asarray(offs, np.int32)
        bad = {"utf8_data": good["utf8_data"], "offsets": o}
        try:
            got = a["dict_len_lut"](bad)
        except (ValueError, IndexError):
            continue                       # the verbatim lane rejected it
        assert got.tolist() == staged_for(good["utf8_data"], o).tolist(), offs
    # an offsets entry that is not an array at all: the verbatim lane owns the
    # error, and it is the same error the pre-M7c code raised (TypeError from
    # np.asarray(None) inside _body_to_list)
    with pytest.raises(TypeError):
        a["dict_len_lut"]({"utf8_data": b"aab", "offsets": None})


@pytest.mark.fast
def test_nul_tail_body_falls_back_to_the_padded_key(kernel):
    a = kernel.alias
    for s in NUL_ALPHABET:
        vals = [s, "a", "m" + s, "b"]
        b = body(vals)
        assert a["dict_len_lut"](b).tolist() == _staged_len(vals).tolist(), repr(s)
        assert a["dict_min_max"](b) == _staged_min_max(vals), repr(s)
        for p in NUL_ALPHABET:
            assert a["dict_startswith"](b, p).tolist() \
                == _staged_startswith(vals, p).tolist(), (repr(s), repr(p))


@pytest.mark.fast
def test_startswith_needle_with_a_trailing_nul_falls_back(kernel):
    """A needle that ends in NUL is truncated by the <U compare, so the raw
    byte answer is NOT the staged answer. The gate is on the needle, so a
    NUL-free body must still fall back."""
    a = kernel.alias
    vals = ["a", "ab", "a" + NUL + "b", "abc", ""]
    b = body(vals)
    for p in NUL_ALPHABET:
        assert a["dict_startswith"](b, p).tolist() \
            == _staged_startswith(vals, p).tolist(), repr(p)
    # a NUL-free needle over a NUL-carrying body: the value's own trailing
    # NULs must not matter, which is what the gate proves
    for p in ("", "a", "ab", "abc", "zzz"):
        assert a["dict_startswith"](b, p).tolist() \
            == _staged_startswith(vals, p).tolist(), repr(p)


@pytest.mark.fast
def test_min_max_body_lane_keeps_first_min_and_last_max(kernel):
    """Duplicates are the only place the tie-break can move: the lane must
    return the FIRST minimum and the LAST maximum of the <U order."""
    a = kernel.alias
    cases = [
        ["b", "a", "b", "a", "b"],
        ["zz", "zz", "aa", "zz"],
        ["", "a", "", "b"],
        ["\U0001f600", "а", "\U0001f600", "z"],
        ["m", "m" + NUL, "m" + NUL + NUL, "a"],
        ["only"],
    ]
    for vals in cases:
        assert a["dict_min_max"](body(vals)) == _staged_min_max(vals), vals
        assert a["dict_min_max"](list(vals)) == _staged_min_max(vals), vals


@pytest.mark.fast
def test_body_lane_is_exact_over_a_multibyte_fuzz(kernel):
    """Random bodies, both carriers, every op, against the staged lane."""
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    alpha = ["a", "b", "Z", "а", "你", "\U0001f600", " ", NUL]
    checked = 0
    for _ in range(1000):
        d = int(rng.integers(0, 30))
        vals = ["".join(alpha[int(j)] for j in rng.integers(0, len(alpha),
                                                           size=int(rng.integers(0, 10))))
                for _ in range(d)]
        b = body(vals)
        for carrier in (b, list(vals)):
            assert a["dict_len_lut"](carrier).tolist() == _staged_len(vals).tolist()
            assert a["dict_min_max"](carrier) == _staged_min_max(vals)
            assert a["dict_ordering"](carrier)["order"].tolist() \
                == _staged_order(vals).tolist()
            for p in ("", "a", "ab", "а", "\U0001f600", NUL, "a" + NUL):
                assert a["dict_startswith"](carrier, p).tolist() \
                    == _staged_startswith(vals, p).tolist()
        checked += d
    assert checked > 3000


@pytest.mark.fast
def test_invalid_utf8_body_is_answered_byte_level(kernel):
    """DELTA, pinned on purpose. The verbatim lane decoded the body and
    raised "not valid UTF-8"; a zero-copy lane cannot decode, so it answers
    the byte-level question -- exactly what dict_contains has always done
    (see its docstring). Engine bodies are valid by construction; a
    hand-crafted invalid buffer now gets a number instead of an error."""
    a = kernel.alias
    bad = {"utf8_data": b"\xff\xfe", "offsets": np.asarray([0, 2], np.int32)}
    assert a["dict_len_lut"](bad).tolist() == [2]
    assert a["dict_startswith"](bad, "a").tolist() == []
    with pytest.raises(ValueError, match="not valid UTF-8"):
        a["dict_min_max"](bad)
