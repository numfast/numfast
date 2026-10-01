# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Gate: the generic TEXT primitives -- TextOps.text_length (bit-identity
over every accepted carrier), dict_encode_arrow (contract + bit-identity
against dictionary_encode) and regex_replace_dict (exactness against a
per-row re.sub oracle).

Every oracle below is independent pure Python: no numfast import inside an
oracle function. Seed 42 where RNG is used (none here).
"""

from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])

# None, "", ascii, cyrillic, 2-byte + 4-byte emoji, astral math letter,
# embedded NUL, BOM, latin-1 supplement, turkish/digraph, ligature,
# 300-char row, and a mixed row.
EDGE = [None, "", "a", "abc", "привет", "\U0001f600", "\U0001d54f",
        "a" * 300, "тест\U0001f600тест", "\x01", "﻿", "ß", "ǅ",
        "ﬀ", "0", "00", "line1\nline2"]
EDGE_NO_NULL = [v if v is not None else "" for v in EDGE]


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


def oracle_length(values):
    """Independent oracle: code-point length, None -> 0."""
    return np.asarray([len(v) if isinstance(v, str) else 0 for v in values],
                      dtype=np.int32)


def oracle_regex(values, pattern, repl):
    """Independent oracle: DuckDB regexp_replace parity, first match only."""
    import re

    rx = re.compile(pattern)
    return [rx.sub(repl, v, count=1) if isinstance(v, str) else None
            for v in values]


def _carriers(kernel, values):
    """Every carrier shape the primitive documents, one Array each."""
    import pyarrow as pa

    arr = pa.array(values, type=pa.string())
    out = {
        "list": values,
        "object_nd": np.asarray(values, dtype=object),
        "U_nd": np.asarray([v if v is not None else "" for v in values], dtype="U"),
        "pa_array": arr,
        "pa_chunked": pa.chunked_array([arr.slice(0, 4), arr.slice(4)]),
        "tuple": tuple(values),
    }
    return out


# ======================= TextOps.text_length =======================

@pytest.mark.fast
def test_text_length_every_carrier_is_bit_identical(kernel):
    a = kernel.alias
    want = oracle_length(EDGE)
    for name, col in _carriers(kernel, EDGE).items():
        got = a["text_length"](col)
        assert got.dtype == np.int32, name
        assert np.array_equal(got, want), name


@pytest.mark.fast
def test_text_length_is_code_points_not_bytes(kernel):
    a = kernel.alias
    # 2-byte cyrillic, 4-byte emoji, astral char: each counts ONE code point.
    col = ["привет", "\U0001f600", "\U0001d54f", "é", "é"]
    got = a["text_length"](col)
    assert got.tolist() == [6, 1, 1, 1, 1]
    assert oracle_length(col).tolist() == got.tolist()


@pytest.mark.fast
def test_text_length_none_and_empty_and_long(kernel):
    a = kernel.alias
    got = a["text_length"]([None, "", "a" * 300])
    assert got.tolist() == [0, 0, 300]
    assert np.array_equal(got, oracle_length([None, "", "a" * 300]))


@pytest.mark.fast
def test_text_length_large_random_column_matches_oracle(kernel):
    a = kernel.alias
    rs = np.random.RandomState(42)
    words = ["привет", "\U0001f600", "abc", "", "\U0001d54f", "x" * 40]
    col = [words[i] if (v := rs.randint(0, 2)) else None
           for i in rs.randint(0, len(words), size=5000)]
    got = a["text_length"](col)
    assert np.array_equal(got, oracle_length(col))


@pytest.mark.fast
def test_text_length_rejects_dict_carrier(kernel):
    a = kernel.alias
    with pytest.raises(TypeError):
        a["text_length"]({"values": ["a", "b"]})


# ======================= dict_encode_arrow =======================

@pytest.mark.fast
def test_dict_encode_arrow_matches_dictionary_encode_on_every_carrier(kernel):
    a = kernel.alias
    # pa_chunked is skipped here: the frozen dictionary_encode rejects it.
    # It is pinned against an independent oracle further down.
    carriers = _carriers(kernel, EDGE)
    carriers.pop("pa_chunked")
    for name, col in carriers.items():
        base = a["dictionary_encode"](col)
        cand = a["dict_encode_arrow"](col, None, "sorted")
        assert cand["values"] == base["values"], name
        assert np.array_equal(np.asarray(cand["codes"], np.int32),
                              np.asarray(base["codes"], np.int32)), name
        assert cand["metadata"]["d"] == base["metadata"]["d"], name
        assert cand["metadata"]["nulls"] == base["metadata"]["nulls"], name
        assert cand["dictionary"].utf8_data == base["dictionary"].utf8_data, name
        assert np.array_equal(cand["dictionary"].offsets,
                              base["dictionary"].offsets), name


@pytest.mark.fast
def test_dict_encode_arrow_pa_chunked_matches_an_independent_oracle(kernel):
    import pyarrow as pa

    a = kernel.alias
    arr = pa.array(EDGE, type=pa.string())
    cand = a["dict_encode_arrow"](
        pa.chunked_array([arr.slice(0, 4), arr.slice(4)]), None, "sorted")
    uniq = sorted({v for v in EDGE if v is not None})
    want_codes = np.asarray(
        [uniq.index(v) if v is not None else 0 for v in EDGE], dtype=np.int32)
    assert cand["values"] == uniq
    assert np.array_equal(np.asarray(cand["codes"], np.int32), want_codes)
    assert cand["metadata"]["nulls"] == 1


@pytest.mark.fast
def test_dict_encode_arrow_sorted_contract_holds(kernel):
    """code == lexicographic rank: min(codes) is the min value."""
    a = kernel.alias
    col = ["b", None, "a", "", "c", "a", "aa"]
    enc = a["dict_encode_arrow"](col, None, "sorted")
    assert enc["values"] == sorted(set(v for v in col if v is not None))
    # a valid row's value is recoverable from its code; an invalid row
    # carries the code-0 placeholder and is read through `validity`
    valid = np.asarray(enc["validity"], bool)
    got = np.asarray(enc["values"], object)[np.asarray(enc["codes"], np.int32)]
    for i, v in enumerate(col):
        if v is None:
            assert not bool(valid[i]) and int(enc["codes"][i]) == 0
        else:
            assert got[i] == v and bool(valid[i])
    # code order == value order
    for i in range(len(enc["values"])):
        for j in range(len(enc["values"])):
            if i != j:
                assert (enc["values"][i] < enc["values"][j]) == (i < j)


@pytest.mark.fast
def test_dict_encode_arrow_first_seen_is_a_row_bijection(kernel):
    a = kernel.alias
    rs = np.random.RandomState(42)
    words = ["bb", "a", "ccc", "", "\U0001f600", "aa"]
    col = [words[i] if rs.rand() > 0.2 else None
           for i in rs.randint(0, len(words), size=4000)]
    srt = a["dict_encode_arrow"](col, None, "sorted")
    fs = a["dict_encode_arrow"](col, None, "first_seen")
    assert sorted(fs["values"]) == srt["values"]
    assert fs["metadata"]["sorted"] is False
    valid = np.asarray(fs["validity"], bool)
    ds = np.asarray(srt["values"], object)[np.asarray(srt["codes"], np.int32)]
    df = np.asarray(fs["values"], object)[np.asarray(fs["codes"], np.int32)]
    assert np.array_equal(ds[valid], df[valid])


@pytest.mark.fast
def test_dict_encode_arrow_validity_sidecar(kernel):
    a = kernel.alias
    col = ["a", "b", "c"]
    v = np.array([True, False, True])
    base = a["dictionary_encode"](col, v)
    cand = a["dict_encode_arrow"](col, v, "sorted")
    assert np.array_equal(np.asarray(cand["codes"], np.int32),
                          np.asarray(base["codes"], np.int32))
    assert np.array_equal(np.asarray(cand["validity"], bool), v)
    assert cand["metadata"]["nulls"] == 1
    with pytest.raises(ValueError):
        a["dict_encode_arrow"](col, np.array([True]), "sorted")


@pytest.mark.fast
def test_dict_encode_arrow_degenerate_columns(kernel):
    a = kernel.alias
    empty = a["dict_encode_arrow"]([], None, "sorted")
    assert empty["metadata"]["d"] == 0 and empty["values"] == []
    allnull = a["dict_encode_arrow"]([None, None], None, "sorted")
    assert allnull["metadata"]["d"] == 0
    assert np.array_equal(np.asarray(allnull["validity"], bool), [False, False])
    allempty = a["dict_encode_arrow"](["", "", ""], None, "sorted")
    assert allempty["values"] == [""] and allempty["metadata"]["d"] == 1


@pytest.mark.fast
def test_dict_encode_arrow_rejects_non_text_carriers(kernel):
    a = kernel.alias
    for bad in ([1, 2, 3], [1, None, 3], np.asarray([1.0, 2.0]),
                np.asarray([1, 2], dtype=np.int64), [b"x", b"y"], [b"x", "y"],
                {"values": ["a"]}, np.asarray([["a"], ["b"]]), "abc",
                ["a", object()]):
        with pytest.raises(ValueError):
            a["dict_encode_arrow"](bad, None, "sorted")
    with pytest.raises(ValueError):
        a["dict_encode_arrow"](["a"], None, "nonsense")


@pytest.mark.fast
def test_dict_encode_arrow_metadata_view(kernel):
    a = kernel.alias
    enc = a["dict_encode_arrow"](["b", "a", None], None, "sorted")
    md = a["dict_encode_arrow_metadata"](enc)
    assert md["d"] == 2 and md["sorted"] is True
    assert md["encoding"] == "dictionary-sorted-v1"
    md2 = a["dict_encode_arrow_metadata"](
        a["dict_encode_arrow"](["b", "a", None], None, "first_seen"))
    assert md2["sorted"] is False


@pytest.mark.fast
def test_dict_encode_arrow_large_column_matches_dictionary_encode(kernel):
    a = kernel.alias
    rs = np.random.RandomState(42)
    words = ["w%04d" % i for i in range(500)]
    col = [words[i] if rs.rand() > 0.1 else None
           for i in rs.randint(0, len(words), size=20000)]
    base = a["dictionary_encode"](col)
    cand = a["dict_encode_arrow"](col, None, "sorted")
    assert cand["values"] == base["values"]
    assert np.array_equal(np.asarray(cand["codes"], np.int32),
                          np.asarray(base["codes"], np.int32))
    assert cand["dictionary"].utf8_data == base["dictionary"].utf8_data


# ======================= regex_replace_dict =======================

@pytest.mark.fast
def test_regex_replace_dict_is_exact_vs_per_row_oracle(kernel):
    a = kernel.alias
    cases = [
        (["a", None, "", "bb"], "b", "X"),
        (["abc", None, ""], "zzz", "Q"),
        (["abc", None, ""], "", "-"),
        (["https://a.b/c", None, ""], r"^(https?)://", r"\1X"),
        (["привет", "\U0001f600", None, "\U0001d54f"], "и", "И"),
        (["a\n", "b\n", None], r"b$", "Z"),
        (["a\nb", "x\n", None], "$", "!"),
        (["ab", "a", "ba", None], "b", ""),
        ([None, None], "a", "B"),
        (["", "", None], "^$", "E"),
        (["abc", None], "(z)?x", r"[\1]"),
        (["a\nb", "c\nd", None], r"^(\w)", r"[\1]"),
        (["aaa"], "a", "X"),
    ]
    for col, pat, rep in cases:
        enc = a["dictionary_encode"](col)
        got = a["regex_replace_dict"](enc["codes"], enc["values"],
                                      enc["validity"], pat, rep)
        want = oracle_regex(col, pat, rep)
        assert got["arrow"].to_pylist() == want, (col, pat, rep)


@pytest.mark.fast
def test_regex_replace_dict_codes_decode_to_the_same_rows(kernel):
    a = kernel.alias
    col = ["aa", "ab", "ba", None, "bb", "aa"]
    pat, rep = "a", "Z"
    enc = a["dictionary_encode"](col)
    got = a["regex_replace_dict"](enc["codes"], enc["values"], enc["validity"],
                                  pat, rep)
    valid = np.asarray(got["validity"], bool)
    dec = np.asarray(got["values"], object)[np.asarray(got["codes"], np.int32)]
    want = oracle_regex(col, pat, rep)
    for i, v in enumerate(col):
        if v is None:
            assert not bool(valid[i]) and int(got["codes"][i]) == 0
            assert want[i] is None
        else:
            assert dec[i] == want[i], (col, i)
            assert bool(valid[i])


@pytest.mark.fast
def test_regex_replace_dict_output_dictionary_is_sorted_and_deduped(kernel):
    a = kernel.alias
    col = ["ab", "a", "ba", "aa", None]
    enc = a["dictionary_encode"](col)
    got = a["regex_replace_dict"](enc["codes"], enc["values"], enc["validity"],
                                  "b", "")
    assert got["values"] == sorted(got["values"])
    assert len(got["values"]) == len(set(got["values"]))
    assert got["metadata"]["sorted"] is True
    # a regex can collapse several distinct values onto one output
    one = a["regex_replace_dict"](enc["codes"], enc["values"], enc["validity"],
                                  "a", "")
    assert one["values"] == sorted({oracle_regex([v], "a", "")[0]
                                    for v in enc["values"]})


@pytest.mark.fast
def test_regex_replace_dict_first_match_only(kernel):
    a = kernel.alias
    col = ["aaaa", "abab", None]
    enc = a["dictionary_encode"](col)
    got = a["regex_replace_dict"](enc["codes"], enc["values"], enc["validity"],
                                  "a", "X")
    assert got["arrow"].to_pylist() == oracle_regex(col, "a", "X")
    assert got["arrow"].to_pylist() == ["Xaaa", "Xbab", None]


@pytest.mark.fast
def test_regex_replace_dict_dollar_is_python_dollar(kernel):
    """Python `$` matches before a trailing newline; RE2's does not."""
    a = kernel.alias
    col = ["a\n", "b\n", "a", None]
    enc = a["dictionary_encode"](col)
    got = a["regex_replace_dict"](enc["codes"], enc["values"], enc["validity"],
                                  "$", "!")
    assert got["arrow"].to_pylist() == ["a!\n", "b!\n", "a!", None]


@pytest.mark.fast
def test_regex_replace_dict_large_dictionary_column(kernel):
    a = kernel.alias
    rs = np.random.RandomState(42)
    words = ["h%03d.example.com" % i for i in range(400)]
    col = ["https://" + words[i] + "/p" if rs.rand() > 0.1 else None
           for i in rs.randint(0, len(words), size=20000)]
    enc = a["dictionary_encode"](col)
    got = a["regex_replace_dict"](enc["codes"], enc["values"], enc["validity"],
                                  r"^https?://(?:www\.)?([^/]+)/.*$", r"\1")
    assert got["arrow"].to_pylist() == oracle_regex(
        col, r"^https?://(?:www\.)?([^/]+)/.*$", r"\1")
    assert got["metadata"]["d"] == 400


@pytest.mark.fast
def test_regex_replace_dict_errors_are_explicit(kernel):
    a = kernel.alias
    enc = a["dictionary_encode"](["a", "b"])
    with pytest.raises(ValueError):
        a["regex_replace_dict"](enc["codes"], enc["values"], enc["validity"],
                                "(unclosed", "x")
    with pytest.raises(ValueError):
        a["regex_replace_dict"](enc["codes"], enc["values"], enc["validity"],
                                5, "x")
    with pytest.raises(ValueError):
        a["regex_replace_dict"](enc["codes"], enc["values"], enc["validity"],
                                "a", 5)
    with pytest.raises(ValueError):
        a["regex_replace_dict"](np.array([0, 9], np.int32), enc["values"],
                                None, "a", "b")
    with pytest.raises(ValueError):
        a["regex_replace_dict"](enc["codes"], enc["values"], np.array([True]),
                                "a", "b")
    with pytest.raises(ValueError):
        a["regex_replace_dict"](np.array([0, 1], np.int32), [1, 2], None, "a", "b")


@pytest.mark.fast
def test_regex_replace_dict_accepts_body_and_array_carriers(kernel):
    a = kernel.alias
    col = ["aa", "ab", None]
    want = oracle_regex(col, "a", "Z")
    enc = a["dictionary_encode"](col)
    import pyarrow as pa

    carriers = {
        "values_list": enc["values"],
        "envelope": enc,
        "body": enc["dictionary"],
        "pa_array": pa.array(enc["values"], type=pa.string()),
        "U_nd": np.asarray(enc["values"], dtype="U"),
    }
    for name, vals in carriers.items():
        got = a["regex_replace_dict"](enc["codes"], vals, enc["validity"],
                                      "a", "Z")
        assert got["arrow"].to_pylist() == want, name