# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Stage 3 gate: dictionary-domain ops (equality/<> ''/LIKE/length/ordering/MIN-MAX).

String work on values[D] only; row-side numeric: allowed codes -> member
mask -> existing filter. Lexicographic guarantee proven on unsorted dicts.
ClickBench hits_1m queries exact vs pandas (SQL 3VL for NULLs).
"""

import random
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])
HITS = Path("C:/App/competitions/ClickBench/data/hits_1m.parquet")
SEED = 42


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


def _filter_count(a, codes, allowed, validity=None):
    """Row-side composition: allowed codes -> mask -> bool series -> filter."""
    mask = a["codes_member_mask"](codes, allowed, validity)
    bufs = a["cpu_execute"]([a["ir_series"]("c", np.ascontiguousarray(codes), dtype="int32"),
                             a["ir_series"]("m", np.ascontiguousarray(mask), dtype="bool"),
                             a["ir_filter"]("f", "c", "m")])
    return int(bufs["f"].size)


# ---------- unit ----------

@pytest.mark.fast
def test_lookup_and_equality(kernel):
    a = kernel.alias
    vals = ["b", "a", "c"]
    assert a["dict_lookup"](vals, "a") == 1
    assert a["dict_lookup"](vals, "zz") is None
    assert list(a["dict_equal_codes"](vals, "a")) == [1]
    assert list(a["dict_equal_codes"](vals, "zz")) == []
    with pytest.raises(ValueError, match="must be str"):
        a["dict_lookup"](vals, 5)


@pytest.mark.fast
def test_not_empty_contains_prefix(kernel):
    a = kernel.alias
    vals = ["", "foo", "foobar", "bar"]
    assert list(a["dict_not_empty_codes"](vals)) == [1, 2, 3]
    assert list(a["dict_not_empty_codes"](["a"])) == [0]
    assert list(a["dict_contains"](vals, "foo")) == [1, 2]
    assert list(a["dict_contains"](vals, "")) == [0, 1, 2, 3]
    assert list(a["dict_contains"](vals, "zz")) == []
    assert list(a["dict_startswith"](vals, "foo")) == [1, 2]
    assert list(a["dict_startswith"](vals, "")) == [0, 1, 2, 3]


@pytest.mark.fast
def test_len_lut_and_member_mask_gates(kernel):
    a = kernel.alias
    assert list(a["dict_len_lut"](["", "ab", "x"])) == [0, 2, 1]
    assert list(a["dict_len_lut"]([])) == []
    m = a["codes_member_mask"](np.array([0, 1, 0, 2], dtype=np.int32),
                               np.array([0, 2], dtype=np.int32),
                               np.array([True, True, False, True]))
    assert list(np.asarray(m, dtype=bool)) == [True, False, False, True]
    with pytest.raises(ValueError, match="integer codes"):
        a["codes_member_mask"](np.array([0.5]), np.array([0]))
    with pytest.raises(ValueError, match="validity size"):
        a["codes_member_mask"](np.array([0], dtype=np.int32), np.array([0]),
                               np.array([True, True]))


@pytest.mark.fast
def test_ordering_generic_unsorted_and_min_max(kernel):
    a = kernel.alias
    vals = ["pear", "apple", "fig", "ёлка", "apple2"]
    o = a["dict_ordering"](vals)
    assert [vals[i] for i in o["order"].tolist()] == sorted(vals)
    inv = np.empty(len(vals), dtype=np.int32)
    inv[o["order"]] = np.arange(len(vals), dtype=np.int32)
    assert list(inv) == list(o["rank_of"])
    assert a["dict_min_max"](vals) == (min(vals), max(vals))
    assert a["dict_min_max"]([]) == (None, None)
    e = a["dict_ordering"]([])
    assert e["order"].size == 0 and e["rank_of"].size == 0


@pytest.mark.fast
def test_lexicographic_guarantee_random(kernel):
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    abc = ["a", "Z", "ё", " ", "~", "0", "-", "ü", "Я"]
    vals = ["".join(rng.choice(abc, size=int(rng.integers(1, 8)))) for _ in range(5_000)]
    vals = sorted(set(vals))
    o = a["dict_ordering"](vals)
    assert [vals[i] for i in o["order"].tolist()] == sorted(vals)
    assert [vals[i] for i in o["order"][:1]] == [min(vals)]
    assert a["dict_min_max"](vals) == (min(vals), max(vals))


# ---------- differential (seed 42, 50K rows, NULLs) ----------

@pytest.mark.fast
def test_differential_predicates_vs_row_scan(kernel):
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    pool = [f"site-{i:04d}.com/pg" for i in range(2_000)] + ["", "google-x"]
    vals = [None if rng.random() < 0.08 else pool[int(rng.integers(0, len(pool)))]
            for _ in range(50_000)]
    t0 = time.perf_counter()
    res = a["resident_prepare"]({"t": {"values": vals}})["t"]
    t_enc = (time.perf_counter() - t0) * 1000
    codes, dv, vv = res["codes"], res["dictionary"], res["validity"]
    notnull = [(v is not None) for v in vals]
    # equality
    n_eq = _filter_count(a, codes, a["dict_equal_codes"](dv, "google-x"), vv)
    assert n_eq == sum(1 for v in vals if v == "google-x")
    # LIKE contains
    n_like = _filter_count(a, codes, a["dict_contains"](dv, "site-00"), vv)
    assert n_like == sum(1 for v in vals if v is not None and "site-00" in v)
    # prefix
    n_pre = _filter_count(a, codes, a["dict_startswith"](dv, "site-000"), vv)
    assert n_pre == sum(1 for v in vals if v is not None and v.startswith("site-000"))
    # <> ''
    n_ne = _filter_count(a, codes, a["dict_not_empty_codes"](dv), vv)
    assert n_ne == sum(1 for v in vals if v is not None and v != "")
    # length via LUT + gather + compare (existing nodes only)
    lut = a["dict_len_lut"](dv)
    bufs = a["cpu_execute"]([a["ir_series"]("lut", lut, dtype="int32"),
                             a["ir_series"]("c", codes, dtype="int32"),
                             a["ir_gather"]("rl", "lut", "c"),
                             a["ir_compare"]("m", "rl", 15, op=">")])
    n_len = int(np.count_nonzero(bufs["m"].to_array()
                                 & np.asarray(vv if vv is not None else True, dtype=bool)))
    assert n_len == sum(1 for v in vals if v is not None and len(v) > 15)
    # MIN/MAX + ordering boundary
    assert a["dict_min_max"](dv) == (min(v for v in vals if v is not None),
                                     max(v for v in vals if v is not None))
    print(f"\nstages ms: encode-50K={t_enc:.1f} eq={n_eq} like={n_like} "
          f"pre={n_pre} ne={n_ne} len={n_len}")


# ---------- ClickBench hits_1m ----------

def _cb_col(col):
    import pandas as pd

    df = pd.read_parquet(HITS, columns=[col])
    return [None if (v is None or (isinstance(v, float) and v != v)) else str(v)
            for v in df[col].tolist()]


@pytest.mark.fast
def test_clickbench_url_like_and_minmax_exact(kernel):
    if not HITS.exists():
        pytest.skip("hits_1m.parquet absent")
    a = kernel.alias
    url = _cb_col("URL")
    t0 = time.perf_counter()
    res = a["resident_prepare"]({"url": {"values": url}})["url"]
    t_enc = (time.perf_counter() - t0) * 1000
    codes, dv = res["codes"], res["dictionary"]
    t0 = time.perf_counter()
    allowed = a["dict_contains"](dv, "google")
    n = _filter_count(a, codes, allowed, res["validity"])
    t_q = (time.perf_counter() - t0) * 1000
    want = sum(1 for v in url if v is not None and "google" in v)
    assert n == want, f"LIKE: {n} != pandas {want}"
    mn, mx = a["dict_min_max"](dv)
    assert (mn, mx) == (min(v for v in url if v is not None),
                        max(v for v in url if v is not None))
    print(f"\nstages ms: encode={t_enc:.1f} LIKE-dict+member={t_q:.1f} "
          f"D={len(dv)} hits={n} min={mn!r} max={mx!r}")


@pytest.mark.fast
def test_clickbench_order_length_notempty_exact(kernel):
    if not HITS.exists():
        pytest.skip("hits_1m.parquet absent")
    a = kernel.alias
    import pandas as pd

    col = _cb_col("Referer")
    res = a["resident_prepare"]({"r": {"values": col}})["r"]
    codes, dv, vv = res["codes"], res["dictionary"], res["validity"]
    # ORDER BY via rank LUT + gather (numeric), head/tail 100 vs pandas
    o = a["dict_ordering"](dv)
    bufs = a["cpu_execute"]([a["ir_series"]("rank", o["rank_of"], dtype="int32"),
                             a["ir_series"]("c", codes, dtype="int32"),
                             a["ir_gather"]("rr", "rank", "c")])
    rr = np.asarray(bufs["rr"])
    s = pd.Series(col)
    order = np.argsort(rr, kind="stable")
    valid_pos = np.nonzero(np.asarray(vv, dtype=bool))[0] if vv is not None else np.arange(len(col))
    # head/tail over valid rows in rank order
    vord = order[np.isin(order, valid_pos)]
    head = [dv[int(codes[i])] for i in vord[:100]]
    tail = [dv[int(codes[i])] for i in vord[-100:]]
    want_sorted = s.dropna().sort_values(kind="stable").tolist()
    assert head == want_sorted[:100], "ORDER BY head mismatch"
    assert tail == want_sorted[-100:], "ORDER BY tail mismatch"
    # length + <> '' (3VL)
    lut = a["dict_len_lut"](dv)
    bufs2 = a["cpu_execute"]([a["ir_series"]("lut", lut, dtype="int32"),
                              a["ir_series"]("c2", codes, dtype="int32"),
                              a["ir_gather"]("rl", "lut", "c2"),
                              a["ir_compare"]("m", "rl", 60, op=">")])
    eff = np.asarray(vv, dtype=bool) if vv is not None else np.ones(len(col), dtype=bool)
    n_len = int(np.count_nonzero(bufs2["m"].to_array() & eff))
    assert n_len == sum(1 for v in col if v is not None and len(v) > 60)
    n_ne = _filter_count(a, codes, a["dict_not_empty_codes"](dv), vv)
    assert n_ne == sum(1 for v in col if v is not None and v != "")
    print(f"\nReferer ORDER-BY head/tail exact, len>60={n_len} ne={n_ne} D={len(dv)}")


# ==========================================================================
# DomainLUT: native-body carrier lane, all-true LUT identity, body byte scan
# ==========================================================================
# Three contracts, pinned here. Nothing below knows a column, a table or a
# query; the needles are test data, and every fixture is run through the
# [str] carrier, the native UTF-8 body, and both forced substring kernels.

MULTIBYTE = "éПривет😀"   # 2-, 2-, 4-byte
NUL = "\x00"


def native_body(values):
    """The ENC dictionary shape: one UTF-8 body + int32 offsets."""
    offs = np.zeros(len(values) + 1, dtype=np.int32)
    offs[1:] = np.cumsum([len(v.encode("utf-8")) for v in values])
    return {"dictionary": {"utf8_data": "".join(values).encode("utf-8"),
                           "offsets": offs},
            "values": list(values)}


def reference(values, needle):
    return np.asarray(pc.match_substring(pa.array(values, type=pa.string()),
                                         needle).to_numpy(zero_copy_only=False),
                      dtype=bool)


# ---------- (1) the carrier lane ----------

@pytest.mark.fast
def test_native_body_lane_equals_list_lane(kernel):
    rng = random.Random(SEED)
    values = sorted({"".join(rng.choice("abcde" + MULTIBYTE + NUL) for _ in range(rng.randint(0, 9)))
                     for _ in range(300)})
    enc = kernel.alias["dictionary_encode"](list(values))
    lut_body = kernel.alias["dict_contains_lut"](enc, "abc")
    lut_list = kernel.alias["dict_contains_lut"](enc["values"], "abc")
    assert np.array_equal(lut_body, reference(values, "abc"))
    assert np.array_equal(lut_list, lut_body)
    # the ENC carrier is the zero-copy lane: no [str] transcode can be paid,
    # and both lanes give the same table over the same dictionary
    ne_ref = np.asarray(pc.not_equal(pa.array(enc["values"], type=pa.string()), "")
                        .to_numpy(zero_copy_only=False), dtype=bool)
    assert np.array_equal(kernel.alias["dict_not_equal_lut"](enc, ""), ne_ref)
    assert np.array_equal(kernel.alias["dict_not_equal_lut"](enc["values"], ""), ne_ref)
    assert np.array_equal(kernel.alias["dict_not_equal_lut"](enc, "abc"),
                          kernel.alias["dict_not_equal_lut"](enc["values"], "abc"))


# ---------- (2) the all-true LUT identity ----------

@pytest.mark.fast
def test_all_true_lut_skips_the_gather_and_stays_exact(kernel):
    rng = random.Random(SEED)
    clm = kernel.alias["codes_lut_mask"]
    n, d = 5000, 400
    c0 = [rng.randrange(d) for _ in range(n)]
    c1 = [rng.randrange(d) for _ in range(n)]
    v0 = np.array([rng.random() > 0.1 for _ in range(n)])
    v1 = np.array([rng.random() > 0.3 for _ in range(n)])
    ones = np.ones(d, bool)
    zeros = np.zeros(d, bool)
    one_off = np.ones(d, bool)
    one_off[d // 2] = False

    def ref(luts, vals):
        g = [np.ones(n, bool) if v is None else v for v in vals]
        return luts[0][c0] & luts[1][c1] & g[0] & g[1]

    for luts, vals, label in (
            ([ones, ones], [v0, v1], "both all-true"),
            ([ones, zeros], [v0, v1], "second all-false"),
            ([one_off, ones], [v0, v1], "first has one rejecting code"),
            ([ones, ones], [v0, None], "no second gate"),
            ([ones, ones], [None, None], "no gates"),
            ([one_off, zeros], [v0, None], "one rejecting code, one dead"),
    ):
        got = clm([c0, c1], luts, vals)
        assert np.array_equal(got, ref(luts, vals)), label
    # the all-true table really does leave the mask to the validity gates
    assert np.array_equal(clm([c0, c1], [ones, ones], None), np.ones(n, bool))
    assert np.array_equal(clm([c0, c1], [ones, ones], [v0, None]), v0)
    # ... and a single rejecting code rejects exactly its own rows, while the
    # second all-true table contributes nothing
    m = clm([c0, c1], [one_off, ones], None)
    assert np.array_equal(m, one_off[c0])
    assert not m[c0 == d // 2].any()


@pytest.mark.fast
def test_all_true_lut_never_weakens_the_empty_dictionary_rule(kernel):
    d = 16
    codes = list(range(d))
    assert not kernel.alias["codes_lut_mask"]([codes], [np.zeros(0, bool)], None).any()


# ---------- (3) the body byte scan ----------

EDGE_CASES = [
    ("empty values and an absent needle", ["", "a"], "b"),
    ("needle longer than every haystack", ["", "a"], "abcdef"),
    ("needle exactly the haystack", ["abc"], "abc"),
    ("repeated values around the match", ["ab", "ab", "ab"], "ab"),
    ("needle overhanging the last value", ["ab", "cd"], "cd"),
    ("boundary-spanning: tail + head", ["ab", "cd"], "bc"),
    ("boundary-spanning: full pair", ["ab", "cd"], "abcd"),
    ("boundary-spanning over three", ["ab", "cd", "ef"], "bcde"),
    ("boundary-spanning repeated tail", ["xaby", "xaby"], "abyx"),
    ("two-byte multi-byte needle", ["Привет", "Привет"], "иве"),
    ("multi-byte needle across a boundary", ["Прив", "ет"], "иве"),
    ("emoji needle", ["a\U0001F600b", "c"], "\U0001F600"),
    ("emoji as one whole value", ["\U0001F600"], "\U0001F600"),
    ("embedded NUL in the value", ["a" + NUL + "b", "a"], NUL),
    ("embedded NUL in the needle", ["a" + NUL + "b", "a"], "a" + NUL + "b"),
    ("NUL-only value", [NUL, "a"], NUL),
    ("needle is a single NUL", ["ab", "cd"], NUL),
    ("leading NUL boundary span", [NUL + "a", "b"], NUL + "ab"),
    ("case sensitivity", ["Ab", "aB", "ab"], "ab"),
    ("one byte values", ["a", "b", "a"], "a"),
    ("the needle is a whole alphabet", ["abcdefghij"], "abcdefghij"),
]


@pytest.mark.fast
@pytest.mark.parametrize("label,values,needle", EDGE_CASES,
                         ids=[c[0] for c in EDGE_CASES])
def test_body_scan_is_bit_identical_to_arrow_on_edge_cases(kernel, label, values, needle):
    a = kernel.alias
    ref = reference(values, needle)
    enc = native_body(values)
    for lane in ("auto", "body", "arrow"):
        got = a["dict_contains_lut"](enc, needle, kernel=lane)
        assert np.array_equal(got, ref), f"{label} / {lane}"
        assert np.array_equal(a["dict_not_contains_lut"](enc, needle, kernel=lane),
                              ~ref), f"{label} / {lane} / not-contains"
    # the same answer through the flat carrier, so the lanes cannot diverge
    assert np.array_equal(a["dict_contains_lut"](values, needle, kernel="arrow"), ref)


@pytest.mark.fast
def test_body_scan_never_matches_across_an_entry_boundary(kernel):
    # the flat body of these two values contains the needle; neither value does
    values = ["the prefix", "the suffix"]
    needle = "xthe"
    assert needle in "".join(values)
    assert needle not in values[0] and needle not in values[1]
    ref = reference(values, needle)
    assert not ref.any()
    assert not kernel.alias["dict_contains_lut"](native_body(values), needle, kernel="body").any()


@pytest.mark.fast
@pytest.mark.parametrize("needle", ["é", "и", "\U0001F600", "Привет"])
def test_a_valid_needle_can_only_match_at_a_code_point_boundary(kernel, needle):
    """Why byte containment IS code-point containment for a str needle.

    UTF-8 is self-synchronizing, so no byte of a valid needle can occur at
    a non-boundary offset of a valid haystack -- unless the bytes belong to
    two different values, which is the boundary case the scan rejects. This
    test brute-forces every offset to keep that property from being an
    assumption in the docstring.
    """
    rng = random.Random(SEED)
    for _ in range(60):
        pre = "".join(rng.choice("ab" + MULTIBYTE) for _ in range(rng.randint(0, 6)))
        post = "".join(rng.choice("ab" + MULTIBYTE) for _ in range(rng.randint(0, 6)))
        hay = pre + needle + post
        raw, nraw = hay.encode("utf-8"), needle.encode("utf-8")
        found = [i for i in range(len(raw) - len(nraw) + 1)
                 if raw[i:i + len(nraw)] == nraw]
        assert found, "the fixture must actually contain the needle"
        start = len(pre.encode("utf-8"))
        assert start in found, "the fixture must contain the needle"
        # every occurrence sits on a code-point boundary: a match at any
        # other offset would be a match inside one code point
        for i in found:
            assert i == 0 or (raw[i] & 0xC0) != 0x80, f"mid-character match at {i}"
        enc = native_body([hay])
        assert kernel.alias["dict_contains_lut"](enc, needle, kernel="body")[0]


@pytest.mark.fast
def test_body_scan_fuzz_against_arrow(kernel):
    rng = random.Random(SEED)
    alphabet = "abcAB" + MULTIBYTE + NUL
    for _ in range(300):
        d = rng.randint(1, 12)
        values = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 8)))
                  for _ in range(d)]
        needle = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 6)))
        ref = reference(values, needle)
        enc = native_body(values)
        assert np.array_equal(kernel.alias["dict_contains_lut"](enc, needle, kernel="body"), ref)
        assert np.array_equal(kernel.alias["dict_contains_lut"](enc, needle, kernel="auto"), ref)
        assert np.array_equal(kernel.alias["dict_not_contains_lut"](enc, needle, kernel="auto"), ~ref)


@pytest.mark.fast
def test_empty_needle_is_all_true_for_contains_and_all_false_otherwise(kernel):
    values = ["", "a", "bb"]
    enc = native_body(values)
    n = len(values)
    assert np.array_equal(kernel.alias["dict_contains_lut"](enc, ""), np.ones(n, bool))
    assert np.array_equal(kernel.alias["dict_not_contains_lut"](enc, ""), np.zeros(n, bool))


@pytest.mark.fast
def test_non_utf8_needle_is_refused_not_scanned(kernel):
    values = ["a", "b"]
    enc = native_body(values)
    for bad in (b"\xff", b"\x80", b"\xc3"):
        with pytest.raises(ValueError):
            kernel.alias["dict_contains_lut"](enc, bad)
        with pytest.raises(ValueError):
            kernel.alias["dict_not_contains_lut"](enc, bad)
    for bad in (5, 3.5, None, ["a"]):
        with pytest.raises(ValueError):
            kernel.alias["dict_contains_lut"](enc, bad)
    # a valid UTF-8 byte needle is accepted and equals its str form
    assert np.array_equal(kernel.alias["dict_contains_lut"](enc, "b".encode("utf-8")),
                          kernel.alias["dict_contains_lut"](enc, "b"))


@pytest.mark.fast
def test_forced_body_kernel_on_a_carrier_without_a_body_is_an_error(kernel):
    with pytest.raises(ValueError):
        kernel.alias["dict_contains_lut"](["abc", "abd"], "ab", kernel="body")
    with pytest.raises(ValueError):
        kernel.alias["dict_contains_lut"](native_body(["abc"]), "ab", kernel="nope")


@pytest.mark.fast
def test_null_value_is_never_a_match_in_either_direction(kernel):
    # a native body carries no null bitmap -- it is the value lane, so the
    # null rule is checked on the flat carrier, where a NULL can exist
    arr = pa.array(["ab", None, "abc"], type=pa.string())
    contains = kernel.alias["dict_contains_lut"](arr, "ab")
    not_contains = kernel.alias["dict_not_contains_lut"](arr, "ab")
    assert not contains[1] and not not_contains[1]
    assert contains[0] and not not_contains[0]
    assert contains[2] and not not_contains[2]


@pytest.mark.fast
def test_large_domain_both_kernels_agree(kernel):
    rng = random.Random(SEED)
    tokens = ["".join(rng.choice("abcdefghij") for _ in range(12)) for _ in range(2000)]
    values = sorted(set(tokens + ["".join(rng.choice("abcdefghij") for _ in range(12))
                                 for _ in range(2000)]))
    enc = kernel.alias["dictionary_encode"](list(values))
    for needle in (values[0][:5], values[len(values) // 2][:7], "zzzzzz"):
        assert np.array_equal(kernel.alias["dict_contains_lut"](enc, needle, kernel="body"),
                              kernel.alias["dict_contains_lut"](enc, needle, kernel="arrow"))
        assert np.array_equal(kernel.alias["dict_contains_lut"](enc, needle, kernel="auto"),
                              kernel.alias["dict_contains_lut"](enc, needle, kernel="arrow"))
