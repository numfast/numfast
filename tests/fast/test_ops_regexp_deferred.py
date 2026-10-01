# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Stage 6 gate: ir_text_regex_replace contract (DuckDB parity).

Pins the contract: IR node + CPU op exist, first-match-only
semantics, NULL->NULL, invalid pattern is an explicit error.
"""

from pathlib import Path
import random
import re

import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


@pytest.mark.fast
def test_regex_replace_exists(kernel):
    a = kernel.alias
    assert callable(a["ir_text_regex_replace"])
    cap = a["cpu_capability"]()
    assert "text_regex_replace" in cap["ops"]
    node = a["ir_text_regex_replace"]("o", ["abc"], "b", "X")
    assert node["op"] == "text_regex_replace"
    assert node["out"] == "o"


@pytest.mark.fast
def test_ordinary_replace_first_match_only(kernel):
    a = kernel.alias
    out = a["cpu_execute"]([a["ir_text_regex_replace"]("o", ["aaa"], "a", "X")])["o"]
    assert list(out) == ["Xaa"]
    out = a["cpu_execute"]([a["ir_text_regex_replace"]("o", ["abc123def"], "[0-9]+", "#")])["o"]
    assert list(out) == ["abc#def"]
    out = a["cpu_execute"]([a["ir_text_regex_replace"]("o", ["abab"], "(ab)", r"X\1")])["o"]
    assert list(out) == ["Xabab"]
    assert list(out) == [re.sub("(ab)", r"X\1", "abab", count=1)]


@pytest.mark.fast
def test_null_never_matches(kernel):
    a = kernel.alias
    res = a["cpu_execute"]([a["ir_text_regex_replace"]("o", ["aaa", None, "bbb"], "a", "X")])
    assert list(res["o"]) == ["Xaa", None, "bbb"]
    assert list(res["o#validity"]) == [True, False, True]
    res = a["cpu_execute"]([a["ir_text_regex_replace"]("o", [123, "abc"], "a", "X")])
    assert list(res["o"]) == [None, "Xbc"]
    assert list(res["o#validity"]) == [False, True]


@pytest.mark.fast
def test_invalid_pattern_explicit_error(kernel):
    a = kernel.alias
    with pytest.raises(ValueError, match="bad pattern"):
        a["cpu_execute"]([a["ir_text_regex_replace"]("o", ["aaa"], "([", "X")])
    with pytest.raises(ValueError, match="must be str"):
        a["ir_text_regex_replace"]("o", ["aaa"], 123, "X")
    with pytest.raises(ValueError, match="must be str"):
        a["ir_text_regex_replace"]("o", ["aaa"], "a", 123)


@pytest.mark.fast
def test_unicode_empty_nomatch_multiple(kernel):
    a = kernel.alias

    def run(vals, pat, rep):
        return list(a["cpu_execute"]([a["ir_text_regex_replace"]("o", vals, pat, rep)])["o"])

    assert run(["cafe"], "e", "E") == ["cafE"]
    assert run(["a,b,a,b"], ",", ";") == ["a;b,a,b"]
    assert run(["abc", "xyz"], "z", "X") == ["abc", "xyX"]
    assert run(["abc", ""], "z", "X") == ["abc", ""]
    assert run(["", "abc"], "a", "X") == ["", "Xbc"]
    assert run(["abc"], "", "X") == ["Xabc"]
    assert run(["aaa"], "a", "X") == [re.sub("a", "X", "aaa", count=1)]


@pytest.mark.fast
def test_parity_seed42_vs_duckdb(kernel):
    duckdb = pytest.importorskip("duckdb")
    a = kernel.alias
    rng = random.Random(42)
    pool = ["aaa", "abc123def", "hello world", "", "a,b,a,b", "cafe", "xyz", "abab", None]
    sample = [rng.choice(pool) for _ in range(20)]
    pattern, repl = "[0-9]+", "#"
    res = a["cpu_execute"]([a["ir_text_regex_replace"]("o", sample, pattern, repl)])
    got = list(res["o"])
    expected = [None if v is None else re.sub(pattern, repl, v, count=1) for v in sample]
    assert got == expected
    for v, g in zip(sample, got):
        if v is None:
            assert g is None
            continue
        exp = duckdb.sql("select regexp_replace(?, ?, ?)", params=[v, pattern, repl]).fetchall()[0][0]
        assert g == exp
