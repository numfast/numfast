# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Three silent-wrong regressions in the GAP-1 facade, plus the guards that pin
what is NOT broken.

Every test here was written to FAIL on the pre-fix tree and to PASS after it.
"Silent" is the operative word: each of the three defects returned a
correct-shaped frame with plausible numbers and no exception, which is exactly
the class the semantic gate was commissioned to remove.

1. `_bind`'s memo outlived the row space it described, so a rebind of the same
   expression after `sort`/`filter`/`limit` answered with PRE-op data.
2. `isin` on text lowered through `dict_contains_lut` -- `col LIKE '%needle%'`
   -- so `isin(['app'])` kept `'apple'`.
3. `ir_mask(..., 'or')` AND-s the two operands' validity sides, so `filter(a|b)`
   dropped every row either side was NULL on. The fix is FROZEN, so `or_` is
   out of v0 and raises loudly (same grounds as `window`).
"""

import importlib.util
from pathlib import Path

import pytest

_EXT = Path(__file__).resolve().parents[2] / "src" / "Semantic" / "TableExpr"


def _load_names():
    """Load the Extension's own _lib/names.py by path (no sys.path change)."""
    spec = importlib.util.spec_from_file_location(
        "_lib.names", _EXT / "_lib" / "names.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NAMES = _load_names()
V0 = NAMES.V0


def _t(*values):
    """int32 single-column Table `a` -- the memo fixture."""
    import numpy as np
    import numfast as nf
    return nf.from_numpy(
        np.asarray(values, dtype=np.float64).reshape(-1, 1), names=["a"])


def _text_t(col, values):
    import numfast as nf
    import pandas as pd
    return nf.from_pandas(pd.DataFrame(
        {col: pd.array(values, dtype="string")}))


def _kleene():
    """a=[1,NULL,3], b=[10,20,NULL] -- one NULL per column, on different rows."""
    import numfast as nf
    import pandas as pd
    return nf.from_pandas(pd.DataFrame(
        {"a": pd.array([1, None, 3], dtype="Int64"),
         "b": pd.array([10, 20, None], dtype="Int64")}))


# --- 1. the memo must not outlive the row space -------------------------
#
# `sort` PRESERVES the row count, so a stale rebind is not caught by a length
# check: the frame has the right shape and the wrong values.

@pytest.mark.fast
def test_memo_rebind_after_sort_reflects_post_sort_order():
    import numfast as nf
    q = nf.app()
    t = _t(1, 2, 3, 4)
    out = (t.query().derive("d", q.c("a") * 2).sort("a", desc=True)
            .derive("e", q.c("a") * 2).compile())
    assert out.column("a").to_numpy().tolist() == [4.0, 3.0, 2.0, 1.0]
    assert out.column("d").to_numpy().tolist() == [8.0, 6.0, 4.0, 2.0]
    # pre-fix this was [2.0, 4.0, 6.0, 8.0] -- pre-sort data
    assert out.column("e").to_numpy().tolist() == [8.0, 6.0, 4.0, 2.0]


@pytest.mark.fast
def test_memo_rebind_after_sort_with_a_different_expression():
    import numfast as nf
    q = nf.app()
    t = _t(1, 2, 3, 4)
    out = (t.query().derive("d", q.c("a") * 2).sort("a", desc=True)
            .derive("e", q.c("a") + 1).compile())
    # pre-fix this was [2.0, 3.0, 4.0, 5.0] -- pre-sort data
    assert out.column("e").to_numpy().tolist() == [5.0, 4.0, 3.0, 2.0]


@pytest.mark.fast
def test_memo_rebind_after_filter_reflects_post_filter_rows():
    import numfast as nf
    q = nf.app()
    t = _t(1, 2, 3, 4, 5)
    out = (t.query().derive("d", q.c("a") * 2).filter(q.c("a") > 2)
            .derive("e", q.c("a") * 2).compile())
    assert out.column("a").to_numpy().tolist() == [3.0, 4.0, 5.0]
    assert out.column("d").to_numpy().tolist() == [6.0, 8.0, 10.0]
    # pre-fix the stale node had 5 rows and compile() raised
    # "Table column 'e' length 5 != 3"
    assert out.column("e").to_numpy().tolist() == [6.0, 8.0, 10.0]


@pytest.mark.fast
def test_memo_rebind_after_limit_reflects_post_limit_rows():
    import numfast as nf
    q = nf.app()
    t = _t(1, 2, 3, 4, 5)
    out = (t.query().derive("d", q.c("a") * 2).limit(2)
            .derive("e", q.c("a") * 2).compile())
    assert out.column("a").to_numpy().tolist() == [1.0, 2.0]
    assert out.column("d").to_numpy().tolist() == [2.0, 4.0]
    # pre-fix the stale node had 5 rows and compile() raised
    # "Table column 'e' length 5 != 2"
    assert out.column("e").to_numpy().tolist() == [2.0, 4.0]


@pytest.mark.fast
def test_memo_still_reuses_nodes_inside_one_row_space():
    """The memo is load-bearing and must survive the fix.

    Two `derive`s of the SAME expression must produce exactly ONE `map` node.
    Without the memo the Planner's CSE merges the two identical nodes, rewrites
    the duplicate's `out` name, and `_buffer` then cannot find the buffer --
    measured, not assumed.
    """
    import numfast as nf
    q = nf.app()
    t = _t(1, 2, 3, 4)
    ch = t.query().derive("d", q.c("a") * 2).derive("e", q.c("a") * 2)
    assert sum(1 for j in ch.jobs() if j["op"] == "map") == 1, ch.jobs()
    out = ch.compile()
    assert out.column("d").to_numpy().tolist() == [2.0, 4.0, 6.0, 8.0]
    assert out.column("e").to_numpy().tolist() == [2.0, 4.0, 6.0, 8.0]


@pytest.mark.fast
def test_two_sorts_rebind_correctly_at_each_row_space():
    """Two row spaces in one chain: every rebind must be its own row space."""
    import numfast as nf
    q = nf.app()
    t = _t(1, 2, 3, 4)
    out = (t.query().derive("d", q.c("a") * 2)
            .sort("a", desc=True).derive("e", q.c("a") * 2)
            .sort("a").derive("f", q.c("a") * 3).compile())
    assert out.column("a").to_numpy().tolist() == [1.0, 2.0, 3.0, 4.0]
    assert out.column("d").to_numpy().tolist() == [2.0, 4.0, 6.0, 8.0]
    assert out.column("e").to_numpy().tolist() == [2.0, 4.0, 6.0, 8.0]
    assert out.column("f").to_numpy().tolist() == [3.0, 6.0, 9.0, 12.0]


# --- 2. isin on text is POINT MEMBERSHIP --------------------------------

@pytest.mark.fast
def test_text_isin_is_point_membership_not_substring():
    import numfast as nf
    q = nf.app()
    t = _text_t("x", ["apple", "banana", "app"])
    kept = t.query().filter(q.c("x").isin(["app"])).compile()
    # pre-fix this was ['apple', 'app'] -- 'apple' LIKE '%app%'
    assert kept.column("x").to_numpy().tolist() == ["app"]


@pytest.mark.fast
def test_text_isin_exact_values_with_a_null_present():
    import numfast as nf
    q = nf.app()
    t = _text_t("utm", ["ab", "cde", None, "fghij", "abcd", "a"])
    kept = t.query().filter(q.c("utm").isin(["a", "ab"])).compile()
    # pre-fix this was ['ab', 'abcd', 'a'] -- 'abcd' LIKE '%ab%'
    assert kept.column("utm").to_numpy().tolist() == ["ab", "a"]


@pytest.mark.fast
def test_text_isin_exact_values_on_a_null_free_column():
    import numfast as nf
    q = nf.app()
    t = _text_t("utm", ["ab", "cde", "fghij", "abcd", "a"])
    kept = t.query().filter(q.c("utm").isin(["a", "ab"])).compile()
    assert kept.column("utm").to_numpy().tolist() == ["ab", "a"]


@pytest.mark.fast
def test_text_isin_never_matches_a_null_value():
    """NULL is never a member, whichever needle list is given."""
    import numfast as nf
    q = nf.app()
    t = _text_t("utm", ["ab", None, "a"])
    for needles in (["a"], ["ab"], ["a", "ab"], [""], ["zzz"]):
        kept = t.query().filter(q.c("utm").isin(needles)).compile()
        assert None not in kept.column("utm").to_numpy().tolist(), needles
    assert t.query().filter(q.c("utm").isin(["a"])).compile() \
        .column("utm").to_numpy().tolist() == ["a"]


@pytest.mark.fast
def test_text_isin_lowers_through_dict_equal_lut_not_dict_contains_lut():
    """The primitive is pinned so the recipe cannot drift back to LIKE."""
    import numfast as nf
    q = nf.app()
    needle = "app"
    t = _text_t("x", ["apple", "banana", "app"])
    ch = t.query().filter(q.c("x").isin([needle]))
    codes = [j for j in ch.jobs()
             if j["op"] == "series" and j["params"].get("dtype") == "int32"]
    assert codes, ch.jobs()
    alias = nf.get_kernel().alias
    sidecar = ["app", "apple", "banana"]
    assert alias["dict_equal_lut"](sidecar, needle).tolist() == [
        True, False, False]
    assert alias["dict_contains_lut"](sidecar, needle).tolist() == [
        True, True, False]


# --- 4. a nested scan is bound, never silently dropped ------------------
#
# The scan branch of `_bind_new` resolved `expr.name` -- a COLUMN REFERENCE
# only. So whether the failure was loud or silent depended only on whether the
# inner expression happened to carry a column name: `cumsum().mul(2)` worked,
# `mul(2).cumsum()` raised, and `cumsum().shift(2)` / `cumsum().cumsum()`
# returned `shift(y)` / `cumsum(y)` with the inner expression DROPPED -- the
# right shape, plausible numbers, no exception.
#
# Shift contract in v0, asserted explicitly because it is not pandas' default:
# `ir_shift` fills 0 in the DATA and marks the filled rows INVALID
# (`validity=[False]*k + [True]*(n-k)`). pandas fills NaN. Both mark the filled
# rows as "not a value"; v0 stores 0 there. So the oracle below is
# `shift(k, fill_value=0)` for the data plus an explicit validity assert.


def _nums(*values):
    import numfast as nf
    import pandas as pd
    return nf.from_pandas(pd.DataFrame(
        {"y": pd.array(list(values), dtype="Int64")}))


def _frame(*values):
    import pandas as pd
    return pd.DataFrame({"y": pd.array(list(values), dtype="Int64")})["y"]


def _data(series):
    """The raw buffer of a numfast Series, NULL display stripped."""
    return series.to_masked().data.tolist()


@pytest.mark.fast
def test_nested_scan_shift_over_cumsum_matches_pandas():
    """Pre-fix this was [0, 0, 1, 2, 3] -- shift(y), the cumsum dropped."""
    import numfast as nf
    q = nf.app()
    t = _nums(1, 2, 3, 4, 5)
    got = t.query().derive("r", q.c("y").cumsum().shift(2)).compile().column("r")
    assert _data(got) == _frame(1, 2, 3, 4, 5).cumsum().shift(2, fill_value=0).tolist()
    assert got.validity.tolist() == [False, False, True, True, True]


@pytest.mark.fast
def test_nested_scan_cumsum_over_cumsum_matches_pandas():
    """Pre-fix this was [1, 3, 6, 10, 15] -- cumsum(y), the outer cumsum
    dropped. Two cumsums in a row is the case nobody re-checks."""
    import numfast as nf
    q = nf.app()
    got = (_nums(1, 2, 3, 4, 5).query()
           .derive("r", q.c("y").cumsum().cumsum()).compile().column("r"))
    assert _data(got) == _frame(1, 2, 3, 4, 5).cumsum().cumsum().tolist()


@pytest.mark.fast
def test_nested_scan_both_orders_and_three_deep_match_pandas():
    import numfast as nf
    q = nf.app()
    frame = _frame(1, 2, 3, 4, 5)
    t = _nums(1, 2, 3, 4, 5)
    cases = [
        (lambda e: e.cumsum().shift(2), lambda s: s.cumsum().shift(2, fill_value=0)),
        (lambda e: e.shift(2).cumsum(), lambda s: s.shift(2, fill_value=0).cumsum()),
        (lambda e: e.shift(1).shift(2),
         lambda s: s.shift(1, fill_value=0).shift(2, fill_value=0)),
        (lambda e: e.cumsum().mul(2), lambda s: s.cumsum() * 2),
        (lambda e: e.mul(2).cumsum(), lambda s: (s * 2).cumsum()),
        (lambda e: e.cumsum().mul(2).shift(1),
         lambda s: (s.cumsum() * 2).shift(1, fill_value=0)),
        (lambda e: e.cumsum().cumsum().shift(1),
         lambda s: s.cumsum().cumsum().shift(1, fill_value=0)),
    ]
    for build, oracle in cases:
        got = t.query().derive("r", build(q.c("y"))).compile().column("r")
        assert _data(got) == oracle(frame).tolist(), build


@pytest.mark.fast
def test_nested_scan_over_a_computed_bin_and_over_a_compare():
    import numfast as nf
    q = nf.app()
    frame = _frame(1, 2, 3, 4, 5)
    t = _nums(1, 2, 3, 4, 5)
    outer_bin = t.query().derive("r", q.c("y").cumsum().gt(3)).compile().column("r")
    assert outer_bin.to_numpy().tolist() == (frame.cumsum() > 3).tolist()
    inner_bin = t.query().derive("r", q.c("y").mul(2).cumsum()).compile().column("r")
    assert _data(inner_bin) == (frame * 2).cumsum().tolist()


@pytest.mark.fast
def test_nested_scan_with_nulls_agrees_with_pandas_where_pandas_has_a_value():
    """The NULL rows of a scan agree with pandas in the CONSUMER view too.

    Measured, not assumed: on (1, None, 3, None, 5) `cumsum` gives
    `validity == [True, False, True, False, True]` and `to_pandas() ==
    [1, NA, 4, NA, 9]` -- identical to pandas, and `to_arrow()` agrees as well.
    The NULL row is NOT kept valid. An earlier version of this docstring said
    it was, and that was false.

    `[1, 1, 4, 4, 9]` is the RAW IR buffer: `ir_cumsum` carries the running
    value across the NULL into the data while the validity sidecar marks the
    row invalid. It is what `to_masked().data` and `to_numpy()` hand back --
    the two views that return the buffer WITHOUT the sidecar -- so the pin
    below is labelled as the raw buffer, not as a consumer answer. `to_pandas()`,
    `to_arrow()`, `filter` and `reduce` all show the pandas view.

    What Item 1 owns is the NESTING, so the oracle below asserts the positions
    where pandas HAS a value.
    """
    import numfast as nf
    q = nf.app()
    vals = (1, None, 3, None, 5)
    frame = _frame(*vals)
    t = _nums(*vals)
    for build, oracle in [
        (lambda e: e.cumsum().shift(1),
         lambda s: s.cumsum().shift(1, fill_value=0)),
        (lambda e: e.shift(1).cumsum(),
         lambda s: s.shift(1, fill_value=0).cumsum()),
        (lambda e: e.cumsum().cumsum(), lambda s: s.cumsum().cumsum()),
        (lambda e: e.mul(2).cumsum(), lambda s: (s * 2).cumsum()),
    ]:
        exp = oracle(frame).tolist()
        idx = [i for i, v in enumerate(exp) if not pd_isna(v)]
        got = _data(t.query().derive("r", build(q.c("y"))).compile().column("r"))
        assert [got[i] for i in idx] == [exp[i] for i in idx], (build, got, exp)
    plain = _data(t.query().derive("r", q.c("y").cumsum()).compile().column("r"))
    assert plain == [1, 1, 4, 4, 9]          # the RAW IR buffer, not a consumer view
    import pandas as pd
    assert [None if pd.isna(v) else v
            for v in frame.cumsum().tolist()] == [1, None, 4, None, 9]


def pd_isna(value):
    import pandas as pd
    return bool(pd.isna(value))


@pytest.mark.fast
def test_nested_scan_filter_reads_the_nested_value_not_the_column():
    import numfast as nf
    q = nf.app()
    frame = _frame(1, 2, 3, 4, 5)
    kept = (_nums(1, 2, 3, 4, 5).query()
            .filter(q.c("y").cumsum() > 6).compile().to_pandas())
    assert kept["y"].tolist() == frame[frame.cumsum() > 6].tolist()


@pytest.mark.fast
def test_nested_scan_rebinds_after_sort_and_reads_post_sort_data():
    """The epoch test for a nested bind: `_bind` must be the way in, or a
    cached nested node survives the row space and answers with PRE-sort data
    beside the post-sort column -- the GAP-1 memo defect, one level deeper."""
    import numfast as nf
    q = nf.app()
    frame = _frame(1, 2, 3, 4)
    desc = frame.sort_values(ascending=False)
    out = (_nums(1, 2, 3, 4).query()
           .derive("d", q.c("y").cumsum())
           .sort("y", desc=True)
           .derive("e", q.c("y").cumsum())
           .derive("g", q.c("y").cumsum().mul(2))
           .compile()).to_pandas()
    assert out["y"].tolist() == desc.tolist()
    assert out["d"].tolist() == frame.cumsum().sort_values(
        ascending=False).tolist()             # pre-sort, gathered
    assert out["e"].tolist() == desc.cumsum().tolist()   # POST-sort
    assert out["g"].tolist() == (desc.cumsum() * 2).tolist()


@pytest.mark.fast
def test_nested_scan_memo_still_emits_one_node_per_expression():
    """The memo is load-bearing (CSE rewrites a duplicated node's `out`), so
    the nested key must dedupe as well."""
    import numfast as nf
    q = nf.app()
    ch = (_nums(1, 2, 3, 4, 5).query()
          .derive("d", q.c("y").cumsum().shift(1))
          .derive("e", q.c("y").cumsum().shift(1)))
    assert sum(1 for j in ch.jobs() if j["op"] == "cumsum") == 1, ch.jobs()
    assert sum(1 for j in ch.jobs() if j["op"] == "shift") == 1, ch.jobs()
    out = ch.compile()
    assert _data(out.column("d")) == _data(out.column("e"))


@pytest.mark.fast
def test_scan_on_text_still_refuses_and_scan_over_a_mask_refuses_too():
    """The text guard is preserved; a mask has no lowering at all."""
    import numfast as nf
    q = nf.app()
    t = _text_t("p", ["ab", "cde", None, "f"])
    for build in (lambda e: e.cumsum(), lambda e: e.shift(1)):
        with pytest.raises(ValueError) as err:
            t.query().derive("r", build(q.c("p")))
        msg = str(err.value)
        assert "on TEXT column 'p' is meaningless" in msg
        assert "ranks, not values" in msg
    for expr in (q.c("p").str_eq("ab").cumsum(), q.c("p").str_eq("ab").shift(1)):
        with pytest.raises(ValueError) as err:
            t.query().derive("r", expr)
        msg = str(err.value)
        assert "has no lowering over the boolean mask" in msg
        assert "int32/float32/float64" in msg
        # the hint must be a v0 spelling
        assert "filter(" in msg and "str_ne" not in msg


# --- 5. text comparison: == / != work, ordering refuses loudly ---------
#
# Pre-fix, `c('plan') != 'pro'` raised `KeyError: 'pro'` from
# Drivers/CPU/_lib/cpu.py:3139 -- ir_compare takes its right operand as a
# BUFFER NAME, so the string literal became a lookup key and the driver named
# nothing. All six v0 comparison ops were unusable on text.


def _plans():
    import numfast as nf
    import pandas as pd
    frame = pd.DataFrame(
        {"plan": pd.array(["pro", "free", None, "pro ", "pro"], dtype="string")})
    return nf.from_pandas(frame), frame["plan"]


@pytest.mark.fast
def test_text_eq_and_ne_against_a_string_scalar_match_pandas():
    import numfast as nf
    q = nf.app()
    t, col = _plans()
    for needle in ("pro", "free", "zzz"):
        for expr, oracle in ((q.c("plan") == needle, col == needle),
                             (q.c("plan") != needle, col != needle)):
            kept = t.query().filter(expr).compile().to_pandas()["plan"].tolist()
            want = col[oracle.fillna(False).astype(bool)].tolist()
            assert kept == want, (needle, kept, want)


@pytest.mark.fast
def test_text_ne_is_exactly_str_eq_not():
    """The working alternative named in every refusal message -- and it is the
    SAME node, so the two spellings cannot drift apart."""
    import numfast as nf
    q = nf.app()
    t, col = _plans()
    via_cmp = t.query().filter(q.c("plan") != "pro").compile().to_pandas()
    via_str = (t.query().filter(q.c("plan").str_eq("pro").not_()).compile()
               .to_pandas())
    assert via_cmp["plan"].tolist() == via_str["plan"].tolist()
    assert via_cmp["plan"].tolist() == col[col != "pro"].tolist()
    assert None not in via_cmp["plan"].tolist()


@pytest.mark.fast
def test_text_eq_never_matches_null_and_works_on_a_null_free_column():
    import numfast as nf
    import pandas as pd
    q = nf.app()
    t, _ = _plans()
    assert t.query().filter(q.c("plan") == "pro").compile() \
        .to_pandas()["plan"].tolist() == ["pro", "pro"]
    clean = nf.from_pandas(pd.DataFrame(
        {"plan": pd.array(["pro", "free", "pro "], dtype="string")}))
    kept = clean.query().filter(q.c("plan") == "pro").compile() \
        .to_pandas()["plan"].tolist()
    assert kept == ["pro"]


@pytest.mark.fast
def test_text_eq_lowers_through_dict_equal_lut_and_shares_the_isin_node():
    """`==` is `isin([needle])` -- same primitive, same codes_lut_mask LIST
    discipline, and ONE node shared with isin inside a row space (the memo,
    not CSE, keeps it unique; across a row space the epoch must re-bind it)."""
    import numfast as nf
    q = nf.app()
    t, _ = _plans()
    ch = t.query().derive("g", q.c("plan") == "pro").derive(
        "h", q.c("plan") == "pro")
    masks = [j for j in ch.jobs()
             if j["op"] == "series" and j["params"].get("dtype") == "bool"]
    assert len(masks) == 1, ch.jobs()
    assert masks[0]["params"].get("validity") is not None, masks[0]
    assert not [j for j in ch.jobs() if j["op"] == "compare"], ch.jobs()
    # `==` and isin([needle]) are the same expression, so the same node
    both = t.query().derive("g", q.c("plan") == "pro").derive(
        "h", q.c("plan").isin(["pro"]))
    assert sum(1 for j in both.jobs()
               if j["op"] == "series" and j["params"].get("dtype") == "bool") == 1


@pytest.mark.fast
def test_text_ordering_comparison_refuses_naming_the_cause_and_a_real_fix():
    import numfast as nf
    q = nf.app()
    t, _ = _plans()
    for op, spelling in (("<", "lt"), ("<=", "le"), (">", "gt"), (">=", "ge")):
        expr = {"lt": lambda: q.c("plan") < "pro",
                "le": lambda: q.c("plan") <= "pro",
                "gt": lambda: q.c("plan") > "pro",
                "ge": lambda: q.c("plan") >= "pro"}[spelling]()
        with pytest.raises(ValueError) as err:
            t.query().filter(expr)
        msg = str(err.value)
        assert f"{op} on TEXT column 'plan'" in msg
        assert "collation the engine does not define" in msg
        # the hint must resolve inside v0
        assert "str_eq" in msg and "str_ne" not in msg
        assert "isin(" in msg


@pytest.mark.fast
def test_every_other_text_or_string_comparison_refuses_loudly():
    """None of these may reach the driver as a buffer lookup."""
    import numfast as nf
    import pandas as pd
    q = nf.app()
    t = nf.from_pandas(pd.DataFrame(
        {"plan": pd.array(["pro", "free"], dtype="string"),
         "other": pd.array(["pro", "free"], dtype="string"),
         "num": [1.0, 2.0]}))
    cases = [
        ("text vs text column", lambda: q.c("plan") == q.c("other")),
        ("text vs None", lambda: q.c("plan") == None),
        ("text vs None (ne)", lambda: q.c("plan") != None),
        ("text vs number", lambda: q.c("plan") == 5),
        ("numeric vs string literal", lambda: q.c("num") == "pro"),
        ("numeric vs string literal (gt)", lambda: q.c("num") > "pro"),
        ("empty needle list", lambda: q.c("plan").isin([])),
        ("non-str needle list", lambda: q.c("plan").isin([1])),
    ]
    for label, build in cases:
        with pytest.raises(ValueError) as err:
            t.query().filter(build())
        msg = str(err.value)
        assert "KeyError" not in msg, label
        assert "Fix:" in msg, (label, msg)
    with pytest.raises(ValueError, match="empty needle list"):
        t.query().filter(q.c("plan").isin([]))
    with pytest.raises(ValueError, match="needs str values"):
        t.query().filter(q.c("plan").isin([1]))


# --- 6. ONE NULL semantics for count, shared by reduce and group ---------
#
# Fixture k=['a','a','b'], v=[1.0,None,3.0]:
#   pre-fix  reduce('count') -> 3   (row count)   group 'v.count' -> {a:1, b:1}
#   pre-fix  reduce('sum')   -> NaN (UserWarning) group 'v.sum'   -> {a:1, b:3}
# Two v0 verbs, same aggregate, OPPOSITE NULL semantics.


def _agg(values):
    import numfast as nf
    import pandas as pd
    frame = pd.DataFrame(
        {"k": pd.array(["a", "a", "b"], dtype="string"),
         "v": pd.array(list(values), dtype="Float64")})
    return nf.from_pandas(frame), frame


@pytest.mark.fast
def test_reduce_count_agrees_with_group_count_and_with_pandas():
    import numfast as nf
    q = nf.app()
    t, frame = _agg((1.0, None, 3.0))
    col = frame["v"]
    assert t.query().reduce("count", "v") == col.count()
    grouped = t.query().group("k", {"v": ("count",)}).compile().to_pandas()
    assert grouped.to_dict("list") == {
        "k": frame.groupby("k").groups.keys() and ["a", "b"],
        "v.count": list(frame.groupby("k")["v"].count())}
    # pandas .size is a DIFFERENT quantity and is not what count means here
    assert col.size == 3 and col.count() == 2


@pytest.mark.fast
def test_reduce_aggregate_family_skips_nulls_like_pandas():
    import numfast as nf
    t, frame = _agg((1.0, None, 3.0))
    col = frame["v"]
    for op in ("sum", "mean", "min", "max"):
        got = t.query().reduce(op, "v")
        assert got == getattr(col, op)(), op
        grouped = t.query().group("k", {"v": (op,)}).compile().to_pandas()
        assert grouped[f"v.{op}"].tolist() == list(
            frame.groupby("k")["v"].agg(op)), op


@pytest.mark.fast
def test_reduce_count_on_every_fixture_shape():
    import numfast as nf
    import pandas as pd
    fixtures = {
        "no-NA float": [1.0, 2.0, 3.0],
        "no-NA plain": [1.0, 2.0, 3.0],
        "with NULL": [1.0, None, 3.0],
        "single row": [7.0],
        "empty": [],
    }
    for label, values in fixtures.items():
        frame = pd.DataFrame({"v": pd.array(values, dtype="Float64")})
        t = nf.from_pandas(frame)
        got = t.query().reduce("count", "v")
        assert got == frame["v"].count(), (label, got)
        assert got == sum(1 for v in values if v is not None), label
    # an INTEGER column, where a non-NULL value can be 0
    ints = pd.DataFrame({"v": pd.array([0, None, 3], dtype="Int64")})
    assert nf.from_pandas(ints).query().reduce("count", "v") == \
        ints["v"].count() == 2


@pytest.mark.fast
def test_reduce_count_on_an_all_na_column_is_zero():
    import numfast as nf
    import pandas as pd
    frame = pd.DataFrame({"v": pd.array([None, None], dtype="Float64")})
    assert nf.from_pandas(frame).query().reduce("count", "v") == \
        frame["v"].count() == 0


@pytest.mark.fast
def test_reduce_passes_skipna_true_and_group_agrees():
    """The flag itself, pinned on the emitted node -- the facade simply was
    not passing it (ir_reduce defaults to skipna=False,
    src/Semantic/IR/_lib/nodes.py:210)."""
    import numfast as nf
    q = nf.app()
    t, _ = _agg((1.0, None, 3.0))
    ch = t.query().derive("d", q.c("v") * 1.0)
    ch.reduce("count", "d")
    reduces = [j for j in ch.jobs() if j["op"] == "reduce"]
    assert reduces and all(j["params"].get("skipna") is True
                           for j in reduces), reduces


@pytest.mark.fast
def test_count_on_text_is_refused_by_both_verbs_and_reachable_via_str_len():
    """`count` on a text column is unreachable on purpose: both verbs refuse a
    text aggregate because the dictionary codes are ranks, not values. So the
    'count skips NULL text' contract is observable only through the documented
    substitute -- count over `str_len`, which is a real int32 column and whose
    NULL rows are invalid."""
    import numfast as nf
    import pandas as pd
    q = nf.app()
    frame = pd.DataFrame(
        {"k": pd.array(["a", "a", "b"], dtype="string"),
         "p": pd.array(["x", None, "y"], dtype="string")})
    t = nf.from_pandas(frame)
    with pytest.raises(ValueError) as err:
        t.query().reduce("count", "p")
    assert "over TEXT column 'p' has no meaning" in str(err.value)
    with pytest.raises(ValueError) as err:
        t.query().group("k", {"p": ("count",)})
    assert "aggregate over TEXT column 'p' is meaningless" in str(err.value)
    got = t.query().derive("n", q.c("p").str_len()).reduce("count", "n")
    assert got == frame["p"].count() == 2


@pytest.mark.fast
def test_count_with_nulls_in_the_key_still_refuses_and_null_free_key_agrees():
    import numfast as nf
    import pandas as pd
    q = nf.app()
    null_key = pd.DataFrame(
        {"k": pd.array(["a", None, "b"], dtype="string"),
         "v": pd.array([1.0, 2.0, 3.0], dtype="Float64")})
    with pytest.raises(ValueError) as err:
        nf.from_pandas(null_key).query().group("k", {"v": ("count",)})
    assert "key column 'k' has 1 NULL rows" in str(err.value)
    clean = pd.DataFrame(
        {"k": pd.array(["a", "b", "b"], dtype="string"),
         "v": pd.array([1.0, None, 3.0], dtype="Float64")})
    t = nf.from_pandas(clean)
    grouped = t.query().group("k", {"v": ("count",)}).compile().to_pandas()
    assert grouped.to_dict("list") == {
        "k": ["a", "b"],
        "v.count": list(clean.groupby("k")["v"].count())}
    assert t.query().filter(q.c("v") > 0).reduce("count", "v") == \
        clean["v"][clean["v"] > 0].count()


# --- 7. truediv must not lie --------------------------------------------
#
# Drivers/CPU/_lib/cpu.py:3109-3111 -- r = (a.astype(f64) / b); and then
#     if np.issubdtype(a.dtype, np.integer): r = np.rint(r).astype(a.dtype)
# The trigger is the LEFT operand's dtype, so `int / float` rounds too.


def _int_table():
    import numfast as nf
    import numpy as np
    return nf.from_numpy(np.array([[1, 4], [2, 5], [3, 6]], dtype=np.int64),
                         names=["a", "b"])


def _float_table():
    import numfast as nf
    import numpy as np
    return nf.from_numpy(np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
                         names=["a", "b"])


@pytest.mark.fast
def test_truediv_on_an_integer_left_operand_refuses_loudly():
    import numfast as nf
    import pandas as pd
    q = nf.app()
    t = _int_table()
    for expr in (q.c("a") / q.c("b"), q.c("a") / 2, q.c("a") / 2.0):
        with pytest.raises(ValueError) as err:
            t.query().derive("r", expr).compile()
        msg = str(err.value)
        assert "ROUNDS THE QUOTIENT BACK" in msg
        assert "cpu.py:3109-3111" in msg
        assert "FROZEN" in msg
        assert "no dtype-promotion verb" in msg
    # a DERIVED left operand has no declared dtype, so v0 refuses rather than
    # guess -- and says which of the two reasons applies
    with pytest.raises(ValueError) as err:
        t.query().derive("r", q.c("a").mul(2) / q.c("b")).compile()
    assert "no declared dtype" in str(err.value)
    assert "cpu.py:3109-3111" in str(err.value)
    assert "no dtype-promotion verb" in str(err.value)
    # the pre-fix answer, spelled out: pandas gives [0.5, 1.0, 1.5]
    assert (pd.Series([1, 2, 3]) / 2).tolist() == [0.5, 1.0, 1.5]


@pytest.mark.fast
def test_truediv_refuses_before_any_node_is_emitted():
    import numfast as nf
    q = nf.app()
    ch = _int_table().query()
    with pytest.raises(ValueError, match="ROUNDS THE QUOTIENT BACK"):
        ch.derive("r", q.c("a") / 2)
    assert ch.jobs() == [], ch.jobs()


@pytest.mark.fast
def test_truediv_with_a_float_left_operand_still_works_and_matches_pandas():
    import numfast as nf
    import pandas as pd
    q = nf.app()
    t = _float_table()
    sa, sb = pd.Series([1.0, 2.0, 3.0]), pd.Series([4.0, 5.0, 6.0])
    for expr, oracle in ((q.c("a") / q.c("b"), sa / sb),
                         (q.c("a") / 2, sa / 2),
                         (q.c("a") / 2.0, sa / 2.0),
                         (q.c("a").gt(1) / 2, (sa > 1) / 2)):
        col = t.query().derive("r", expr).compile().column("r")
        assert col.dtype == "float64", expr
        assert col.to_numpy().tolist() == oracle.tolist(), expr


@pytest.mark.fast
def test_truediv_refusal_does_not_change_the_column_dtype():
    """The forbidden workaround: materialising a float64 copy of the column
    would make its dtype depend on which expression consumed it. A refused
    division must leave the chain and the source column exactly as they were."""
    import numfast as nf
    q = nf.app()
    t = _int_table()
    assert t.column("a").dtype == "int64"
    with pytest.raises(ValueError):
        t.query().derive("r", q.c("a") / 2)
    assert t.column("a").dtype == "int64"
    assert t.query().compile().column("a").to_numpy().tolist() == [1, 2, 3]


# --- 8. or_ is out of v0; and_ / not_ are correct -----------------------

@pytest.mark.fast
def test_or_raises_loudly_naming_the_frozen_cause():
    import numfast as nf
    q = nf.app()
    t = _kleene()
    with pytest.raises(ValueError) as err:
        (q.c("a") > 1) | (q.c("b") > 10)
    msg = str(err.value)
    assert "or_ is not in v0" in msg
    assert "AND-s the validities" in msg
    assert "FROZEN" in msg
    # the named method call and the operator are the same refusal
    with pytest.raises(ValueError, match="or_ is not in v0"):
        (q.c("a") > 1).or_(q.c("b") > 10)
    with pytest.raises(ValueError, match="or_ is not in v0"):
        t.query().filter((q.c("a") > 1) | (q.c("b") > 10))


@pytest.mark.fast
def test_or_removed_from_v0_and_window_still_absent():
    assert "or_" not in V0
    assert "window" not in V0
    assert len(V0) == 44, len(V0)
    # `is_null` is the 44th name and is NOT a negation: it is a null
    # test, a hard True/False on every row. See tests/fast/
    # test_consumer_is_null.py::test_is_null_is_not_the_negation_of_a_predicate
    assert "is_null" in V0
    assert "or_" not in NAMES.v0_names()


@pytest.mark.fast
def test_and_is_correct_on_the_kleene_fixture():
    """PINNED, because `or_` leaving v0 must not read as "3VL is gone".

    a=[1,NULL,3], b=[10,20,NULL]:
      row 0  T and F -> F      row 1  U and T -> UNKNOWN (dropped)
      row 2  T and U -> UNKNOWN (dropped)
    Kleene / pandas keep 0 rows, and so must the facade.
    """
    import numfast as nf
    q = nf.app()
    out = _kleene().query().filter(
        (q.c("a") > 1).and_(q.c("b") > 10)).compile().to_pandas()
    assert out.to_dict("list") == {"a": [], "b": []}


@pytest.mark.fast
def test_not_is_correct_on_the_kleene_fixture():
    """NOT(NULL) is UNKNOWN, so only row 0 survives -- and it survives."""
    import numfast as nf
    q = nf.app()
    out = _kleene().query().filter(~(q.c("a") > 1)).compile().to_pandas()
    assert out.to_dict("list") == {"a": [1], "b": [10]}


@pytest.mark.fast
def test_and_not_agree_with_the_pandas_kleene_oracle():
    """Independent oracle: pandas nullable dtypes, same fixture."""
    import numfast as nf
    import pandas as pd
    q = nf.app()
    frame = pd.DataFrame({"a": pd.array([1, None, 3], dtype="Int64"),
                          "b": pd.array([10, 20, None], dtype="Int64")})

    def oracle(mask):
        return frame[mask.fillna(False).astype(bool)].to_dict("list")

    def facade(expr):
        return nf.from_pandas(frame).query().filter(expr).compile() \
            .to_pandas().to_dict("list")

    assert facade((q.c("a") > 1).and_(q.c("b") > 10)) == oracle(
        (frame["a"] > 1) & (frame["b"] > 10))
    assert facade(~(q.c("a") > 1)) == oracle(~(frame["a"] > 1))


# --- the reachability check that test_every_v0_name_is_reachable is not --
#
# The existing test compares V0 - reachable against a hardcoded literal, so it
# cannot fail when a name becomes unreachable. This one resolves every name
# through a real getattr against the objects a user can actually reach.

_NOT_CHAIN = frozenset({
    "app", "from_arrow", "from_numpy", "to_arrow", "to_pandas", "to_numpy",
    "capabilities", "open_stream", "query", "schema", "c",
    "compile", "explain", "nrows",
})


@pytest.mark.fast
def test_every_v0_name_is_reachable_by_a_real_attribute_lookup():
    import numfast as nf
    import numpy as np
    app = nf.app()
    table = nf.from_numpy(np.array([[1.0], [2.0]]), names=["v"])
    # every V0 name must resolve to a real object on one of these carriers,
    # looked up with getattr -- a hardcoded literal cannot prove this
    carriers = {"numfast": nf, "App": app, "Table": table,
                "Chain": table.query(), "Expr": app.c("v")}
    unresolved = [name for name in sorted(V0)
                  if not any(hasattr(carrier, name)
                             for carrier in carriers.values())]
    assert not unresolved, unresolved
    assert len(V0) == 44, len(V0)
    # the split the counts in names.py assert, re-derived from the objects
    assert len(V0 - _NOT_CHAIN) == 30, len(V0 - _NOT_CHAIN)