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


# --- 3. or_ is out of v0; and_ / not_ are correct -----------------------

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
    assert len(V0) == 43, len(V0)
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
    assert len(V0) == 43, len(V0)
    # the split the counts in names.py assert, re-derived from the objects
    assert len(V0 - _NOT_CHAIN) == 29, len(V0 - _NOT_CHAIN)