# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""`Expr.is_null()` -- the NULL test, and the §2.2b guard's own advice.

The two NULL-key guards (`group`, `sort`) already read validity OUTSIDE the
graph through `Chain.material()` in order to REFUSE. `is_null()` calls that
same helper and returns the answer, so the §2.2b advice "filter those rows out
before group()" is executable from the public surface instead of being prose.

ORACLE DISCIPLINE (the two mistakes this project has already paid for):
every expected value below is computed by pandas over the IDENTICAL input, and
the subject is read through the SAME view (`to_pandas()`, the consumer view) in
both directions. An out-of-graph read is never compared against a chained
expression, and `to_numpy()` is never compared against `to_pandas()`.

FIXTURE DTYPES, stated because two dtype traps have each caused a false
conclusion in this project:
  * every nullable numeric fixture uses `pd.array(..., dtype="Float64")` or
    `dtype="Int64"` -- NEVER a bare `pd.DataFrame({"v": [None, None]})`, which
    is OBJECT dtype and NumFast reads it as TEXT;
  * `pd.array([...], dtype="Int64")` is read by NumFast as **int32**, not
    int64, so the engine-side dtype is asserted where it matters;
  * text fixtures use `dtype="string"` (StringDtype, `na_value=<NA>`), which
    NumFast reads as a `text` column carrying a validity sidecar.

`is_null()` is NOT `not_()`. `not_` is a correct 3VL negation of a predicate
(`NOT(NULL)` is UNKNOWN); `is_null()` asks whether the row IS NULL, which is a
fact, so its answer is hard True/False on every row. The two are pinned apart
by `test_is_null_is_not_the_negation_of_a_predicate`.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_EXT = Path(__file__).resolve().parents[2] / "src" / "Semantic" / "TableExpr"


def _load_names():
    """The Extension's own _lib/names.py by path (no sys.path change)."""
    spec = importlib.util.spec_from_file_location(
        "_lib.names", _EXT / "_lib" / "names.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NAMES = _load_names()


def _c(name):
    """`q.c(name)` from the LIVE kernel -- never a module-level singleton."""
    import numfast as nf
    return nf.app().c(name)


def _mask(chain, column, name="m"):
    """Read one boolean mask column out of a chain, consumer view only.

    `nf.app()` (not a module-level singleton) so the expression is built by the
    kernel that is live at call time -- same rule as `_table_query`.
    """
    import numfast as nf
    return chain.derive(name, nf.app().c(column).is_null()).compile() \
        .to_pandas()[name].tolist()


# --- 1..6: the four per-row classes, each with a pandas oracle -------------

@pytest.mark.fast
def test_is_null_numeric_column_with_a_sidecar_matches_pandas_isna():
    import numfast as nf
    frame = pd.DataFrame({"v": pd.array([1.0, None, 3.0], dtype="Float64")})
    t = nf.from_pandas(frame)
    assert t.column("v").dtype == "float64"
    assert t.column("v").validity.tolist() == [True, False, True]
    assert _mask(t.query(), "v") == frame["v"].isna().tolist()
    assert _mask(t.query(), "v") == [False, True, False]


@pytest.mark.fast
def test_is_null_numeric_column_without_a_sidecar_is_all_false():
    import numfast as nf
    # float64 ndarray column -> NumFast carries NO validity sidecar at all
    frame = pd.DataFrame({"v": np.array([1.0, 2.0, 3.0], dtype=np.float64)})
    t = nf.from_pandas(frame)
    assert t.column("v").validity is None
    assert _mask(t.query(), "v") == frame["v"].isna().tolist()
    assert _mask(t.query(), "v") == [False, False, False]


@pytest.mark.fast
def test_is_null_nullable_text_column_matches_pandas_isna():
    import numfast as nf
    frame = pd.DataFrame({"k": pd.array(["a", None, "c"], dtype="string")})
    t = nf.from_pandas(frame)
    assert t.column("k").dtype == "text"
    assert _mask(t.query(), "k") == frame["k"].isna().tolist()
    assert _mask(t.query(), "k") == [False, True, False]


@pytest.mark.fast
def test_is_null_text_validity_is_the_dictionary_encode_validity():
    """§2.2b names TWO observers for the same fact. `material()` reads the
    first one (`Series.validity`, copied onto the Col by `_source_col`);
    this pins that it equals the second (`dictionary_encode(...)['validity']`)
    row for row, which is what makes the text case a reuse and not a guess."""
    import numfast as nf
    frame = pd.DataFrame(
        {"k": pd.array(["a", None, "c", None, "b"], dtype="string")})
    t = nf.from_pandas(frame)
    series = t.column("k")
    enc = nf.get_kernel().alias["dictionary_encode"](
        list(series.to_numpy()), series.validity)
    assert np.array_equal(np.asarray(series.validity, dtype=bool),
                          np.asarray(enc["validity"], dtype=bool))
    assert _mask(t.query(), "k") == frame["k"].isna().tolist()


@pytest.mark.fast
def test_is_null_null_free_text_column_is_all_false():
    import numfast as nf
    frame = pd.DataFrame({"k": pd.array(["a", "b", "c"], dtype="string")})
    t = nf.from_pandas(frame)
    assert t.column("k").validity is None
    assert _mask(t.query(), "k") == frame["k"].isna().tolist()
    assert _mask(t.query(), "k") == [False, False, False]


@pytest.mark.fast
def test_is_null_all_null_column_is_all_true():
    import numfast as nf
    frame = pd.DataFrame({"v": pd.array([None, None], dtype="Float64")})
    t = nf.from_pandas(frame)
    assert t.column("v").validity.tolist() == [False, False]
    assert _mask(t.query(), "v") == frame["v"].isna().tolist()
    assert _mask(t.query(), "v") == [True, True]
    # the same for an all-NULL TEXT column -- not numeric-only
    txt = pd.DataFrame({"k": pd.array([None, None], dtype="string")})
    assert _mask(nf.from_pandas(txt).query(), "k") == \
        txt["k"].isna().tolist() == [True, True]


@pytest.mark.fast
def test_is_null_on_an_empty_column_is_empty():
    import numfast as nf
    for dtype, col in (("Float64", "v"), ("string", "k")):
        frame = pd.DataFrame({col: pd.array([], dtype=dtype)})
        t = nf.from_pandas(frame)
        chain = t.query().derive("m", _c(col).is_null())
        assert chain.nrows() == 0
        assert _mask(t.query(), col) == frame[col].isna().tolist() == []
    # a NULL-free numeric column with zero rows takes the all-False branch,
    # where the length comes from the DATA, not from a sidecar
    empty = pd.DataFrame({"v": np.array([], dtype=np.float64)})
    assert _mask(nf.from_pandas(empty).query(), "v") == []


# --- 7: a derived column -- the mask reads the DERIVED sidecar --------------

@pytest.mark.fast
def test_is_null_on_a_derived_column_reads_the_derived_validity():
    import numfast as nf
    vals = (1, None, 3, None, 5)
    # Int64 nullable -> NumFast reads int32 (asserted), and pandas cumsum
    # carries exactly the same NULL pattern, so the two oracles agree.
    frame = pd.DataFrame({"y": pd.array(list(vals), dtype="Int64")})
    t = nf.from_pandas(frame)
    assert t.column("y").dtype == "int32"
    oracle = frame["y"].cumsum().isna().tolist()
    assert oracle == [False, True, False, True, False]
    chain = t.query().derive("s", _c("y").cumsum())
    assert chain.derive("m", _c("s").is_null()).compile() \
        .to_pandas()["m"].tolist() == oracle

    # `shift` on a NULL-FREE column: v0 fills 0 into the DATA where pandas fills
    # NaN (a known, pinned v0/pandas difference), but both mark the SAME rows as
    # carrying no value -- so the oracle for the MASK is pandas' `shift().isna()`.
    clean = pd.DataFrame({"y": pd.array([1, 2, 3, 4, 5], dtype="Int64")})
    shift_oracle = clean["y"].shift(1).isna().tolist()
    assert shift_oracle == [True, False, False, False, False]
    shifted = nf.from_pandas(clean).query().derive("s", _c("y").shift(1))
    assert shifted.compile().column("s").validity.tolist() == \
        [not v for v in shift_oracle]
    assert shifted.derive("m", _c("s").is_null()).compile() \
        .to_pandas()["m"].tolist() == shift_oracle


# --- 8..9: row space. The stale answer must be DIFFERENT, not merely absent.

def _row_space_frame():
    """6 rows, asymmetric NULL pattern, NULL-free ASCENDING `o`.

    `desc=True` on `o` reverses the order, so the post-sort NULL pattern is
    NOT the pre-sort one -- a stale mask is provably distinguishable.
    `i` is a NULL-free int32 row-identity column.
    """
    return pd.DataFrame(
        {"v": pd.array([1.0, 2.0, None, 4.0, 5.0, None], dtype="Float64"),
         "i": np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
         "o": np.array([10., 20., 30., 40., 50., 60.], dtype=np.float64)})


@pytest.mark.fast
def test_is_null_after_filter_describes_the_filtered_rows():
    import numfast as nf
    frame = _row_space_frame()
    keep = frame["i"] > 2
    oracle = frame.loc[keep, "v"].isna().tolist()
    assert oracle == [False, False, True]
    chain = nf.from_pandas(frame).query().filter(_c("i") > 2)
    out = chain.derive("m", _c("v").is_null()).compile().to_pandas()
    assert out["i"].tolist() == frame.index[keep].tolist() == [3, 4, 5]
    assert out["m"].tolist() == oracle
    # the pre-filter mask is a different vector, so this cannot pass trivially
    assert out["m"].tolist() != frame["v"].isna().tolist()


@pytest.mark.fast
def test_is_null_after_sort_follows_the_rows_in_both_orders():
    """pre-pass THEN sort, and sort THEN pre-pass. The mask must follow the
    rows either way, and in both directions must differ from the stale mask."""
    import numfast as nf
    frame = _row_space_frame()
    stale = frame["v"].isna().tolist()
    ordered = frame.sort_values("o", ascending=False)
    oracle = ordered["v"].isna().tolist()
    assert oracle != stale, "fixture is degenerate: stale == correct"

    before = nf.from_pandas(frame).query() \
        .derive("m", _c("v").is_null()).sort("o", desc=True)
    out_before = before.compile().to_pandas()
    after = nf.from_pandas(frame).query().sort("o", desc=True) \
        .derive("m", _c("v").is_null())
    out_after = after.compile().to_pandas()

    assert out_before["i"].tolist() == ordered.index.tolist()
    assert out_after["i"].tolist() == ordered.index.tolist()
    assert out_before["m"].tolist() == oracle
    assert out_after["m"].tolist() == oracle
    assert out_before["m"].tolist() != stale
    assert out_after["m"].tolist() != stale
    # and the pre-pass node is genuinely gathered by the sort
    gathers = [(j["inputs"][0], j["inputs"][1]) for j in before.jobs()
               if j["op"] == "gather"]
    mask_node = [j["out"] for j in before.jobs()
                 if j["op"] == "series" and j["params"].get("dtype") == "bool"]
    assert len(mask_node) == 1, before.jobs()
    assert (mask_node[0], gathers[0][1]) in gathers, (mask_node, gathers)


# --- 10..11: the memo and repeated compilation ---------------------------

@pytest.mark.fast
def test_two_is_null_in_a_row_emit_one_node_and_agree():
    import numfast as nf
    frame = _row_space_frame()
    chain = nf.from_pandas(frame).query() \
        .derive("m1", _c("v").is_null()) \
        .derive("m2", _c("v").is_null())
    oracle = frame["v"].isna().tolist()
    bool_series = [j for j in chain.jobs()
                   if j["op"] == "series" and j["params"].get("dtype") == "bool"]
    # one node, not two: two IDENTICAL ir_series nodes would be merged by the
    # Planner's CSE and the duplicate's `out` rewritten under `_buffer`
    assert len(bool_series) == 1, chain.jobs()
    out = chain.compile().to_pandas()
    assert out["m1"].tolist() == out["m2"].tolist() == oracle


@pytest.mark.fast
def test_repeated_compile_after_is_null_is_stable():
    """Repeated `compile()` must not add a NEW bare internal-key error.

    `KeyError: '<node>#carry'` is the PRE-EXISTING node-reuse bug (the Planner's
    CSE rewrote a node whose sidecar is a bare internal name); the counter
    shifts by however many nodes this verb emitted. Any OTHER KeyError is a
    regression of this step and fails here.
    """
    import numfast as nf
    frame = _row_space_frame()
    chain = nf.from_pandas(frame).query().filter(_c("i") > 1) \
        .derive("m", _c("v").is_null()).sort("o", desc=True)
    oracle = frame.loc[frame["i"] > 1].sort_values(
        "o", ascending=False)["v"].isna().tolist()
    try:
        first = chain.compile().to_pandas()["m"].tolist()
        second = chain.compile().to_pandas()["m"].tolist()
    except KeyError as err:
        assert str(err).strip("'").endswith("#carry"), \
            f"NEW bare internal-key error from is_null: {err!r}"
        pytest.xfail("pre-existing CSE carry KeyError, node counter shifted")
    assert first == second == oracle


# --- 12: filter on the mask ------------------------------------------------

@pytest.mark.fast
def test_filter_is_null_selects_exactly_the_null_rows():
    import numfast as nf
    frame = _row_space_frame()
    kept = nf.from_pandas(frame).query().filter(_c("v").is_null()) \
        .compile().to_pandas()
    # row identity, not "the values happen to be NA"
    assert kept["i"].tolist() == frame.index[frame["v"].isna()].tolist() == [2, 5]
    assert bool(kept["v"].isna().all())


# --- 13: THE HEADLINE. The §2.2b advice, executed. ------------------------

@pytest.mark.fast
def test_the_group_guard_advice_is_executable():
    """"Fix: filter those rows out before group()" -- §2.2b, verbatim.

    Oracle: `df.dropna(subset=['k']).groupby('k')[...]` over the IDENTICAL
    frame. `k` is `string` (a NumFast `text` column with a sidecar); `rev` is a
    plain float64 ndarray column with no sidecar.
    """
    import numfast as nf
    frame = pd.DataFrame(
        {"k": pd.array(["a", None, "b", None, "c"], dtype="string"),
         "rev": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)})
    oracle = frame.dropna(subset=["k"]).groupby("k")["rev"].sum()
    out = nf.from_pandas(frame).query() \
        .filter(_c("k").is_null().not_()) \
        .group("k", {"rev": ("sum",)}).compile().to_pandas()
    assert out["k"].tolist() == list(oracle.index) == ["a", "b", "c"]
    assert out["rev.sum"].tolist() == [float(v) for v in oracle.tolist()] == \
        [1.0, 3.0, 5.0]


# --- 14: not_ / and_ untouched, and is_null is a DIFFERENT question --------

@pytest.mark.fast
def test_not_and_kleene_pins_are_unchanged():
    import numfast as nf
    frame = pd.DataFrame({"a": pd.array([1, None, 3], dtype="Int64"),
                          "b": pd.array([10, 20, None], dtype="Int64")})

    def facade(expr):
        return nf.from_pandas(frame).query().filter(expr).compile() \
            .to_pandas().to_dict("list")

    assert facade((_c("a") > 1).and_(_c("b") > 10)) == {"a": [], "b": []}
    assert facade(~(_c("a") > 1)) == {"a": [1], "b": [10]}
    assert facade(~_c("a").gt(1)) == {"a": [1], "b": [10]}


@pytest.mark.fast
def test_is_null_is_not_the_negation_of_a_predicate():
    """Two names, two questions, two paths -- pinned apart.

    `~(c('a') > 1)` asks "is the predicate NOT true", so `NULL > 1` makes the
    row UNKNOWN and `filter` drops it. `c('a').is_null().not_()` asks "is this
    row a value", and every non-NULL row is a hard True. Same 3-row fixture,
    same column, different answers -- if these two ever shared a path one of
    them would be wrong.
    """
    import numfast as nf
    frame = pd.DataFrame({"a": pd.array([1, None, 3], dtype="Int64"),
                          "b": pd.array([10, 20, None], dtype="Int64")})

    def facade(expr):
        return nf.from_pandas(frame).query().filter(expr).compile() \
            .to_pandas().to_dict("list")

    negated = facade(~(_c("a") > 1))
    null_free = facade(_c("a").is_null().not_())
    assert negated["a"] == [1]
    assert null_free["a"] == frame[frame["a"].notna()]["a"].tolist() == [1, 3]
    assert negated != null_free
    # `is_null` itself: the rows pandas calls NULL, and nothing else
    mask = nf.from_pandas(frame).query().derive("m", _c("a").is_null()) \
        .compile().to_pandas()["m"].tolist()
    assert mask == frame["a"].isna().tolist() == [False, True, False]
    # the mask is a hard fact, so it carries NO validity sidecar; that is what
    # lets `not_()` over it be a plain 3VL-free negation
    column = nf.from_pandas(frame).query().derive("m", _c("a").is_null()) \
        .compile().column("m")
    assert column.validity is None


# --- 15: the group guard still fires; sort needs no guard at all ------------

@pytest.mark.fast
def test_group_guard_fires_and_sort_needs_no_guard():
    """`group` refuses a NULL key (an ABSENT group, cpu.py:3325).

    `sort` does NOT refuse, and needs no advice: `ir_sort` keeps the NULL row
    and puts it LAST, in input order -- pandas' own `na_position="last"`.
    The old refusal cited GATE_semantics.md §9, whose own rendered
    `[1, 2, 3, 0]` is the NULL-LAST order; §9 was corrected on 2026-10-04.
    """
    import numfast as nf
    frame = pd.DataFrame(
        {"k": pd.array(["a", None, "b", None, "c"], dtype="string"),
         "rev": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)})
    t = nf.from_pandas(frame)
    with pytest.raises(ValueError) as group_err:
        t.query().group("k", {"rev": ("sum",)}).compile()
    group_msg = str(group_err.value)
    assert "key column 'k' has 2 NULL rows" in group_msg
    assert "silently dropped" in group_msg
    assert "Fix: filter those rows out before group()" in group_msg
    # sort keeps the NULL rows, last, in input order -- same as pandas
    ordered = t.query().sort("k").compile().to_pandas()
    assert ordered["k"].tolist() == frame.sort_values("k", kind="stable")["k"].tolist()
    assert [pd.isna(v) for v in ordered["k"].tolist()] ==         [False, False, False, True, True]
    assert ordered["rev"].tolist() == [1.0, 3.0, 5.0, 2.0, 4.0]
    # and the guard's own advice is executable, still agreeing with pandas
    filtered = t.query().filter(_c("k").is_null().not_()) \
        .sort("k").compile().to_pandas()
    assert filtered["k"].tolist() == frame.dropna(subset=["k"]).sort_values(
        "k")["k"].tolist() == ["a", "b", "c"]


# --- 16: the mask is an ordinary chunkable `ir_series`; `ir_where` absent --

@pytest.mark.fast
def test_is_null_mask_is_a_chunkable_series_and_ir_where_stays_absent():
    import numfast as nf
    frame = pd.DataFrame({"v": pd.array([1.0, None, 3.0], dtype="Float64")})
    chain = nf.from_pandas(frame).query().derive("m", _c("v").is_null())
    nodes = [j for j in chain.jobs() if j["out"].startswith("nul_")]
    assert len(nodes) == 1, chain.jobs()
    assert nodes[0]["op"] == "series", nodes
    assert nodes[0]["params"]["dtype"] == "bool", nodes
    assert nodes[0]["params"].get("validity") is None, nodes
    caps = nf.app().capabilities()
    hints = caps["chunkable_hints"]
    # the key is the bare op name; `ir_series` is the node constructor
    assert hints["series"] is True
    assert len(hints) == 32, len(hints)
    assert len(caps["ops"]) == 33, len(caps["ops"])
    assert "where" not in hints and "ir_where" not in hints
    assert "fill_null" not in NAMES.V0
    assert "ir_where" not in [j["op"] for j in chain.jobs()]


# --- the surface bookkeeping ----------------------------------------------

@pytest.mark.fast
def test_is_null_is_in_v0_and_the_registry_counts_are_44():
    assert "is_null" in NAMES.V0
    assert "is_null" in NAMES.v0_names()
    assert len(NAMES.V0) == 44, len(NAMES.V0)
    assert len(NAMES.v0_names()) == 44
    assert "window" not in NAMES.V0 and "or_" not in NAMES.V0
    assert "fill_null" not in NAMES.V0
    assert hasattr(_c("v"), "is_null")


@pytest.mark.fast
def test_is_null_refuses_on_an_expression_because_validity_is_a_column():
    """No declared validity on a derived expression, so refuse, don't guess."""
    import numfast as nf
    frame = pd.DataFrame({"v": pd.array([1.0, None, 3.0], dtype="Float64")})
    t = nf.from_pandas(frame)
    for build in (lambda e: e.cumsum().is_null(),
                  lambda e: e.mul(2).is_null(),
                  lambda e: e.gt(1).is_null()):
        with pytest.raises(ValueError) as err:
            t.query().derive("m", build(_c("v")))
        assert "is_null() needs a column reference" in str(err.value)
        assert "Fix: derive(" in str(err.value)
    # and on a column that is not in the table
    with pytest.raises(ValueError, match="no column 'nope'"):
        t.query().derive("m", _c("nope").is_null())