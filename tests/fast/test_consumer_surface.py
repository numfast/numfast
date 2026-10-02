# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""GAP-1 consumer surface: the two executable checks of DESIGN §3.4.

Both live INSIDE the Extension's own scope and read `_lib.names` at relative
level 1 -- the one path the import-guard permits. Importing `numfast._lib`
from outside would itself be the PRIVATE access §2.1 forbids.

`check_alias_delta` counts the DELTA, not a screenshot: the Extension adds
exactly the four `tableexpr_*` aliases of TableExpr.toml, independently of
which methods the chain happens to have.
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


_NAMES = _load_names()
V0 = _NAMES.V0

ALIASES_BEFORE = 126                          # public alias names BEFORE

NOT_CHAIN = frozenset({
    "app", "from_arrow", "from_numpy", "to_arrow", "to_pandas", "to_numpy",
    "capabilities", "open_stream", "query", "schema", "c",
    "compile", "explain", "nrows",
})


def check_no_internal_leak():
    # Rule §3: no surface-B name collides with an internal alias name.
    # One documented exception: `compile` -- 07:60 requires a public
    # `.compile()` method while kernel.alias['compile'] (07:59,
    # Runtime/Planner) stays internal. The level is part of the name:
    # `Chain.compile` != `compile`.
    import numfast as nf
    a = set(nf.get_kernel().alias)
    internal = {n for n in a if not n.startswith("_")} - {"compile"}
    module_level = set(nf.__all__)
    assert not (module_level & internal), module_level & internal
    chain = V0 - NOT_CHAIN
    assert not (chain & internal), chain & internal
    assert not any(m.startswith(("ir_", "CompositeGroup")) for m in chain)
    return True


def check_alias_delta(before=ALIASES_BEFORE, expected=4):
    # The screenshot 126 is replaced by a delta: the Extension adds exactly
    # the four tableexpr_* aliases declared in TableExpr.toml.
    import numfast as nf
    after = len([n for n in nf.get_kernel().alias if not n.startswith("_")])
    assert after - before == expected, (after, before)
    return True


# --- V0 itself -------------------------------------------------------------

@pytest.mark.fast
def test_v0_is_44_names_and_has_no_window():
    assert len(V0) == 44, len(V0)
    assert "window" not in V0
    assert len(V0 - NOT_CHAIN) == 30
    assert len(_NAMES.v0_names()) == 44


@pytest.mark.fast
def test_window_is_absent_from_the_chain_surface():
    import numfast as nf
    app = nf.app()
    assert not hasattr(app, "window")
    import numpy as np
    chain = app.query(nf.from_numpy(np.ones((2, 1), np.float64), names=["v"]))
    assert not hasattr(chain, "window")


@pytest.mark.fast
def test_check_no_internal_leak_passes():
    assert check_no_internal_leak() is True


@pytest.mark.fast
def test_check_alias_delta_passes():
    assert check_alias_delta() is True


# --- manifest shape (07:20 / 07:21) ---------------------------------------

@pytest.mark.fast
def test_manifest_alias_equals_mods():
    import tomllib
    data = tomllib.loads((_EXT / "TableExpr.toml").read_text(encoding="utf-8"))
    assert data["name"] == "TableExpr"
    assert data["alias"] == data["mods"]
    assert len(data["alias"]) == 4
    assert set(data["alias"]) == {"tableexpr_app", "tableexpr_chain",
                                  "tableexpr_expr", "tableexpr_names"}


@pytest.mark.fast
def test_every_v0_name_is_reachable_from_nf():
    """§3: a V0 name is public iff it is reachable from nf.* / an nf class."""
    import numfast as nf
    app = nf.app()
    t = nf.from_numpy([1.0, 2.0], name="v")
    table_surface = {"column", "names", "ncols", "to_arrow", "to_numpy",
                     "to_pandas", "query", "schema", "c"}
    chain_surface = {"filter", "derive", "group", "reduce", "sort", "limit",
                     "compile", "jobs", "explain", "nrows"}
    expr_surface = {"add", "sub", "mul", "truediv", "mod", "pow",
                    "eq", "ne", "lt", "le", "gt", "ge", "and_", "or_",
                    "not_", "isin", "cumsum", "shift", "str_len",
                    "str_contains", "str_startswith", "str_endswith",
                    "str_eq"}
    reachable = set(nf.__all__) | table_surface | chain_surface | expr_surface
    reachable |= {"capabilities", "open_stream"}
    missing = sorted(V0 - reachable)
    assert not missing, missing
    assert app.c("v").kind == "col"


# --- the two loud guards (DESIGN §2.2b + GATE §9) -------------------------

def _table(**cols):
    import numpy as np
    import numfast as nf
    return nf.from_numpy(np.column_stack(
        [np.asarray(v) for v in cols.values()]).astype(np.float64),
        names=list(cols))


def _nullable_table():
    import numfast as nf
    import pandas as pd
    return nf.from_pandas(pd.DataFrame(
        {"plan": pd.array(["a", None, "b", None, "c"], dtype="string"),
         "rev": [1.0, 2.0, 3.0, 4.0, 5.0]}))


def _clean_table():
    import numfast as nf
    import pandas as pd
    return nf.from_pandas(pd.DataFrame(
        {"plan": pd.array(["a", "b", "a", "c", "b"], dtype="string"),
         "rev": [1.0, 2.0, 3.0, 4.0, 5.0]}))


@pytest.mark.fast
def test_group_refuses_null_key_loudly():
    import numfast as nf
    with pytest.raises(ValueError) as err:
        _nullable_table().query().group("plan", {"rev": ("sum",)}).compile()
    msg = str(err.value)
    assert "key column 'plan' has 2 NULL rows" in msg
    assert "silently dropped" in msg
    assert "Fix: filter those rows out before group()" in msg


@pytest.mark.fast
def test_sort_refuses_null_key_loudly():
    import numfast as nf
    with pytest.raises(ValueError) as err:
        _nullable_table().query().sort("plan").compile()
    msg = str(err.value)
    assert "key column 'plan' has 2 NULL rows" in msg
    assert "NULL rows first" in msg


@pytest.mark.fast
def test_null_free_group_and_sort_still_work():
    grouped = (_clean_table().query()
               .group("plan", {"rev": ("sum",)}).compile())
    assert grouped.column("plan").to_numpy().tolist() == ["a", "b", "c"]
    assert grouped.column("rev.sum").to_numpy().tolist() == [4.0, 7.0, 4.0]
    ordered = _clean_table().query().sort("plan").compile()
    assert ordered.column("plan").to_numpy().tolist() == [
        "a", "a", "b", "b", "c"]


@pytest.mark.fast
def test_guards_also_fire_on_a_numeric_nullable_key():
    """GATE §4.2 case: keys [1,2,0,1,3] with validity [T,T,F,T,T]."""
    import numfast as nf
    import numpy as np
    import pandas as pd
    frame = pd.DataFrame({"k": np.array([1, 2, 0, 1, 3], np.int32),
                          "v": [10.0, 20.0, 30.0, 40.0, 50.0]})
    nullable = frame.copy()
    nullable.loc[2, "k"] = None
    t = nf.from_pandas(nullable)
    with pytest.raises(ValueError, match="key column 'k' has 1 NULL rows"):
        t.query().group("k", {"v": ("sum",)})
    with pytest.raises(ValueError, match="key column 'k' has 1 NULL rows"):
        t.query().sort("k")
    clean = nf.from_pandas(frame)
    ordered = clean.query().sort("k").compile()
    assert ordered.column("k").to_numpy().tolist() == [0, 1, 1, 2, 3]
    grouped = clean.query().group("k", {"v": ("sum", "count")}).compile()
    assert grouped.column("k").to_numpy().tolist() == [0, 1, 2, 3]
    assert grouped.column("v.count").to_numpy().tolist() == [1, 2, 1, 1]
    assert grouped.column("v.sum").to_numpy().tolist() == [30.0, 50.0, 20.0, 50.0]


# --- the chain itself (07:60 vocabulary) -----------------------------------

@pytest.mark.fast
def test_jobs_are_ir_nodes_and_compile_runs_one_planner_path():
    import numfast as nf
    import numpy as np
    q = nf.app()
    t = nf.from_numpy(np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
                      names=["k", "v"])
    ch = t.query().filter(q.c("v") > 0).derive("d", q.c("v") * 2)
    jobs = ch.jobs()
    assert jobs and all({"op", "inputs", "params", "out"} <= set(j) for j in jobs)
    ops = [j["op"] for j in jobs]
    assert sorted(ops) == sorted(["series", "series", "compare", "filter",
                                  "filter", "map"])
    out = ch.compile()
    assert out.column("d").to_numpy().tolist() == [20.0, 40.0, 60.0]
    assert ch.nrows() == 3
    assert "GRAPH" in ch.explain()


@pytest.mark.fast
def test_reduce_and_sort_limit_and_group_topk():
    import numfast as nf
    import numpy as np
    q = nf.app()
    t = nf.from_numpy(np.array([[1, 10], [1, 20], [2, 35]], dtype=np.int32),
                      names=["k", "v"])
    assert t.query().reduce("sum", "v") == 65
    top = (t.query().group("k", {"v": ("sum",)}).sort("v.sum", desc=True)
           .limit(1).compile())
    assert top.column("k").to_numpy().tolist() == [2]
    assert top.column("v.sum").to_numpy().tolist() == [35]
    ordered = t.query().sort("v", desc=True).limit(2).compile()
    assert ordered.column("v").to_numpy().tolist() == [35, 20]


@pytest.mark.fast
def test_text_isin_never_matches_null_and_survives_a_null_free_column():
    import numfast as nf
    import pandas as pd
    q = nf.app()
    with_null = nf.from_pandas(pd.DataFrame(
        {"utm": pd.array(["ab", "cde", None, "fghij", "abcd", "a"],
                         dtype="string")}))
    kept = (with_null.query().filter(q.c("utm").isin(["a", "ab"])).compile())
    assert kept.column("utm").to_numpy().tolist() == ["ab", "abcd", "a"]
    null_free = nf.from_pandas(pd.DataFrame(
        {"utm": pd.array(["ab", "cde", "fghij", "abcd", "a"], dtype="string")}))
    kept2 = (null_free.query().filter(q.c("utm").isin(["a", "ab"])).compile())
    assert kept2.column("utm").to_numpy().tolist() == ["ab", "abcd", "a"]


@pytest.mark.fast
def test_text_group_key_source_carries_validity():
    """L3: the key source MUST pass validity= into ir_series.

    Without it `ir_groupby` folds the NULL row's measure into code 0 and
    overstates `count` (GATE_semantics.md §1.3). The guard refuses a NULL key
    outright, so the discipline is asserted on the emitted node itself.
    """
    import numfast as nf
    import pandas as pd
    q = nf.app()
    t = nf.from_pandas(pd.DataFrame(
        {"g": pd.array(["b", "a", None, "a", "c"], dtype="string"),
         "v": [10.0, 20.0, 30.0, 40.0, 50.0]}))
    with pytest.raises(ValueError):
        t.query().group("g", {"v": ("sum",)})
    filtered = t.query().filter(q.c("g").str_eq("a"))
    key_source = [j for j in filtered.jobs()
                  if j["op"] == "series" and j["params"].get("dtype") == "int32"]
    assert key_source, filtered.jobs()
    assert all(j["params"].get("validity") is not None for j in key_source)
    g = filtered.group("g", {"v": ("sum",)}).compile()
    assert g.column("g").to_numpy().tolist() == ["a"]
    assert g.column("v.sum").to_numpy().tolist() == [60.0]


def test_text_group_on_a_null_free_key():
    import numfast as nf
    import pandas as pd
    t = nf.from_pandas(pd.DataFrame(
        {"g": pd.array(["b", "a", "a", "c"], dtype="string"),
         "v": [10.0, 20.0, 40.0, 50.0]}))
    g = t.query().group("g", {"v": ("sum", "count")}).compile()
    assert g.column("g").to_numpy().tolist() == ["a", "b", "c"]
    assert g.column("v.sum").to_numpy().tolist() == [60.0, 10.0, 50.0]
    assert g.column("v.count").to_numpy().tolist() == [2, 1, 1]


@pytest.mark.fast
def test_app_surface_facts():
    import numfast as nf
    q = nf.app()
    caps = q.capabilities()
    assert "ops" in caps and "max_dispatch" in caps
    assert callable(q.open_stream)
    assert "TableExpr" in nf.get_kernel().metadata
    assert nf.get_kernel().metadata["TableExpr"]["version"] == "0.1.0"


# --- the Table KEY is the column name (GAP-1 silent-lie regression) ---------
#
# `nf.from_numpy(...)` names EVERY Series 'v' and `Series.name` is a read-only
# property, so a `Table` built as nf.Table(k, {"utm": ...}) used to lose the
# key: `col.name` became 'v', and the group output, both NULL-key guards and
# every node tag named a column that is not in the table. Values were correct,
# so the defect was silent. The key the user wrote is authoritative.


def _default_named_table(**cols):
    """Table whose Series ALL carry the default name 'v'."""
    import numfast as nf
    table = {name: nf.from_numpy(list(values)) for name, values in cols.items()}
    out = nf.Table(nf.get_kernel(), table)
    assert [out.column(n).name for n in out.names] == ["v"] * len(out.names)
    return out


@pytest.mark.fast
def test_group_text_key_keeps_the_table_key_not_the_series_name():
    t = _default_named_table(utm=["a", "b", "a"], price=[1.0, 2.0, 3.0])
    g = t.query().group("utm", {"price": ("sum", "count")}).compile()
    assert g.names == ["utm", "price.sum", "price.count"]
    assert g.column("utm").to_numpy().tolist() == ["a", "b"]
    assert g.column("price.sum").to_numpy().tolist() == [4.0, 2.0]
    assert g.column("price.count").to_numpy().tolist() == [2, 1]
    # the key the user wrote is readable on the result
    assert g.column("utm").name == "utm"


@pytest.mark.fast
def test_group_numeric_key_keeps_the_table_key_not_the_series_name():
    t = _default_named_table(gid=[7, 7, 8], v=[1.0, 2.0, 3.0])
    g = t.query().group("gid", {"v": ("sum",)}).compile()
    assert g.names == ["gid", "v.sum"]
    assert g.column("gid").to_numpy().tolist() == [7, 8]
    assert g.column("v.sum").to_numpy().tolist() == [3.0, 3.0]


@pytest.mark.fast
def test_group_null_key_guard_names_the_users_column():
    t = _default_named_table(plan=["a", "b", None, "a"], rev=[1.0, 2.0, 9.0, 1.0])
    with pytest.raises(ValueError) as err:
        t.query().group("plan", {"rev": ("sum",)}).compile()
    msg = str(err.value)
    assert "key column 'plan' has 1 NULL rows" in msg
    assert "'v'" not in msg
    assert "Fix: filter those rows out before group()" in msg


@pytest.mark.fast
def test_sort_null_key_guard_names_the_users_column():
    t = _default_named_table(plan=["a", "b", None, "a"], rev=[1.0, 2.0, 9.0, 1.0])
    with pytest.raises(ValueError) as err:
        t.query().sort("plan").compile()
    msg = str(err.value)
    assert "key column 'plan' has 1 NULL rows" in msg
    assert "'v'" not in msg
    assert "NULL rows first" in msg


@pytest.mark.fast
def test_two_default_named_source_columns_do_not_collide():
    """Both source Series are named 'v'; their node tags and group output
    must still be distinct and keyed by the Table keys."""
    import numfast as nf
    q = nf.app()
    t = _default_named_table(a=[1, 1, 2], b=[10.0, 20.0, 30.0])
    ch = t.query().filter(q.c("b") > 0)
    outs = [j["out"] for j in ch.jobs()]
    assert len(set(outs)) == len(outs), outs
    sources = [j["out"] for j in ch.jobs() if j["op"] == "series"]
    assert len(set(sources)) == 2, sources
    assert {o.split("_")[1] for o in sources} == {"a", "b"}, sources
    g = ch.group("a", {"b": ("sum", "count")}).compile()
    assert g.names == ["a", "b.sum", "b.count"]
    assert g.column("a").to_numpy().tolist() == [1, 2]
    assert g.column("b.sum").to_numpy().tolist() == [30.0, 30.0]


@pytest.mark.fast
def test_compile_round_trip_names_equal_table_keys():
    import numfast as nf
    q = nf.app()
    t = _default_named_table(a=[1, 2, 1], b=[10.0, 20.0, 30.0])
    # bare pass-through: every column is still a SOURCE Col, so the name
    # reaches nf.Series through _kernel_series(col.name).
    bare = t.query().compile()
    assert bare.names == ["a", "b"]
    assert [bare.column(n).name for n in bare.names] == ["a", "b"]
    ordered = t.query().sort("b", desc=True).compile()
    assert ordered.names == ["a", "b"]
    assert [ordered.column(n).name for n in ordered.names] == ["a", "b"]
    filtered = t.query().filter(q.c("b") > 15).compile()
    assert filtered.names == ["a", "b"]
    assert [filtered.column(n).name for n in filtered.names] == ["a", "b"]
    grouped = t.query().group("a", {"b": ("sum",)}).compile()
    assert grouped.names == ["a", "b.sum"]
    assert [grouped.column(n).name for n in grouped.names] == ["a", "b.sum"]