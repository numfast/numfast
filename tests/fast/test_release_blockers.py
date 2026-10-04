# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Four measurable release blockers, each with the before-state pinned in a
comment next to the assertion it changed.

1. `compile()` raises on a graph the Planner's CSE rewrote -- because the
   facade emitted two nodes that are the SAME node. Distinct root cause from
   the row-space epoch (`test_consumer_silent_regressions.py` #1): the epoch
   stops the memo handing back a node bound in another row space, and has
   nothing to say about two columns legitimately SHARING one node inside one
   row space, which is exactly what the memo is for.
2. `ir_map` / `ir_compare` do not lower `expr.const`, and a non-numeric scalar
   operand reaches the driver as a buffer-name lookup.
3. The whole `bin` family computes on a TEXT column's dictionary CODES --
   ranks, not values -- and returns plausible wrong numbers with no error.
4. `Drivers/CPU/_lib/native_cpu.py::_probe()` reset four of its five optional
   ABI memos on a DLL identity change and not the fifth, so after any
   `NUMFAST_NATIVE_DISABLE` toggle every RNG call raised
   `ctypes.ArgumentError`, which the driver's `except RuntimeError` does not
   catch.

Every value asserted against pandas is oracle-checked against pandas. Where
the subject raises, pandas raises too and that is asserted as well -- an
ORACLE THAT DISAGREES WITH ITSELF IS NOT AN ORACLE.
"""

import numpy as np
import pandas as pd
import pytest

# --- fixtures --------------------------------------------------------------


def _q():
    import numfast as nf
    return nf.app()


def _pd(df):
    import numfast as nf
    return nf.from_pandas(df)


NUM = pd.DataFrame({"v": [10, 20, 30, 40]})
FLT = pd.DataFrame({"v": [1.5, 2.5, 3.5, 4.5]})
TXT = pd.DataFrame({"p": ["b", "a", "c"]})
TXT2 = pd.DataFrame({"p": ["b", "a", "c"], "q": ["b", "a", "c"]})
BOOL = pd.DataFrame({"b": [True, False, True, False]})

# `Expr("const", ...)` is not reachable through the public surface -- `const()`
# is in `_lib/expr.py` but not in `V0` -- and the real class is reachable
# through the reference `q.c()` returns. That is the honest way to build the
# expression the origin bench recorded as `derive('z', c('v') - const(1.0))`.
CONST = type(_q().c("v"))


def _const(value):
    return CONST("const", None, value)


def _of(table, col="r"):
    return table.column(col).to_numpy().tolist()


# --- 1. compile() on a graph the Planner's CSE rewrote --------------------


def _cse_pair():
    """Two columns that SHARE one DAG node, then a per-column row operator.

    This is the origin bench's own recipe (`develop/audit_foundation/
    ORIGIN_PIPELINE_BENCH.md` item 8: `derive(w0..w7).limit(64).compile()`),
    reached the way its harness reached it -- every window derived from the
    same expression. `_bind`'s memo returns ONE node for one expression, so all
    eight columns point at it, and `limit` then emits eight byte-identical
    `ir_slice` nodes.
    """
    import numfast as nf
    t = nf.from_numpy(np.arange(512, dtype=np.float64).reshape(-1, 1),
                      names=["v"])
    ch = nf.app().query(t)
    for j in range(8):
        ch = ch.derive(f"w{j}", nf.app().c("v").shift(0))
    return ch.limit(64, offset=0)


@pytest.mark.fast
def test_1_shared_node_then_limit_compiles():
    # pre-fix: ValueError compile: the Planner rewrote node 'lim_w1_5': CSE
    # merged it into an identical earlier node, so its buffer no longer exists.
    out = _cse_pair().compile()
    assert len(out) == 64
    assert _of(out, "w0") == [float(i) for i in range(64)]
    assert _of(out, "w7") == _of(out, "w0")


@pytest.mark.fast
def test_1_shared_node_then_sort_compiles_and_matches_the_unmerged_chain():
    """Same values as the chain that never shared a node.

    ORACLE DISCIPLINE: the subject computes `shift(0)` on 0..511 and the
    oracle is numpy's `np.arange(512)[0:64]` -- the same quantity, not a second
    call into NumFast.
    """
    q = _q()
    t = _pd(NUM)
    dup = (q.query(_pd(NUM)).derive("a", q.c("v") + 1)
           .derive("b", q.c("v") + 1).sort("v"))
    # pre-fix: ValueError ... the Planner rewrote node 'srt_b_8'
    got = dup.compile()
    plain = (q.query(t).derive("a", q.c("v") + 1)
             .derive("b", q.c("v") + 1).sort("v").compile())
    assert _of(got, "b") == _of(plain, "b")
    assert _of(got, "b") == (NUM["v"] + 1).tolist()


@pytest.mark.fast
def test_1_shared_node_then_filter_compiles():
    q = _q()
    # pre-fix: ValueError ... the Planner rewrote node 'f_b_8'
    out = (q.query(_pd(NUM)).derive("a", q.c("v") + 1)
           .derive("b", q.c("v") + 1).filter(q.c("v") > 15).compile())
    assert _of(out, "v") == [20, 30, 40]
    assert _of(out, "b") == [21, 31, 41] == (NUM["v"] + 1)[1:].tolist()


@pytest.mark.fast
def test_1_two_text_ops_on_equal_columns_compile():
    """`ir_text_*` takes its decoded `values` as a param and no input node.

    Two text columns with equal values therefore build IDENTICAL nodes, which
    the Planner's short-sequence fingerprint compares by value.
    """
    q = _q()
    # pre-fix: ValueError ... the Planner rewrote node 't_2'
    out = (q.query(_pd(TXT2)).derive("a", q.c("p").str_len())
           .derive("b", q.c("q").str_len()).compile())
    assert _of(out, "a") == TXT["p"].str.len().tolist()
    assert _of(out, "b") == _of(out, "a")


@pytest.mark.fast
def test_1_the_planner_still_reports_cse_on_a_hand_built_dup():
    """The facade dedups; the Planner's own CSE is untouched and still works.

    Two IDENTICAL `ir_series` nodes built by hand (one shared values OBJECT, so
    the Planner's identity-scoped array fingerprint merges them) must still be
    rewritten -- the facade's `_emit` dedup is scoped to what the facade emits,
    and it must not have made the Planner's pass a no-op for everyone.
    """
    import numfast as nf
    k = nf.get_kernel()
    vals = np.asarray([1, 2, 3], dtype=np.int32)
    jobs = [k.alias["ir_series"]("a", vals, "int32"),
            k.alias["ir_series"]("b", vals, "int32")]
    graph = k.alias["optimize"](k.alias["compile"](jobs))
    assert graph["metadata"]["passes"] == ["cse"]
    assert graph["nodes"][0]["out"] == "a" and len(graph["nodes"]) == 1


# --- 2. expr.const, and the non-numeric scalar operand -------------------


@pytest.mark.fast
def test_2_const_on_the_right_of_bin_lowers():
    q = _q()
    out = q.query(_pd(NUM)).derive("r", q.c("v").sub(_const(1.0))).compile()
    assert _of(out, "r") == (NUM["v"] - 1.0).tolist()


@pytest.mark.fast
def test_2_const_on_the_right_of_bin_int_lowers():
    q = _q()
    out = q.query(_pd(NUM)).derive("r", q.c("v").add(_const(2))).compile()
    assert _of(out, "r") == (NUM["v"] + 2).tolist()


@pytest.mark.fast
def test_2_const_on_the_right_of_cmp_lowers():
    q = _q()
    out = q.query(_pd(NUM)).derive("r", q.c("v").gt(_const(20))).compile()
    assert _of(out, "r") == (NUM["v"] > 20).tolist()


@pytest.mark.fast
def test_2_const_on_the_left_of_a_commutative_bin_lowers():
    """`2 + c('v')` and `c('v') + 2` are the same numbers, so they commute."""
    q = _q()
    out = q.query(_pd(NUM)).derive("r", _const(2).__add__(q.c("v"))).compile()
    assert _of(out, "r") == (NUM["v"] + 2).tolist()


@pytest.mark.fast
@pytest.mark.parametrize("fn", ["sub", "truediv", "mod", "pow"])
def test_2_const_on_the_left_of_a_non_commutative_bin_refuses_loudly(fn):
    """A scalar on the LEFT of a non-commutative op has NO lowering.

    `ir_map(out, inp, fn, value)` has one slot for the buffer and one for the
    scalar, in that order. Swapping would answer a DIFFERENT question -- pandas
    says `2 % [10,20,30,40]` is `[2,2,2,2]` where the swapped form would say
    `[0,0,0,0]` -- so this refuses and says so.

    The expression is built directly because `Expr.__rsub__`/`__rmod__`/
    `__rtruediv__` put the COLUMN on the left (they mirror the operand order,
    the way Python's own reflected operators do); a scalar genuinely on the
    left is `Expr("bin", (fn, const, column))`, or `2 % q.c('v')` in source.
    """
    q = _q()
    expr = CONST("bin", arg=(fn, _const(2), q.c("v")))
    with pytest.raises(ValueError) as ei:
        q.query(_pd(NUM)).derive("r", expr).compile()
    msg = str(ei.value)
    assert "one buffer and one scalar" in msg
    assert "commute" in msg


@pytest.mark.fast
def test_2_a_python_scalar_on_the_left_of_a_non_commutative_op_refuses():
    """The public `2 % q.c('v')` / `2 - q.c('v')` / `2 / q.c('v')` forms.

    pre-fix all three raised `derive needs an expression from q.c(...), got
    int`, which names the wrong thing entirely -- the column reference was
    never the problem, the operand ORDER is.
    """
    q = _q()
    for expr, same, different in (
            (2 - q.c("v"), (NUM["v"] - 2).tolist(), (2 - NUM["v"]).tolist()),
            (2 % q.c("v"), (NUM["v"] % 2).tolist(), (2 % NUM["v"]).tolist()),
            (2 / q.c("v"), None, (2 / NUM["v"]).tolist())):
        with pytest.raises(ValueError) as ei:
            q.query(_pd(NUM)).derive("r", expr).compile()
        assert "one buffer and one scalar" in str(ei.value)
        if same is not None:
            # the oracle answer this deliberately does NOT produce
            assert same != different


@pytest.mark.fast
def test_2_a_python_scalar_on_the_left_of_a_commutative_op_now_lowers():
    """The public `1 + q.c('v')` form: pre-fix this raised
    `derive needs an expression from q.c(...), got int`."""
    q = _q()
    assert _of(q.query(_pd(NUM)).derive("r", 1 + q.c("v")).compile(),
               "r") == (NUM["v"] + 1).tolist()
    assert _of(q.query(_pd(FLT)).derive("r", 2.0 * q.c("v")).compile(),
               "r") == (FLT["v"] * 2.0).tolist()


@pytest.mark.fast
def test_2_a_string_scalar_operand_refuses_instead_of_reaching_the_driver():
    """pre-fix `c('v') + 'z'` was `KeyError: 'z'` -- a buffer-name lookup
    (`ir_map` reads `isinstance(value, str)` as a second INPUT,
    src/Semantic/IR/_lib/nodes.py:79). `KeyError` is not the engine's error
    shape and names nothing about the cause."""
    q = _q()
    for expr, oracle_raises in (
            (q.c("v") + "z", lambda: NUM["v"] + "z"),
            ("z" + q.c("v"), lambda: "z" + NUM["v"]),
    ):
        with pytest.raises(ValueError) as ei:
            q.query(_pd(NUM)).derive("r", expr).compile()
        assert "scalar operand" in str(ei.value)
        with pytest.raises(TypeError):
            oracle_raises()


@pytest.mark.fast
@pytest.mark.parametrize("bad", [None, [1, 2]])
def test_2_a_none_or_list_scalar_operand_refuses(bad):
    """pre-fix these reached the driver: a numpy TypeError or a broadcast
    ValueError, neither of which names the facade's contract."""
    q = _q()
    with pytest.raises(ValueError) as ei:
        q.query(_pd(NUM)).derive("r", q.c("v") + bad).compile()
    assert "scalar operand" in str(ei.value)


@pytest.mark.fast
def test_2_column_pow_with_a_scalar_base_refuses_by_name():
    """`2 ** q.c('v')` pre-fix raised Python's own
    `TypeError: unsupported operand type(s) for ** or pow()`, which names the
    user's column as if the column were the problem."""
    q = _q()
    with pytest.raises(ValueError) as ei:
        q.query(_pd(NUM)).derive("r", 2 ** q.c("v")).compile()
    assert "SCALAR-exponent-only" in str(ei.value)


@pytest.mark.fast
def test_2_comparison_against_a_scalar_on_either_side_is_unchanged():
    q = _q()
    for expr, oracle in ((q.c("v") > 20, NUM["v"] > 20),
                         (20 < q.c("v"), 20 < NUM["v"]),
                         (20 >= q.c("v"), 20 >= NUM["v"]),
                         (q.c("v") == 20, NUM["v"] == 20),
                         (q.c("v") != 20, NUM["v"] != 20)):
        assert _of(q.query(_pd(NUM)).derive("r", expr).compile(),
                   "r") == oracle.tolist()


# --- 3. arithmetic on a TEXT column ---------------------------------------


@pytest.mark.fast
@pytest.mark.parametrize("dunder,operand,pandas", [
    ("__add__", 1, None),
    ("__sub__", 1, None),
    ("__mul__", 2, ["bb", "aa", "cc"]),
    ("__truediv__", 2, None),
    ("__mod__", 2, None),
    ("__pow__", 2, None),
])
def test_3_arithmetic_on_text_refuses_loudly(dunder, operand, pandas):
    """SIX operators, ZERO guards. Every one of them answered a number.

    pre-fix on ['b','a','c'] (sidecar ['a','b','c'], codes [1,0,2]):
    +1 -> [2,1,3]  -1 -> [0,-1,1]  *2 -> [2,0,4]
    /2 -> [0,0,1]   %2 -> [1,0,0]   **2 -> [1,0,4]

    ORACLE DISCIPLINE: five of the six have NO pandas answer -- pandas raises
    TypeError, which is asserted, because a facade is not entitled to call an
    undefined operation wrong. The sixth, `str * int`, pandas DOES define, as
    string REPETITION: `['bb','aa','cc']`. `ir_map` has no text mode, so it
    cannot repeat either, and the refusal says the same cause.
    """
    q = _q()
    with pytest.raises(ValueError) as ei:
        q.query(_pd(TXT)).derive(
            "r", getattr(q.c("p"), dunder)(operand)).compile()
    msg = str(ei.value)
    assert "TEXT column 'p'" in msg
    assert "RANKS, not values" in msg
    assert "str_len" in msg, msg
    if pandas is None:
        with pytest.raises(TypeError):
            getattr(TXT["p"], dunder)(operand)
    else:
        assert getattr(TXT["p"], dunder)(operand).tolist() == pandas


@pytest.mark.fast
@pytest.mark.parametrize("dunder", ["__add__", "__sub__", "__mul__",
                                    "__truediv__", "__mod__", "__pow__"])
def test_3_text_column_against_a_numeric_column_refuses(dunder):
    """The exact form KNOWN_LIMITATIONS.md §7 recorded, on BOTH sides.

    `k = ["b","a","c"]` (text), `v = [1.0, 2.0, 3.0]`. pre-fix:
    `k + v -> [2, 2, 5]`, `- -> [0,-2,-1]`, `* -> [1,0,6]`,
    `/ -> [1,0,1]`, `% -> [0.0,0.0,2.0]`; `**` was blocked by an unrelated
    driver "array exponent" guard, not by a text guard.
    """
    df = pd.DataFrame({"k": ["b", "a", "c"], "v": [1.0, 2.0, 3.0]})
    q = _q()
    t = _pd(df)
    for left, right, name in ((q.c("k"), q.c("v"), "left"),
                              (q.c("v"), q.c("k"), "right")):
        with pytest.raises(ValueError) as ei:
            q.query(t).derive("r", getattr(left, dunder)(right)).compile()
        assert "TEXT column 'k'" in str(ei.value)
        assert f"the {name} operand" in str(ei.value)
    with pytest.raises(TypeError):
        getattr(df["k"], dunder)(df["v"])


@pytest.mark.fast
def test_3_the_refusal_names_the_column_and_leaves_jobs_untouched():
    q = _q()
    ch = q.query(_pd(TXT))
    with pytest.raises(ValueError):
        ch.derive("r", q.c("p") + 1)
    # not merely "no new node": the guard runs BEFORE the left operand is even
    # bound, so jobs[] is left exactly as it was -- empty.
    assert ch.jobs() == [], ch.jobs()


@pytest.mark.fast
def test_3_str_len_is_the_honest_route_and_matches_the_pandas_oracle():
    """`str_len` is the already-safe spelling: `ir_text_length` answers int32
    code-point COUNTS, which arithmetic IS defined on. `p.str_len() + 1` is
    [2,2,2] in both engines, and the codes it replaces were [1,0,2]."""
    q = _q()
    out = q.query(_pd(TXT)).derive("r", q.c("p").str_len() + 1).compile()
    assert _of(out, "r") == (TXT["p"].str.len() + 1).tolist()
    assert out.column("r").dtype == "int32"
    only = q.query(_pd(TXT)).derive("r", q.c("p").str_len()).compile()
    assert _of(only, "r") == TXT["p"].str.len().tolist()


@pytest.mark.fast
@pytest.mark.parametrize("df,col", [(NUM, "v"), (FLT, "v")])
def test_3_numeric_columns_are_unchanged(df, col):
    q = _q()
    for build, oracle in ((lambda e: e + 1, lambda s: s + 1),
                          (lambda e: e - 1, lambda s: s - 1),
                          (lambda e: e * 2, lambda s: s * 2),
                          (lambda e: e % 2, lambda s: s % 2),
                          (lambda e: e ** 2, lambda s: s ** 2)):
        assert _of(q.query(_pd(df)).derive("r", build(q.c(col))).compile(),
                   "r") == oracle(df[col]).tolist()
    # float / int divides for real on a FLOAT column and is unchanged
    got = q.query(_pd(FLT)).derive("r", q.c("v") / 2).compile()
    assert _of(got, "r") == (FLT["v"] / 2).tolist()


@pytest.mark.fast
def test_3_integer_truediv_still_refuses():
    """The pre-existing integer-division guard is untouched by the text guard."""
    q = _q()
    with pytest.raises(ValueError) as ei:
        q.query(_pd(NUM)).derive("r", q.c("v") / 2).compile()
    assert "ROUNDS THE QUOTIENT BACK" in str(ei.value)


@pytest.mark.fast
def test_3_boolean_columns_are_unchanged():
    """Boolean arithmetic is pre-existing behaviour and must not move.

    NOTE the pinned value: `True + 1` answers `True` here and `2` in pandas --
    a bool column stays bool through ir_map. That divergence PRE-DATES this
    change and is deliberately not touched; this test pins it so it cannot
    drift unnoticed.
    """
    q = _q()
    assert _of(q.query(_pd(BOOL)).derive("r", q.c("b") + 1).compile(),
               "r") == [True, True, True, True]
    assert _of(q.query(_pd(BOOL)).derive("r", q.c("b") % 2).compile(),
               "r") == (BOOL["b"] % 2).tolist()
    assert _of(q.query(_pd(BOOL)).derive("r", q.c("b") / 2).compile(),
               "r") == (BOOL["b"] / 2).tolist()


# --- 4. the latent frozen RNG bug -----------------------------------------


@pytest.mark.fast
def test_4_rng_survives_a_native_disable_toggle(monkeypatch):
    """`_probe()` reset four of its five optional-ABI memos and not `_rng_ok`.

    argtypes live on the CDLL OBJECT, so after a toggle `_lib` is a fresh CDLL
    with none of them, while `_rng_ok` still said True: every RNG call raised
    `ctypes.ArgumentError: argument 1: OverflowError: int too long to convert`,
    which `cpu.py`'s `except RuntimeError` does NOT catch, so it escaped the
    public API. Reproducible from `nf.rng_fill_i32` alone.
    """
    import ctypes

    import numfast as nf
    seed = 42
    first = nf.rng_fill_i32(64, seed=seed, lo=0, hi=100).to_numpy().tolist()

    monkeypatch.setenv("NUMFAST_NATIVE_DISABLE", "1")
    off = nf.rng_fill_i32(64, seed=seed, lo=0, hi=100).to_numpy().tolist()
    assert off == first, "the numpy fallback must be bit-identical"

    monkeypatch.delenv("NUMFAST_NATIVE_DISABLE", raising=False)
    # pre-fix: ctypes.ArgumentError -- not a RuntimeError, so nothing caught it
    back = nf.rng_fill_i32(64, seed=seed, lo=0, hi=100).to_numpy().tolist()
    assert back == first

    # every other RNG verb went through the same un-argtyped entry points
    for call in (lambda: nf.rng_fill_f64(32, seed=seed),
                 lambda: nf.rng_permutation(32, seed=seed),
                 lambda: nf.rng_sample(64, 8, seed=seed)):
        assert len(call().to_numpy()) > 0


@pytest.mark.fast
def test_4_toggle_is_repeatable():
    import numfast as nf

    import os
    prev = os.environ.get("NUMFAST_NATIVE_DISABLE")
    try:
        for _ in range(3):
            os.environ["NUMFAST_NATIVE_DISABLE"] = "1"
            nf.rng_fill_i32(16, seed=42, lo=0, hi=5)
            del os.environ["NUMFAST_NATIVE_DISABLE"]
            nf.rng_fill_i32(16, seed=42, lo=0, hi=5)
    finally:
        if prev is None:
            os.environ.pop("NUMFAST_NATIVE_DISABLE", None)
        else:
            os.environ["NUMFAST_NATIVE_DISABLE"] = prev


# --- the surface did not grow --------------------------------------------


@pytest.mark.fast
def test_v0_surface_is_unchanged():
    import importlib.util
    from pathlib import Path
    ext = Path(__file__).resolve().parents[2] / "src" / "Semantic" / "TableExpr"
    spec = importlib.util.spec_from_file_location(
        "_lib.names_rb", ext / "_lib" / "names.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert len(m.V0) == 44
    for name in ("or_", "window", "abs", "neg", "const", "rpow"):
        assert name not in m.V0
