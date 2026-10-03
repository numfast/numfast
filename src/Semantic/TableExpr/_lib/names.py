# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""V0 -- the canonical public surface of the consumer facade.

PRIVATE file (07:36: PRIVATE = all of _lib). It is the *specification of
publicness*: a name is public iff it is listed here. Nothing else in the
Extension decides what is public.

44 names, derived from the normative `Semantic/TableExpr` vocabulary of
specs/core/07:60 (query/filter/derive/group/reduce, jobs/compile) plus the
boundary adapters and the expression vocabulary.

`window` is DELIBERATELY ABSENT (owner decision, DESIGN §3.3): the lowering
`ir_rolling_sum` silently loses exactness on int64 above 2**53
(Drivers/CPU/_lib/cpu.py:3270 casts to float64 BEFORE folding), so an
operation that lies quietly is not shipped. The fix is FROZEN
(CPU_Driver), therefore the operation is deferred, not "documented as is".

`or_` is DELIBERATELY ABSENT, on the same grounds: `ir_mask(..., 'or')`
produces a correct 3VL data vector but AND-s the two operands' validity
sides, so `ir_filter` drops every row either side was NULL on -- an empty
frame where Kleene keeps rows. The fix is FROZEN (IR + CPU_Driver). The
name stays on `Expr` as a loud refusal (`Expr.or_` raises) so the mistake is
a `ValueError`, never a silently empty result. `and_` and `not_` are correct
and stay in v0.

`is_null` is the 44th name and is a NULL TEST, not a negation: it reuses the
`material()` the two NULL-key guards already call and answers hard True/False
on every row, so it shares no path with `not_` (which is a correct 3VL
predicate negation and is unchanged). `fill_null` stays out of v0 -- it would
need `ir_where`, which `cpu_capability()['chunkable_hints']` does not carry,
and its chunkability is unresolved.
"""

# boundary adapters (6)
BOUNDARY = frozenset({
    "app",
    "from_arrow", "from_numpy",
    "to_arrow", "to_pandas", "to_numpy",
})

# App + Table + column reference factory (5)
APP_TABLE = frozenset({
    "capabilities", "open_stream",
    "query", "schema",
    "c",
})

# the 07:60 chain: query -> filter -> derive -> group -> reduce, plus
# sort/limit (row operators kept on the chain) and the terminal
# compile/jobs/explain/nrows (10)
CHAIN = frozenset({
    "filter", "derive", "group", "reduce", "sort", "limit",
    "compile", "jobs", "explain", "nrows",
})

# expression vocabulary (23)
EXPR = frozenset({
    "add", "sub", "mul", "truediv", "mod", "pow",
    "eq", "ne", "lt", "le", "gt", "ge", "and_", "not_",
    "isin", "is_null", "cumsum", "shift",
    "str_len", "str_contains", "str_startswith", "str_endswith", "str_eq",
})

V0 = frozenset(BOUNDARY | APP_TABLE | CHAIN | EXPR)

# recounted, not estimated: 6 + 5 + 10 + 23 = 44
assert len(BOUNDARY) == 6, sorted(BOUNDARY)
assert len(APP_TABLE) == 5, sorted(APP_TABLE)
assert len(CHAIN) == 10, sorted(CHAIN)
assert len(EXPR) == 23, sorted(EXPR)
assert len(V0) == 44, sorted(V0)
assert "window" not in V0
assert "or_" not in V0
# `is_null` is a null test; `not_` is the 3VL negation of a predicate. Two
# names, two questions -- this step must not let one answer for the other.
assert {"is_null", "not_"} <= V0

SURFACE_VERSION = "0.1.0"


def v0_names():
    """Sorted tuple of the canonical v0 public names (44)."""
    return tuple(sorted(V0))


def v0_surface():
    """frozenset copy of V0 -- the registry `check_no_internal_leak` reads."""
    return V0


def v0_surface_version():
    return SURFACE_VERSION