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
    "eq", "ne", "lt", "le", "gt", "ge", "and_", "or_", "not_",
    "isin", "cumsum", "shift",
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

SURFACE_VERSION = "0.1.0"


def v0_names():
    """Sorted tuple of the canonical v0 public names (44)."""
    return tuple(sorted(V0))


def v0_surface():
    """frozenset copy of V0 -- the registry `check_no_internal_leak` reads."""
    return V0


def v0_surface_version():
    return SURFACE_VERSION