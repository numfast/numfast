# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""expr.py -- column references and the expression vocabulary (V0 Expr names).

An `Expr` is a tiny immutable tagged record. It carries NO engine knowledge:
`chain.py` walks it and `plan.py` lowers it. Keeping the two apart is what
stops the facade from becoming a second IR (DESIGN §8.3).

    col     -> a column reference (must resolve to a chain column)
    const   -> a literal scalar
    bin     -> ir_map      (add sub mul div floor_div mod pow)
    cmp     -> ir_compare  (== != < <= > >=)
    logic   -> ir_mask     (and not; `or` is NOT in v0 -- Expr.or_ refuses)
    isin    -> out-of-DAG dictionary pre-pass + ir_series(bool) + ir_filter
    text    -> ir_text_*   (length contains startswith endswith equals)
    scan    -> ir_cumsum / ir_shift
"""

BIN_OPS = ("add", "sub", "mul", "truediv", "mod", "pow")
CMP_OPS = ("eq", "ne", "lt", "le", "gt", "ge")
TEXT_OPS = ("str_len", "str_contains", "str_startswith", "str_endswith",
            "str_eq")
SCAN_OPS = ("cumsum", "shift")

_CMP_TO_NODE = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">",
                "ge": ">="}


class Expr:
    """Immutable expression node. Truth: never evaluated in Python."""

    __slots__ = ("kind", "name", "arg")

    def __init__(self, kind, name=None, arg=None):
        self.kind = kind
        self.name = name
        self.arg = arg

    # -- identity --------------------------------------------------------
    def __repr__(self):
        if self.kind == "col":
            return f"Expr(col {self.name!r})"
        if self.kind == "const":
            return f"Expr(const {self.arg!r})"
        if self.kind == "text":
            return f"Expr(text {self.name} {self.arg[0]} {self.arg[1]!r})"
        if self.kind == "isin":
            return f"Expr(isin {self.name} {list(self.arg)!r})"
        return f"Expr({self.kind} {self.arg!r})"

    # -- arithmetic (V0: add sub mul truediv mod pow) --------------------
    def add(self, other):
        return Expr("bin", arg=("add", self, other))

    def sub(self, other):
        return Expr("bin", arg=("sub", self, other))

    def mul(self, other):
        return Expr("bin", arg=("mul", self, other))

    def truediv(self, other):
        return Expr("bin", arg=("truediv", self, other))

    def mod(self, other):
        return Expr("bin", arg=("mod", self, other))

    def pow(self, other):
        return Expr("bin", arg=("pow", self, other))

    # -- comparison (V0: eq ne lt le gt ge) ------------------------------
    def eq(self, other):
        return Expr("cmp", arg=("eq", self, other))

    def ne(self, other):
        return Expr("cmp", arg=("ne", self, other))

    def lt(self, other):
        return Expr("cmp", arg=("lt", self, other))

    def le(self, other):
        return Expr("cmp", arg=("le", self, other))

    def gt(self, other):
        return Expr("cmp", arg=("gt", self, other))

    def ge(self, other):
        return Expr("cmp", arg=("ge", self, other))

    # -- logic (V0: and_ not_ ; or_ refuses loudly, §3.3) ----------------
    def and_(self, other):
        return Expr("logic", arg=("and", self, other))

    def or_(self, other):
        """REFUSES LOUDLY -- `or_` is not in v0 (owner decision, §3.3).

        `ir_mask(out, a, b, 'or')` computes a correct 3VL DATA vector
        (`[F,T,T]` for the Kleene fixture) but AND-s the two VALIDITY sides
        together (`[T,F,F]`), and `ir_filter` ANDs the mask validity in, so
        every row either operand was UNKNOWN about is dropped. The answer is
        an empty frame where Kleene / pandas keep rows 1 and 2 -- right
        shape, plausible numbers, no error. `and_` and `not_` are correct
        (verified, pinned by tests) and stay; only `or` is broken, and the
        fix is FROZEN (IR + CPU_Driver).
        """
        raise ValueError(
            "or_ is not in v0: ir_mask(..., 'or') AND-s the validities of its "
            "two operands, so ir_filter drops every row either side was NULL "
            "on -- filter((a>1)|(b>10)) returns an empty frame where Kleene "
            "keeps both rows. The 3VL data vector is right; the validity is "
            "AND-ed. The fix is FROZEN (IR + CPU_Driver), so the operation is "
            "deferred, not documented as is. Fix: use .and_() / .not_() "
            "(both verified correct), or rewrite as the two filters you "
            "actually mean. See DESIGN_consumer_api_v0.md 3.3.")

    def not_(self):
        return Expr("logic", arg=("not", self, None))

    # -- set / scan ------------------------------------------------------
    def isin(self, values):
        seq = list(values)
        if not seq:
            raise ValueError(
                f"isin on column {self.name!r} got an empty needle list. "
                "Fix: pass at least one value, or drop the predicate. "
                "See specs/core/07-builder-extension.md")
        for v in seq:
            if not isinstance(v, str):
                raise ValueError(
                    f"isin on text column {self.name!r} needs str values, got "
                    f"{type(v).__name__}. v0 lowers isin through the "
                    "dictionary path only. Fix: pass str values. "
                    "See specs/core/07-builder-extension.md")
        return Expr("isin", self.name, seq)

    def cumsum(self):
        return Expr("scan", self.name, ("cumsum", None))

    def shift(self, periods):
        return Expr("scan", self.name, ("shift", int(periods)))

    # -- text ------------------------------------------------------------
    def str_len(self):
        return Expr("text", self.name, ("str_len", None))

    def str_contains(self, substr):
        return Expr("text", self.name, ("str_contains", substr))

    def str_startswith(self, prefix):
        return Expr("text", self.name, ("str_startswith", prefix))

    def str_endswith(self, suffix):
        return Expr("text", self.name, ("str_endswith", suffix))

    def str_eq(self, value):
        return Expr("text", self.name, ("str_eq", value))

    # -- operators (same lowering, sugar over the named methods) ---------
    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return Expr("bin", arg=("add", other, self))

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return Expr("bin", arg=("sub", other, self))

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return Expr("bin", arg=("mul", other, self))

    def __truediv__(self, other):
        return self.truediv(other)

    def __rtruediv__(self, other):
        return Expr("bin", arg=("truediv", other, self))

    def __mod__(self, other):
        return self.mod(other)

    def __rmod__(self, other):
        return Expr("bin", arg=("mod", other, self))

    def __pow__(self, other):
        return self.pow(other)

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def __lt__(self, other):
        return self.lt(other)

    def __le__(self, other):
        return self.le(other)

    def __gt__(self, other):
        return self.gt(other)

    def __ge__(self, other):
        return self.ge(other)

    def __and__(self, other):
        return self.and_(other)

    __rand__ = __and__

    def __or__(self, other):
        return self.or_(other)      # refuses loudly -- `or` is not in v0

    __ror__ = __or__

    def __invert__(self):
        return self.not_()

    __hash__ = None


def ref(name):
    """Column reference -- what `q.c('price')` returns."""
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"column reference needs a non-empty str, got {name!r}. "
            "Fix: q.c('price'). See specs/core/07-builder-extension.md")
    return Expr("col", name)


def const(value):
    return Expr("const", arg=value)


def is_expr(obj):
    return isinstance(obj, Expr)


def _scalar_key(value):
    """Hashable, structural key for a literal operand."""
    if isinstance(value, bool):
        return ("b", value)
    if isinstance(value, int):
        return ("i", value)
    if isinstance(value, float):
        return ("f", value)
    if isinstance(value, str):
        return ("s", value)
    return ("o", repr(value))


def expr_key(expr):
    """Structural key of an expression -- the facade's own CSE key.

    The Planner's CSE merges nodes with equal (kernel_id, inputs, params) and
    REWRITES the duplicate's `out` name, so a facade node name is only stable
    while the chain never contains two such nodes. Memoizing binds by this key
    guarantees exactly that: identical expression -> one node, never two.
    """
    if isinstance(expr, Expr):
        kind = expr.kind
        if kind == "col":
            return ("col", expr.name)
        if kind == "const":
            return ("const", _scalar_key(expr.arg))
        if kind == "scan":
            return ("scan", expr.name, expr.arg)
        if kind == "text":
            return ("text", expr.name, expr.arg[0], _scalar_key(expr.arg[1]))
        if kind == "isin":
            return ("isin", expr.name, tuple(expr.arg))
        if kind == "bin":
            fn, left, right = expr.arg
            return ("bin", fn, expr_key(left), _operand_key(right))
        if kind == "cmp":
            op, left, right = expr.arg
            return ("cmp", op, expr_key(left), _operand_key(right))
        if kind == "logic":
            op, a, b = expr.arg
            return ("logic", op, expr_key(a), expr_key(b))
        return (kind, repr(expr))
    if isinstance(expr, str):
        return ("col", expr)
    return ("raw", type(expr).__name__, repr(expr))


def _operand_key(value):
    return expr_key(value) if isinstance(value, Expr) else _scalar_key(value)


def column_name(obj, op="derive"):
    """Accept an Expr column reference or a bare column name (str)."""
    if isinstance(obj, Expr):
        if obj.kind != "col":
            raise ValueError(
                f"{op} needs a column reference, got {obj!r}. "
                "Fix: pass q.c('name') or a column name.")
        return obj.name
    if isinstance(obj, str) and obj:
        return obj
    raise ValueError(
        f"{op} needs a column reference or a column name, got {obj!r}. "
        "Fix: pass q.c('name') or a column name.")


__all__ = ["Expr", "ref", "const", "is_expr", "column_name", "expr_key",
           "BIN_OPS", "CMP_OPS", "TEXT_OPS", "SCAN_OPS", "_CMP_TO_NODE"]