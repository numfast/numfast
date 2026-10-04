# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""chain.py -- Table -> query -> filter -> derive -> group -> reduce -> compile.

The 07:60 vocabulary, implemented as a lazy chain over one IR node per
column. Every op appends nodes to `jobs[]`; `compile()` makes exactly ONE
planner call over that list and executes it on the CPU oracle.

Three honest deviations (1, 2, 4), one measured non-deviation (3) and one
contract note (5), all recorded in DESIGN §2.2a / §2.2b / §3.3:

1. `group()` is TWO graphs. `ir_groupby_multi(result='carry')` returns a
   ColumnCarry, which is not a column and cannot re-enter the DAG. So
   `group()` runs the grouping graph, materializes the group result, and the
   following ops build the second graph. That second graph IS the top-k
   recipe (sort -> slice -> gather), so `topk` inherits `group()`'s
   precondition for free.
2. `group()` REFUSES LOUDLY on a NULL in any key column, before
   `groupby_multi` is called. `ir_groupby*` compacts NULL keys away
   (Drivers/CPU/_lib/cpu.py:3325/:3507), so a NULL key is not a wrong
   value, it is a MISSING value, and the consumer cannot tell the difference
   from the answer. The check reads validity OUTSIDE the graph
   (`Series.validity` / `dictionary_encode(...)['validity']`), so no FROZEN
   component changes.
3. `sort()` KEEPS a NULL key and puts the row LAST, in input order. This is
   NOT a deviation -- it is the measured behaviour of `ir_sort`, and it is
   what pandas `sort_values` does (`na_position="last"`, stable). The CPU
   driver partitions the permutation explicitly (`_sort_perm`,
   Drivers/CPU/_lib/cpu.py:1228-1229: `perm[:n_valid] = ...`,
   `perm[n_valid:] = invalid positions`) and the GPU driver does the same
   (`gpu.py:3149`: `concatenate([vpos[sub], ipos])`). GATE_semantics.md §9
   inferred "NULL rows go FIRST" from the rendered `[1, 2, 3, 0]`, which is
   its own evidence of the opposite; the §9 note was corrected on
   2026-10-04 and the false refusal it produced was removed. Parity is pinned
   by `test_sort_null_key_matches_the_pandas_oracle`.

4. `_bind`'s memo is scoped to ONE row space. `filter`, `sort` and `limit`
   rebuild every column node into a different row space (new order or new
   count), so a cached `(expr_key) -> node` entry from before the op still
   names the PRE-op node -- the rebind would return pre-sort data with the
   post-sort column beside it, which is a wrong answer with the right shape
   and no exception (silent). The memo therefore carries a row-space
   epoch: every op that changes row order or row count bumps it, so a
   rebind in a new row space misses the cache instead of hitting it. The
   memo still does its real job inside one row space -- see `_spawn`.

5. `is_null()` is the guard's own reader, returned instead of refused.
   Deviation 2 reads validity OUTSIDE the graph (`material()`) in order to
   raise. `is_null()` reuses that exact helper and turns the answer into an
   `ir_series(bool)` (`_bind_is_null`), so the §2.2b advice "filter those rows
   out before group()" is executable from the public surface. It is NOT
   `not_()`: `not_` is a 3VL negation of a predicate and is correct as it
   stands; `is_null` is a null test whose answer is hard True/False on every
   row, so the mask carries no validity sidecar.

`window` is NOT here and must not be added: `ir_rolling_sum` loses exactness
silently on int64 above 2**53 (DESIGN §3.3).

`or_` is NOT here either and must not be added: `ir_mask(..., 'or')` AND-s the
two operands' validity sides, so `ir_filter` silently drops every row either
side was UNKNOWN on (DESIGN §2.2a п. 8). `Expr.or_` refuses loudly -- the
`logic` branch below only ever sees `and` / `not`.
"""

import numpy as np

from _lib import plan
from _lib import stream as _stream
from _lib.expr import Expr, _CMP_TO_NODE, column_name, expr_key, ref

_NUMERIC_LOGICAL = ("int32", "int64", "float32", "float64", "bool")

# A scalar operand ir_map / ir_compare can actually carry. `bool` is a
# subclass of `int`, so it is deliberately IN: pandas defines `True + 1`, and
# refusing it would be a refusal of something the oracle answers.
_NUMERIC_SCALARS = (bool, int, float)

# The two MAP_FN entries whose operands commute. `sub`, `truediv`, `floor_div`,
# `mod` and `pow` answer a DIFFERENT question if the two are swapped, so a
# scalar on the LEFT of one of those has no lowering at all.
_COMMUTATIVE_BIN = ("add", "mul")

# Every comparison has an exact mirror, so a scalar on the LEFT of a `cmp` is
# normalised by flipping rather than refused: `2 < c('v')` IS `c('v') > 2`.
_CMP_MIRROR = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le",
               "eq": "eq", "ne": "ne"}


class Col:
    """One column of the chain: DAG node, or host values, or both."""

    __slots__ = ("name", "dtype", "node", "values", "validity", "sidecar")

    def __init__(self, name, dtype, node=None, values=None, validity=None,
                 sidecar=None):
        self.name = name
        self.dtype = dtype
        self.node = node
        self.values = values
        self.validity = validity
        self.sidecar = sidecar

    def __repr__(self):
        return f"Col({self.name!r}, {self.dtype}, node={self.node!r})"


# --- column construction ---------------------------------------------------

def _kernel_table(kernel, cols):
    import numfast as nf
    return nf.Table(kernel, cols)


def _text_series(kernel, name, values, validity=None):
    """[str|None] -> numfast Series on the dictionary path."""
    import numfast as nf
    enc = plan.prepass("dictionary_encode")(
        list(values), plan.as_validity(validity))
    sidecar = list(enc["values"]) or [""]
    codes = np.ascontiguousarray(np.asarray(enc["codes"], dtype=np.int32))
    enc_valid = plan.as_validity(enc["validity"])
    return nf.Series(kernel, name, codes, "text", validity=enc_valid,
                     sidecar={"values": sidecar})


def _kernel_series(kernel, col, values, validity, backend="cpu"):
    """Buffer + column record -> numfast Series (public boundary type).

    `backend` is the executor that produced `values` -- `compile()` reads
    cpu_execute buffers -- and it is carried onto the Series so a result says
    where it came from. `None` for the text path, whose values come from the
    out-of-graph dictionary pre-pass rather than from any executor.
    """
    import numfast as nf
    buf = np.ascontiguousarray(np.asarray(values))
    valid = plan.as_validity(validity)
    if col.dtype == "text":
        backend = None
        if buf.dtype == object or buf.dtype.kind in "US":
            return _text_series(kernel, col.name, buf.tolist(), valid)
        sidecar = list(col.sidecar or []) or [""]
        return nf.Series(kernel, col.name,
                         buf.astype(np.int32, copy=False), "text",
                         validity=valid, sidecar={"values": sidecar})
    logical = plan.logical_of(buf) or col.dtype
    if logical not in _NUMERIC_LOGICAL:
        raise plan.fail(
            "compile",
            f"column {col.name!r} came back as dtype {buf.dtype}, which has "
            "no logical mapping.",
            "use int32/int64/float32/float64/bool columns")
    return nf.Series(kernel, col.name, buf, logical, validity=valid,
                      backend=backend)


def _source_col(kernel, name, series):
    """Table key + numfast Series -> Col, using PUBLIC observers only.

    `name` is the Table KEY, and the Table key is the authoritative name.
    `Series.name` is a read-only property that defaults to 'v' for every
    `nf.from_numpy(...)` column, so reading it would discard the name the
    user actually wrote in the `Table` constructor dict -- and every
    downstream reader of `col.name` (node tags, the two NULL-key guards,
    group output naming) would then report a column that is not in the
    table.
    """
    dtype = series.dtype
    validity = series.validity
    if dtype == "text":
        values = np.asarray(series.to_numpy(), dtype=object)
    else:
        values = np.ascontiguousarray(np.asarray(series.to_masked().data))
    return Col(name, dtype, None, values=values, validity=validity)


def _single_key(keys):
    if isinstance(keys, str):
        return keys
    seq = list(keys) if isinstance(keys, (list, tuple)) else []
    if len(seq) == 1 and isinstance(seq[0], str):
        return seq[0]
    if not seq:
        raise plan.fail(
            "group", "keys is empty.",
            "group('utm', {'price': ('sum',)})")
    raise plan.fail(
        "group",
        f"v0 groups by exactly one key column, got {seq!r}. Composite keys "
        "have no TableExpr lowering (DESIGN §2.2a).",
        "group by one column, or pack the keys yourself")


# --- the chain -------------------------------------------------------------

class Chain:
    """Lazy query chain over one IR node per column."""

    def __init__(self, kernel, cols, nrows, jobs=None, pending=None, seq=0,
                 bufs=None, memo=None, space=0, emitted=None):
        self._kernel = kernel
        self._cols = dict(cols)
        self._order = list(self._cols)
        self._n = int(nrows)
        self._jobs = list(jobs or ())
        self._pending = pending
        self._seq = int(seq)
        self._bufs = bufs
        self._memo = dict(memo or {})
        # row-space epoch -- bumped by every op that changes row order or row
        # count. It is part of the memo key, so the cache is valid inside one
        # row space and unreachable in the next one.
        self._space = int(space)
        # node identity -> node name, for every node this chain has emitted.
        # NOT the row-space epoch's job: this is about the Planner's CSE, which
        # sees IR nodes and not expressions. See `_emit`.
        self._emitted = dict(emitted or {})

    # -- construction ---------------------------------------------------
    @classmethod
    def from_table(cls, kernel, table):
        import numfast as nf
        if not isinstance(table, nf.Table):
            raise plan.fail(
                "query",
                f"query() needs a numfast Table, got {type(table).__name__}.",
                "nf.from_arrow(...) / nf.from_pandas(...) / nf.from_numpy(2d)")
        cols = {}
        for name in table.names:
            cols[name] = _source_col(kernel, name, table.column(name))
        return cls(kernel, cols, len(table))

    # -- plumbing -------------------------------------------------------
    def _next(self, tag):
        self._seq += 1
        return f"{tag}_{self._seq}"

    def _emit(self, op, out, *args, **kw):
        """Append one node to jobs[] and return the name that survived.

        RETURNS THE EFFECTIVE `out`, which is not always the one passed in: the
        Planner's CSE merges nodes with equal (kernel_id, inputs, params) and
        REWRITES the duplicate's `out` to the earlier node's name, so a graph
        with two such nodes has no buffer under the second name and `compile()`
        raises (`_buffer`). Every caller must use the returned name.

        This is NOT the row-space epoch's job, and it is a DIFFERENT root cause
        from it. The epoch (deviation 4) stops the EXPRESSION MEMO from handing
        back a node bound in a different row space; this stops the facade from
        emitting two IR nodes that are the same node. The memo cannot do it: it
        keys on expression structure, which is strictly finer than node
        identity, and returning one node for one expression is exactly what
        makes two chain columns SHARE a node -- at which point `filter`, `sort`
        and `limit`, which each emit one node per column, emit the shared node
        twice. `ir_text_*` walks into the same room from the other side: it
        carries its decoded `values` as a param and takes no input node, so two
        text columns with equal values build identical nodes.

        Reusing an identical node is value-preserving by the Planner's own
        definition of identity, and it needs no new node, no new op and no
        FROZEN change. `plan.node_identity` carries the argument, including
        which direction it is allowed to be wrong in.
        """
        job = plan.node(op, out, *args, **kw)
        ident = plan.node_identity(job)
        prior = self._emitted.get(ident)
        if prior is not None:
            return prior
        self._jobs.append(job)
        self._emitted[ident] = out
        self._bufs = None
        return out

    def _branch(self):
        """Detach the job list: a chain stays reusable after an op."""
        self._jobs = list(self._jobs)
        return self

    def _spawn(self, cols, nrows=None, pending=None, new_space=False):
        """Child chain. `new_space=True` for every op that changes row order
        or row count -- it bumps the epoch, so the child's binds miss the
        parent's memo instead of reusing a node bound in another row space.
        `derive` does NOT pass it: it adds a column and keeps the rows.
        """
        return Chain(self._kernel, cols, self._n if nrows is None else nrows,
                     self._jobs, pending, self._seq, self._bufs, self._memo,
                     self._space + 1 if new_space else self._space,
                     self._emitted)

    def _col(self, name, op="query"):
        col = self._cols.get(name)
        if col is None:
            raise plan.fail(
                op, f"no column {name!r} (have {self._order}).",
                "use q.c() with one of the existing column names")
        return col

    # -- declared dtype, WITHOUT emitting a node ---------------------------
    def _dtype_of(self, expr, op="query"):
        """Declared logical dtype of `expr`, or None when it is not knowable.

        Read-only on purpose: the guards below have to decide BEFORE any node
        goes into jobs[], so they cannot ask `_bind` for it. `_dtype_of`
        answers only what the chain's own column records already say and
        returns None for `bin` / nested-derived expressions -- those carry no
        declared dtype, and `None` is what makes the callers refuse rather
        than guess. Guessing here would be the whole defect class this facade
        exists to remove.
        """
        if isinstance(expr, str):
            return self._col(expr, op).dtype
        if isinstance(expr, Expr):
            if expr.kind == "col":
                return self._col(expr.name, op).dtype
            if expr.kind == "scan":
                scan_op, *rest = expr.arg
                source = expr.arg[2] if len(expr.arg) == 3 else expr.name
                return self._dtype_of(source, op)
            if expr.kind == "text":
                return "int32" if expr.arg[0] == "str_len" else "bool"
            if expr.kind == "cmp" or expr.kind == "logic":
                return "bool"
            if expr.kind == "isin":
                return "bool"
            if expr.kind == "null":
                return "bool"
        return None

    def _buffers(self):
        if self._bufs is None:
            self._bufs = plan.buffers_of(self._jobs)
        return self._bufs

    def _buffer(self, node):
        bufs = self._buffers()
        try:
            return bufs[node], bufs.get(node + "#validity")
        except KeyError:
            raise plan.fail(
                "compile",
                f"the Planner rewrote node {node!r}: CSE merged it into an "
                "identical earlier node, so its buffer no longer exists.",
                "file this against Semantic/TableExpr -- the facade memoizes "
                "expression binds and this node had no unique form") from None

    # -- column realization ---------------------------------------------
    def _codes_node(self, col):
        """int32 dictionary codes of a TEXT column, as a DAG source node.

        This is the §2.2 pre-pass: `dictionary_encode` OUTSIDE the graph,
        then `ir_series(codes, 'int32', validity=)`. Passing `validity=` is
        mandatory -- without it `ir_groupby` folds the NULL row's measure
        into group 0 and overstates `count` (GATE_semantics.md §1.3, L3).
        """
        if col.node is not None:
            return col.node
        enc = plan.prepass("dictionary_encode")(list(col.values), col.validity)
        out = self._next(f"c_{col.name}")
        codes = np.ascontiguousarray(np.asarray(enc["codes"], dtype=np.int32))
        enc_valid = plan.as_validity(enc["validity"])
        if enc_valid is None:
            out = self._emit("series", out, codes, "int32")
        else:
            out = self._emit("series", out, codes, "int32", enc_valid)
        col.node = out
        col.sidecar = list(enc["values"])
        col.values = None
        col.validity = None
        return out

    def _need(self, col):
        """DAG node of a column (sources are materialized on demand)."""
        if col.dtype == "text":
            return self._codes_node(col)
        if col.node is not None:
            return col.node
        out = self._next(f"s_{col.name}")
        values = np.ascontiguousarray(np.asarray(col.values))
        if col.validity is None:
            out = self._emit("series", out, values, col.dtype)
        else:
            out = self._emit("series", out, values, col.dtype, col.validity)
        col.node = out
        return out

    def material(self, col):
        """(values, validity) of a column in the CURRENT row space."""
        if col.node is None:
            return col.values, col.validity
        buf, validity = self._buffer(col.node)
        return plan.as_array(buf), plan.as_validity(validity)

    def _text_values(self, col):
        """Decoded [str|None] of a TEXT column in the current row space.

        A source column already holds decoded values; once the column has a
        DAG node it holds int32 codes, and the sidecar recorded by
        `_codes_node` decodes them.
        """
        if col.node is None:
            return list(col.values)
        codes, validity = self.material(col)
        return plan.decode_text(codes, col.sidecar or [""], validity)

    # -- the one loud guard ----------------------------------------------
    def _refuse_null_key(self, col):
        _, validity = self.material(col)
        n_null = plan.null_count(validity)
        if n_null:
            raise plan.fail(
                "group",
                f"key column {col.name!r} has {n_null} NULL rows; v0 does "
                "not group NULL keys -- NULL keys are silently dropped "
                "(Drivers/CPU/_lib/cpu.py:3325), so a NULL group is an "
                "ABSENT value, not a wrong one, and cannot be told apart "
                "from the answer.",
                "filter those rows out before group()")

    # -- the text-comparison and integer-division guards -------------------
    def _cmp_guard(self, cmp_op, left, right, op):
        """Every way a `cmp` can reach a text column or a bare string.

        Measured on the pre-fix tree: `ir_compare` takes its right operand as
        a BUFFER NAME (`Drivers/CPU/_lib/cpu.py:3139`), so a string literal
        arrives as a lookup key and the driver answers `KeyError: 'pro'` --
        an exception that names nothing. The capability exists (`str_eq`), the
        public comparison spelling was a trap.

        The decisions, per operator:

        * `==` / `!=`, TEXT column against a **string scalar**: lower through
          `dict_equal_lut` (`col = needle`,
          `src/Relational/DomainLUT/_lib/text_lut.py:307`) -- the same
          primitive, and the same `codes_lut_mask(validities=[...])` LIST
          discipline, that `isin` uses and that the semantic gate verified.
          `!=` is `==` composed with `not_()`, which is correct under 3VL.
          A NULL value never matches: the codes sidecar carries the validity.
        * `<`, `<=`, `>`, `>=` on text: NO lowering. Ordering on text needs a
          collation, and the engine defines none -- its dictionary codes are
          RANKs, not values, so comparing them would answer a different
          question than the one asked.
        * everything else that involves text or a string literal: refuse.
          None of it may reach the driver as a buffer lookup.
        """
        ltype = self._dtype_of(left, op)
        left_is_text = ltype == "text"
        right_is_expr = isinstance(right, Expr)
        right_is_str = isinstance(right, str)
        if not left_is_text:
            if right_is_str:
                raise plan.fail(
                    op,
                    f"{_CMP_TO_NODE[cmp_op]} against the string literal "
                    f"{right!r}: column {left.name!r} is {ltype!r}, not text, "
                    "and the driver would receive the literal as a buffer "
                    "name to look up (Drivers/CPU/_lib/cpu.py:3139).",
                    "compare a numeric column with a number: "
                    f"q.c('price') > 0")
            return
        colname = left.name if isinstance(left, Expr) else left
        if cmp_op in ("lt", "le", "gt", "ge"):
            raise plan.fail(
                op,
                f"{_CMP_TO_NODE[cmp_op]} on TEXT column {colname!r}: ordering on "
                "text has no lowering, because it needs a collation the engine "
                "does not define (the dictionary codes are ranks, not values).",
                "compare membership instead: "
                f"q.c({colname!r}).isin([...]) or "
                f"q.c({colname!r}).str_eq({right!r})")
        if right_is_expr or not right_is_str:
            raise plan.fail(
                op,
                f"{_CMP_TO_NODE[cmp_op]} on TEXT column {colname!r} against "
                f"{right!r}: v0 compares text only against a string scalar. "
                "There is no text-to-text comparison node, and a non-string "
                "right operand would reach the driver as a buffer lookup "
                "(Drivers/CPU/_lib/cpu.py:3139).",
                f"q.c({colname!r}).str_eq('...') -- and for 'not equal', "
                f"q.c({colname!r}).str_eq('...').not_()")
        # == / != against a string scalar: the one supported spelling.
        # Returns the NODE (not the (node, dtype) pair), so the caller wraps
        # it; `return None` below means "proceed with ir_compare".
        needle = Expr("isin", colname, [right])
        if cmp_op == "ne":
            needle = Expr("logic", arg=("not", needle, None))
        # rebind through the memo so the eq/ne pair shares one node with any
        # isin() of the same needle and the row-space epoch still applies.
        return self._bind(needle, op)[0]

    def _refuse_int_div(self, left, op):
        """`truediv` whose LEFT operand is an integer.

        `ir_map(..., 'div')` computes in float64 and then ROUNDS THE QUOTIENT
        BACK to the left operand's integer dtype
        (`Drivers/CPU/_lib/cpu.py:3109-3111`):

            r = (a.astype(np.float64) / b)
            if np.issubdtype(a.dtype, np.integer):
                r = np.rint(r).astype(a.dtype)

        The trigger is the LEFT operand's dtype, not "both are integers":
        `int64 / float_literal` rounds too (measured `i / 2.0` ->
        `[0,1,2]` where pandas gives `[0.5,1.0,1.5]`). So the guard keys on the
        left, and it refuses on an UNDECLARED left dtype as well: the facade
        cannot prove a derived expression is float, and the engine rounds
        whenever it is not.

        Not worked around by materialising a float64 copy of the column
        outside the DAG -- rule 02 (float64 at display only, never for series
        data), and it would make a column's dtype depend on which expression
        happened to consume it, which is a different lie.
        """
        ltype = self._dtype_of(left, op)
        if ltype not in ("int32", "int64", None):
            return
        colname = (left.name if isinstance(left, Expr) else left) \
            if isinstance(left, (Expr, str)) else repr(left)
        # The honest fix, because v0 has NO dtype-promotion verb: no cast, no
        # where, and a float LITERAL does not promote either -- ir_map casts
        # add/sub/mul back to the left dtype too
        # (Drivers/CPU/_lib/cpu.py:3133), measured `int64 * 2.0` -> int64.
        promote = ("v0 has no dtype-promotion verb: no cast and no where, and "
                   "a float literal does not promote either (ir_map casts "
                   "add/sub/mul back to the left dtype, "
                   "Drivers/CPU/_lib/cpu.py:3133), so an int column cannot "
                   "become a float column inside the DAG. Load the column as "
                   "float (from_pandas / from_numpy with float values), or "
                   "express the ratio with mul/add.")
        if ltype is None:
            raise plan.fail(
                op,
                f"truediv over {colname!r}: the left operand has no declared "
                "dtype -- it is a derived expression -- and ir_map('div') "
                "rounds the quotient back whenever the LEFT operand's buffer "
                "is an integer dtype (Drivers/CPU/_lib/cpu.py:3109-3111), so "
                "v0 cannot promise the float result it would be asked for. "
                + promote,
                "divide a column that is float at the source: "
                "q.c('rate') / q.c('other')")
        raise plan.fail(
            op,
            f"truediv over {ltype} column {colname!r} is refused: ir_map"
            "('div') computes in float64 and then ROUNDS THE QUOTIENT BACK to "
            "the left operand's integer dtype (Drivers/CPU/_lib/cpu.py:3109-"
            "3111), so i / 2.0 answers [0, 1, 2] where pandas gives "
            "[0.5, 1.0, 1.5]. The trigger is the LEFT dtype, so int / float "
            "rounds too. The fix is FROZEN (CPU_Driver). " + promote,
            "truediv needs a float LEFT operand. Note that v0 also has no "
            "honest INTEGER-division spelling: floor_div is in the driver's op "
            "list (cpu.py:3116) but not in v0.")

    # -- the const / scalar-operand normalisation --------------------------
    @staticmethod
    def _const_operand(value):
        """`expr.const(v)` -> v. Anything else is returned unchanged.

        `expr.const` is the facade's own representation of a scalar constant
        and it is NOT a DAG node -- it was never one. `_bind_new` used to send
        every Expr down the node path, so `c('v') - const(1.0)` reached
        `ir_map(out, node, 'sub', Expr(const 1.0))`: through the facade's own
        Expr class the node bind refused first with a message about PREDICATES
        (naming the wrong cause), and through any other Expr class the object
        went straight into the C driver, which answered
        `TypeError: int() argument must be a string, a bytes-like object or a
        real number, not 'Expr'`. Either way the constant did not lower.
        """
        if isinstance(value, Expr) and value.kind == "const":
            return value.arg
        return value

    @staticmethod
    def _is_literal(value):
        """True when the operand is a scalar LITERAL, not an expression."""
        return not isinstance(value, Expr)

    def _refuse_non_numeric(self, value, node_op, side, op, colname):
        """A scalar operand that is not a number.

        `ir_map` decides between "a scalar value" and "a second input buffer"
        with `isinstance(value, str)` (src/Semantic/IR/_lib/nodes.py:79), so a
        str operand is read as a buffer NAME to look up and the driver answers
        `KeyError: 'a'` -- an exception that names the literal and nothing
        else. `None` and a list reach it as an array/None and come back a numpy
        TypeError or a broadcast ValueError. pandas says the whole thing in one
        line: `str` + `int` is not defined. Same shape as `_cmp_guard`, which
        already owns the same trap on `ir_compare`.
        """
        which = f" on column {colname!r}" if colname else ""
        raise plan.fail(
            op,
            f"{node_op} with the {side} operand {value!r}{which} is refused: "
            "ir_map / ir_compare carry a NUMBER in their scalar operand, and a "
            "non-numeric one does not stop at the facade. A str is read by "
            "ir_map as the name of a second input buffer "
            "(src/Semantic/IR/_lib/nodes.py:79) and reaches the driver as a "
            "buffer lookup keyed by the literal itself; None and a list reach "
            "it as an array and come back a numpy TypeError or a broadcast "
            "ValueError. pandas refuses here too: there is no defined result.",
            "arithmetic and comparison take a number: "
            "q.c('price') > 0, q.c('price') * 2")

    def _refuse_text_bin(self, left, right, node_op, op):
        """The WHOLE `bin` family on a TEXT column -- six operators, one cause.

        `_codes_node` turns a TEXT column into `dictionary_encode` codes, and
        those codes are RANKS, not values. Measured on `['b','a','c']`: the
        sidecar is `['a','b','c']` and the codes are `[1, 0, 2]`, so
        `c('p') + 1` answered `[2, 1, 3]`, `* 2` answered `[2, 0, 4]`, and
        `/ 2`, `% 2`, `** 2` each answered a third set of plausible wrong
        numbers -- six operators, zero guards, no exception. `ir_map` has no
        text mode: it is one numeric buffer plus a scalar.

        `cumsum`, `shift`, `group` and `reduce` already refuse a text column
        for exactly this reason (the scan guard above, and the group/reduce
        type guards). `bin` was the one that did not.

        Refused BEFORE any node is emitted, so jobs[] is untouched.
        """
        for side, operand in (("left", left), ("right", right)):
            if self._dtype_of(operand, op) != "text":
                continue
            colname = operand.name if isinstance(operand, Expr) else operand
            raise plan.fail(
                op,
                f"{node_op} on TEXT column {colname!r} (the {side} operand) is "
                "refused: a text column is a dictionary_encode CODE vector in "
                "the DAG, and those codes are RANKS, not values. Measured on "
                "['b','a','c']: the sidecar is ['a','b','c'] and the codes are "
                "[1, 0, 2], so arithmetic on them returns plausible wrong "
                "numbers with no error, where pandas raises TypeError. "
                "ir_map has no text mode -- it is one numeric buffer plus a "
                "scalar.",
                "measure the text first: q.c('p').str_len() is an int32 "
                "code-point count, and arithmetic on THAT is defined -- "
                "q.c('p').str_len() + 1 means what it says")

    def _bin_operands(self, fn, left, right, op):
        """`bin` operands -> (buffer Expr on the left, node-or-scalar right).

        `ir_map(out, inp, fn, value)` has ONE slot for the buffer and ONE for
        the scalar, in that order, so where the scalar sits is part of the
        question:

        * scalar on the RIGHT: already representable, nothing to do;
        * scalar on the LEFT: representable only where the two commute. `add`
          and `mul` do, so they are swapped -- `1 + c('v')` is `c('v') + 1`,
          the same numbers. `sub`, `truediv`, `mod` and `pow` do NOT (`10 - 2`
          is 8, `2 - 10` is -8), so those refuse and say why, rather than
          silently answering the other question.
        """
        if self._is_literal(left):
            if isinstance(left, str) or not isinstance(left, _NUMERIC_SCALARS):
                self._refuse_non_numeric(left, plan.MAP_FN[fn], "left", op, None)
            if fn not in _COMMUTATIVE_BIN:
                raise plan.fail(
                    op,
                    f"{fn} with the scalar {left!r} on the LEFT has no "
                    f"lowering: ir_map(out, inp, {plan.MAP_FN[fn]!r}, value) "
                    "takes one buffer and one scalar in that order, and only "
                    "add and mul commute -- for the rest, swapping the "
                    "operands answers a DIFFERENT question (10 - 2 is 8 and "
                    "2 - 10 is -8; 10 % 2 is 0 and 2 % 10 is 2).",
                    f"put the column on the left, where the primitive is "
                    f"defined: q.c('v') - 2")
            if self._is_literal(right):
                raise plan.fail(
                    op,
                    f"{fn} between two literals ({left!r} and {right!r}) has "
                    "no column to compute over, and evaluating it in Python "
                    "would make a derived column's dtype depend on which "
                    "expression happened to consume it.",
                    "derive from a column: q.c('v') + 2")
            left, right = right, left
        if self._is_literal(right) and (
                isinstance(right, str)
                or not isinstance(right, _NUMERIC_SCALARS)):
            colname = left.name if isinstance(left, Expr) else repr(left)
            self._refuse_non_numeric(right, plan.MAP_FN[fn], "right", op, colname)
        return left, right

    def _cmp_operands(self, cmp_op, left, right, op):
        """`cmp` operands -> (buffer Expr on the left, node-or-scalar right).

        A scalar on the LEFT is mirrored rather than refused: every comparison
        has an exact mirror (`2 < c('v')` IS `c('v') > 2`), so the operand
        order is normalised and the existing right-hand path owns the rest. A
        str on the right is NOT touched here -- `_cmp_guard` owns it, because
        text `==` a string scalar is the one lowering that does exist.
        """
        if self._is_literal(left):
            if isinstance(left, str) or not isinstance(left, _NUMERIC_SCALARS):
                self._refuse_non_numeric(left, _CMP_TO_NODE[cmp_op], "left", op,
                                         None)
            if self._is_literal(right):
                raise plan.fail(
                    op,
                    f"{_CMP_TO_NODE[cmp_op]} between two literals ({left!r} "
                    f"and {right!r}) has no column to compare, and evaluating "
                    "it in Python would make a predicate depend on which "
                    "expression happened to consume it.",
                    "compare a column: q.c('price') > 0")
            cmp_op, left, right = _CMP_MIRROR[cmp_op], right, left
        if self._is_literal(right) and not isinstance(right, str) and (
                not isinstance(right, _NUMERIC_SCALARS)):
            colname = left.name if isinstance(left, Expr) else repr(left)
            self._refuse_non_numeric(right, _CMP_TO_NODE[cmp_op], "right", op,
                                     colname)
        return cmp_op, left, right

    # -- expression binding ---------------------------------------------
    def _bind(self, expr, op="filter"):
        """Expr -> (node name, declared logical dtype or None).

        Memoized by (row-space epoch, the expression's structural key). The key
        is also what keeps the Planner's CSE from rewriting a facade node name
        (see `expr.expr_key`); the epoch is what keeps the cache from surviving
        a change of row order or row count, where the same expression denotes a
        different column.
        """
        key = (self._space, expr_key(expr))
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        bound = self._bind_new(expr, op)
        self._memo[key] = bound
        return bound

    def _bind_new(self, expr, op="filter"):
        if isinstance(expr, Expr):
            if expr.kind == "col":
                col = self._col(expr.name, op)
                return self._need(col), col.dtype
            if expr.kind == "const":
                raise plan.fail(
                    op, f"a bare literal {expr.arg!r} is not a predicate.",
                    "compare a column: q.c('price') > 0")
            if expr.kind == "bin":
                fn, left, right = expr.arg
                left, right = (self._const_operand(left),
                               self._const_operand(right))
                left, right = self._bin_operands(fn, left, right, op)
                # BEFORE any bind: the guards must not even emit the left
                # operand's own node, so a refusal leaves jobs[] untouched.
                self._refuse_text_bin(left, right, plan.MAP_FN[fn], op)
                if fn == "truediv":
                    self._refuse_int_div(left, op)
                lnode, _ = self._bind(left, op)
                target = (self._bind(right, op)[0]
                          if isinstance(right, Expr) else right)
                out = self._emit("map", self._next("m"), lnode,
                                 plan.MAP_FN[fn], target)
                return out, None
            if expr.kind == "cmp":
                cmp_op, left, right = expr.arg
                left, right = (self._const_operand(left),
                               self._const_operand(right))
                cmp_op, left, right = self._cmp_operands(cmp_op, left, right, op)
                lowered = self._cmp_guard(cmp_op, left, right, op)
                if lowered is not None:
                    # text == / != a string scalar: lowered through the LUT,
                    # so there is no ir_compare node to emit at all.
                    return lowered, "bool"
                lnode, _ = self._bind(left, op)
                target = (self._bind(right, op)[0]
                          if isinstance(right, Expr) else right)
                out = self._emit("compare", self._next("cmp"), lnode, target,
                                 _CMP_TO_NODE[cmp_op])
                return out, "bool"
            if expr.kind == "logic":
                logic_op, a, b = expr.arg
                anode, _ = self._bind(a, op)
                if logic_op == "not":
                    out = self._emit("mask", self._next("mk"), anode, None,
                                     "not")
                else:
                    out = self._emit("mask", self._next("mk"), anode,
                                     self._bind(b, op)[0], logic_op)
                return out, "bool"
            if expr.kind == "isin":
                return self._bind_isin(expr.name, expr.arg), "bool"
            if expr.kind == "null":
                return self._bind_is_null(expr.name), "bool"
            if expr.kind == "text":
                text_op, arg = expr.arg
                col = self._col(expr.name, op)
                if col.dtype != "text":
                    raise plan.fail(
                        op,
                        f"{text_op} needs a TEXT column, {col.name!r} is "
                        f"{col.dtype!r}.",
                        "use q.c('name') arithmetic instead")
                out = self._next("t")
                out = self._emit(plan.TEXT_OP_NODE[text_op], out,
                                 self._text_values(col),
                                 *(() if arg is None else (arg,)))
                return out, ("int32" if text_op == "str_len" else "bool")
            if expr.kind == "scan":
                scan_op, arg = expr.arg[0], expr.arg[1]
                # The SOURCE is a nested expression when there is one, and the
                # column name otherwise (`expr._scan`). Binding it through
                # `_bind` -- not through `_col`/`_need` -- is what keeps the
                # row-space epoch: `sort`/`filter`/`limit` bump `_space`, so a
                # re-bound nested scan misses the memo and reads POST-op data
                # instead of handing back the pre-op node.
                source = expr.arg[2] if len(expr.arg) == 3 else expr.name
                snode, sdtype = self._bind(source, op)
                if sdtype == "text":
                    colname = source if isinstance(source, str) else source.name
                    raise plan.fail(
                        op,
                        f"{scan_op} on TEXT column {colname!r} is meaningless "
                        "(dictionary codes are ranks, not values).",
                        f"derive a numeric column, then {scan_op}")
                if sdtype == "bool":
                    # Measured: ir_cumsum / ir_shift accept int32/float32/float64
                    # only and refuse a packed bool vector
                    # (Drivers/CPU/_lib/cpu.py:1262, :3308). So a scan over a
                    # comparison has NO lowering -- and the engine says so only
                    # at execute time, after the node is already in jobs[].
                    # Refused here instead, at bind time, with a v0 fix.
                    raise plan.fail(
                        op,
                        f"{scan_op} has no lowering over the boolean mask "
                        f"{source!r}: ir_cumsum/ir_shift cover int32/float32/"
                        "float64 only and reject a packed bool vector "
                        "(Drivers/CPU/_lib/cpu.py:1262). v0 has no cast and no "
                        "where, so a mask cannot become numbers inside the DAG.",
                        "use the comparison as a filter mask: "
                        f".filter(q.c('y') > 3)")
                out = self._next("sc")
                if scan_op == "cumsum":
                    return self._emit("cumsum", out, snode), sdtype
                return self._emit("shift", out, snode, arg), sdtype
            raise plan.fail(op, f"unsupported expression {expr!r}.",
                            "use filter(q.c('x') > 0)")
        if isinstance(expr, str):
            col = self._col(expr, op)
            return self._need(col), col.dtype
        raise plan.fail(
            op,
            f"{op} needs an expression from q.c(...), got "
            f"{type(expr).__name__}.",
            f"{op}(q.c('price') > 0)")

    def _bind_isin(self, colname, needles):
        """`c.isin([...])` on TEXT: out-of-DAG pre-pass -> bool series.

        POINT MEMBERSHIP, so the D-scale predicate is EXACT equality:
        `dict_equal_lut(values, needle)` is `col = needle`
        (src/Relational/DomainLUT/_lib/text_lut.py:307). `dict_contains_lut`
        is `col LIKE '%needle%'` (text_lut.py:276) -- substring containment,
        and lowering `isin` through it made `isin(['app'])` keep `'apple'`
        with no error. One LUT per needle, OR-ed here, is the whole
        composition the engine offers (the engine has no isin node), so the
        disjunction lives here at D scale (GATE §9).
        """
        col = self._col(colname, "filter")
        if col.dtype != "text":
            raise plan.fail(
                "filter",
                f"isin v0 lowers through the dictionary path only; "
                f"{colname!r} is {col.dtype!r}.",
                "compare with < / == , or filter on a text column")
        self._codes_node(col)
        codes, validity = self.material(col)
        lut = None
        for needle in needles:
            one = np.asarray(
                plan.prepass("dict_equal_lut")(list(col.sidecar or []), needle),
                dtype=bool)
            lut = one if lut is None else np.logical_or(lut, one)
        # codes_lut_mask(codes, luts, validities): validities is a LIST with
        # one column per code column. A bare array raises (GATE §2.3), and
        # omitting it entirely matches NULL rows (L4). [None] is the correct
        # "no gate" form for a NULL-free column -- np.asarray(None) is the
        # scalar False that empties the whole mask (L9).
        mask = np.ascontiguousarray(
            np.asarray(plan.prepass("codes_lut_mask")(
                [codes], [lut],
                [None if validity is None else np.asarray(validity, dtype=bool)]),
                dtype=bool))
        # The codes' validity goes onto the mask node as well. codes_lut_mask
        # gates a NULL row to DATA-False, and a data-False that is not marked
        # invalid is a hard False: `not_()` would then turn a NULL row into
        # `True` and `c != 'x'` would KEEP it, where pandas drops it (3VL:
        # NULL != 'x' is UNKNOWN). Carrying the validity makes the NULL row
        # UNKNOWN, which `filter` drops and `not_()` preserves. Measured on the
        # pre-fix tree: `filter(c != 'pro')` returned ['free', None, 'pro ']
        # where pandas returns ['free', 'pro '].
        out = self._next("isin")
        if validity is None:
            return self._emit("series", out, mask, "bool")
        return self._emit("series", out, mask, "bool",
                          np.ascontiguousarray(np.asarray(validity, dtype=bool)))

    def _bind_is_null(self, colname):
        """`c.is_null()` -> `ir_series(bool)`, True exactly on NULL rows.

        THE SAME `material()` THE NULL-KEY GUARD ALREADY CALLS, and that is
        the whole justification for shipping the verb. `_refuse_null_key` does
        exactly

            _, validity = self.material(col)
            n_null = plan.null_count(validity)

        and it runs in production on every `group()` build, so an out-of-graph
        validity read is already how this facade answers "is this row NULL" --
        it was answering it in order to REFUSE. `is_null()` returns
        that answer instead. A second materialisation helper would have destroyed
        the argument for shipping: it would be a new mechanism, not a
        composition of existing ones.

        `material()` returns HOST arrays when `col.node is None` (a source
        column), so a source column costs zero executions; it reads the node's
        `#validity` buffer once the column has a DAG node. Both row-space
        directions hold for free: after `sort`/`filter`/`limit` the Col's `node`
        is already the post-op gather/filter/slice node, so the read sees
        post-op rows, and a `sort` AFTER this verb emits
        `ir_gather(nul_N, perm_M)` over the mask column like any other column.
        No extra node, no new op, no FROZEN component touched.

        No validity sidecar on the mask node, deliberately. "This row is NULL"
        is a fact about the row, so the answer is hard False on every non-NULL
        row -- and a sidecar would make those rows UNKNOWN under 3VL, so
        `is_null().not_()` would drop them in `filter` and the guard's own advice
        would not work. (`_bind_isin` is the opposite case: there the predicate
        really is UNKNOWN on a NULL row, so it carries the codes' validity.)

        Per-row types, as pinned by `test_is_null_agrees_with_the_pandas_oracle`:
        a sidecar-bearing column gives `~validity`; a column with no sidecar
        (NULL-free source, aggregate output of `group`, empty column) gives
        all-False; a TEXT column's sidecar is `Series.validity`, measured equal
        to `dictionary_encode(...)['validity']` row for row, which is the second
        observer §2.2b names.
        """
        col = self._col(colname, "filter")
        values, validity = self.material(col)
        if validity is None:
            mask = np.zeros(len(values), dtype=bool)
        else:
            mask = np.ascontiguousarray(~np.asarray(validity, dtype=bool))
        out = self._next("nul")
        return self._emit("series", out, mask, "bool")

    # -- the chain vocabulary (07:60) ------------------------------------
    def filter(self, expr):
        self._branch()
        node, dtype = self._bind(expr, "filter")
        if dtype != "bool":
            raise plan.fail(
                "filter", f"filter needs a boolean mask, {expr!r} is "
                          f"{dtype!r}.", "filter(q.c('price') > 0)")
        cols = {}
        for name, col in self._cols.items():
            # the SAME `_emit` every other node goes through, so two columns
            # that share a DAG node (the expression memo's whole job) cannot
            # emit two identical `ir_filter` nodes for the Planner's CSE to
            # merge -- see `_emit`.
            out = self._emit("filter", self._next(f"f_{name}"),
                             self._need(col), node)
            cols[name] = Col(name, col.dtype, out, sidecar=col.sidecar)
        return self._spawn(cols, new_space=True)

    def derive(self, name, expr):
        if not isinstance(name, str) or not name:
            raise plan.fail("derive",
                            f"column name must be a non-empty str, got {name!r}.",
                            "derive('n', expr)")
        if name in self._cols:
            raise plan.fail(
                "derive", f"column {name!r} already exists "
                          f"(have {self._order}).", "derive under a new name")
        self._branch()
        node, dtype = self._bind(expr, "derive")
        cols = dict(self._cols)
        cols[name] = Col(name, dtype, node)
        return self._spawn(cols)

    def sort(self, *keys, desc=False):
        if not keys:
            raise plan.fail("sort", "sort needs >=1 key column.",
                            "sort('ts', desc=True)")
        if isinstance(desc, (list, tuple)):
            raise plan.fail(
                "sort",
                f"v0 sort takes one desc flag for all keys, got {desc!r}.",
                "sort('k1', 'k2', desc=False)")
        self._branch()
        perm = self._next("perm")
        # descending MUST be a keyword: ir_sort(out, *keys, descending=) -- a
        # third positional is another KEY, not a flag (L5 -> GAP-7).
        self._emit("sort", perm,
                   *[self._need(self._col(k, "sort")) for k in keys],
                   descending=bool(desc))
        cols = {}
        for name, col in self._cols.items():
            out = self._emit("gather", self._next(f"srt_{name}"),
                             self._need(col), perm)
            cols[name] = Col(name, col.dtype, out, sidecar=col.sidecar)
        return self._spawn(cols, new_space=True)

    def limit(self, n, offset=0):
        self._branch()
        cols = {}
        for name, col in self._cols.items():
            out = self._emit("slice", self._next(f"lim_{name}"),
                             self._need(col), n, offset)
            cols[name] = Col(name, col.dtype, out, sidecar=col.sidecar)
        return self._spawn(cols, new_space=True)

    def group(self, keys, aggs):
        self._branch()
        key = _single_key(keys)
        kcol = self._col(key, "group")
        if not isinstance(aggs, dict) or not aggs:
            raise plan.fail(
                "group",
                f"aggs must be a non-empty {{col: (ops,)}} dict, got {aggs!r}.",
                "group('utm', {'price': ('sum', 'count')})")
        specs = {}
        for raw_name, ops in aggs.items():
            col_name = column_name(raw_name, "group")
            ops_t = (ops,) if isinstance(ops, str) else tuple(ops)
            if not ops_t:
                raise plan.fail("group", f"aggs[{col_name!r}] is empty.",
                                "name at least one of "
                                "sum/count/mean/min/max/nunique")
            for agg in ops_t:
                if agg != "nunique" and agg not in plan.GROUPBY_OPS:
                    raise plan.fail(
                        "group", f"unknown aggregate {agg!r} for "
                                 f"{col_name!r}.",
                        "use one of sum/count/mean/min/max/nunique")
            specs[col_name] = ops_t
        for col_name in specs:
            mcol = self._col(col_name, "group")
            if mcol.dtype == "text":
                raise plan.fail(
                    "group",
                    f"aggregate over TEXT column {col_name!r} is meaningless "
                    "(dictionary codes are ranks, not values).",
                    "encode the text first (filter / str_len), group that")

        # §2.2b -- the loud refusal, BEFORE groupby_multi is ever called.
        self._refuse_null_key(kcol)

        knode = self._need(kcol)
        gb_ops = {}
        nuniq = []
        for col_name, ops_t in specs.items():
            for agg in ops_t:
                if agg == "nunique":
                    if col_name not in nuniq:
                        nuniq.append(col_name)
                else:
                    gb_ops.setdefault(col_name, []).append(agg)

        carry = None
        carry_cols = {}
        if gb_ops:
            cols_order = list(gb_ops)
            if len(cols_order) == 1:
                values = self._need(self._col(cols_order[0], "group"))
                carry_cols[cols_order[0]] = values
                ops_arg = tuple(gb_ops[cols_order[0]])
            else:
                values = [self._need(self._col(v, "group"))
                          for v in cols_order]
                for v in cols_order:
                    carry_cols[v] = values[cols_order.index(v)]
                ops_arg = {v: tuple(gb_ops[v]) for v in cols_order}
            gnode = self._emit("groupby_multi", self._next("G"), values, knode,
                              ops_arg, result="carry")
            carry = self._buffers()[gnode + "#carry"]

        distinct = {}
        for col_name in nuniq:
            dnode = self._emit(
                "count_distinct", self._next("nd"),
                self._need(self._col(col_name, "group")), knode)
            distinct[col_name] = self._buffers()[dnode]

        return self._grouped(kcol, specs, carry, carry_cols, distinct)

    def _grouped(self, kcol, specs, carry, carry_cols, distinct):
        """GroupedResult -> the next chain (two-graph recipe, DESIGN §2.2)."""
        if carry is not None:
            ukeys = np.ascontiguousarray(
                np.asarray(carry.ukeys, dtype=np.int64).reshape(-1))
        else:
            first = distinct[next(iter(distinct))]
            ukeys = np.ascontiguousarray(
                np.asarray(sorted(int(k) for k in first), dtype=np.int64))
        if kcol.dtype == "text":
            key_col = Col(kcol.name, "text", None,
                          values=np.asarray(
                              plan.decode_text(ukeys, kcol.sidecar or [""]),
                              dtype=object))
        else:
            key_col = Col(kcol.name, kcol.dtype, None, values=ukeys)

        cols = {kcol.name: key_col}
        for col_name, ops_t in specs.items():
            # ColumnCarry is keyed by the DAG node name, which is the column
            # node AFTER every earlier op ('vf', not 'v') -- DESIGN §5.2.
            node_name = carry_cols.get(col_name, col_name)
            for agg in ops_t:
                field = f"{col_name}.{agg}"
                if agg == "nunique":
                    lut = distinct[col_name]
                    cols[field] = Col(field, "int64", None, values=(
                        np.ascontiguousarray(np.asarray(
                            [lut[int(k)] for k in ukeys.tolist()],
                            dtype=np.int64))))
                    continue
                if agg == "count":
                    vals = carry.counts
                elif agg == "mean":
                    vals = carry.means(node_name)
                elif agg == "min":
                    vals = carry.mins[node_name]
                elif agg == "max":
                    vals = carry.maxs[node_name]
                else:
                    vals = carry.sums[node_name]
                cols[field] = Col(field, None, None,
                                  values=np.ascontiguousarray(
                                      np.asarray(vals)))
        table = _kernel_table(self._kernel, {
            name: _kernel_series(self._kernel, cols[name], cols[name].values,
                                 cols[name].validity)
            for name in cols})
        fresh = {name: _source_col(self._kernel, name, table.column(name))
                 for name in cols}
        return Chain(self._kernel, fresh, len(ukeys), [], pending=table)

    def reduce(self, op, col=None):
        if op not in plan.REDUCE_OPS:
            raise plan.fail(
                "reduce",
                f"unknown reduce op {op!r}: use one of {list(plan.REDUCE_OPS)}.",
                "reduce('sum')")
        if col is None:
            if len(self._cols) != 1:
                raise plan.fail(
                    "reduce",
                    f"reduce needs a column: the chain has {self._order}.",
                    "reduce('sum', 'price')")
            col = self._order[0]
        target = self._col(column_name(col, "reduce"), "reduce")
        if target.dtype == "text":
            raise plan.fail(
                "reduce", f"reduce over TEXT column {target.name!r} has no "
                          "meaning.", "str_len(...) first, then reduce('sum')")
        self._branch()
        # skipna=True is the ONE NULL semantics of the v0 aggregates, shared
        # with `group`: NULL values are skipped, as in pandas' default.
        # Measured on the pre-fix tree, without it `reduce('count')` returned
        # the ROW count (3 where pandas `.count()` says 2) and
        # `reduce('sum')` returned NaN with only a UserWarning -- so the two
        # v0 verbs disagreed on the same aggregate over the same fixture.
        # `count` therefore means the number of NON-NULL values, here and in
        # `group`; pandas' `.size` (rows including NULL) is a different
        # quantity and is not what `count` means in v0. The engine already
        # carries the flag (`nodes.py:210 ir_reduce(..., skipna=False)`,
        # read at `Drivers/CPU/_lib/cpu.py:3849`); the facade was not passing
        # it.
        out = self._emit("reduce", self._next("red"), self._need(target), op,
                         skipna=True)
        value = np.asarray(self._buffers()[out]).reshape(-1)
        return value[0].item() if value.size else None

    # -- terminal + introspection (07:60) -------------------------------
    def jobs(self):
        """jobs[] -- {op, inputs, params, out} per node, in build order."""
        return [dict(j) for j in self._jobs]

    def explain(self):
        """Planner EXPLAIN of jobs[] (verbatim text report)."""
        return plan.report_of(self._jobs)

    def nrows(self):
        if not self._cols:
            return self._n
        first = self._cols[self._order[0]]
        if first.node is None:
            return self._n
        buf, _ = self._buffer(first.node)
        return int(np.asarray(buf).shape[0])

    def compile(self):
        """Execute the chain: one planner call, CPU, -> numfast Table."""
        if self._pending is not None and all(c.node is None
                                            for c in self._cols.values()):
            return self._pending
        cols = {}
        for name in self._order:
            col = self._cols[name]
            if col.node is None:
                values, validity = col.values, col.validity
            else:
                values, validity = self._buffer(col.node)
            cols[name] = _kernel_series(self._kernel, col, values, validity)
        return _kernel_table(self._kernel, cols)

    def __repr__(self):
        return (f"Chain({len(self._cols)} cols {self._order}, "
                f"jobs={len(self._jobs)})")


def _gpu_capability():
    """gpu_capability() dict, or None when this kernel has no GPU driver.

    NEVER raises. `capabilities()` answers the CPU question today on a kernel
    that was built without Drivers/GPU, and adding a probe must not turn that
    into an exception -- a capability call that starts raising where it did
    not is a behaviour change dressed up as disclosure.

    But it reports absence VISIBLY (`None`, surfaced as `gpu_ops: None` with
    `gpu_note` naming the missing extension) rather than degrading to an empty
    GPU op list, because "the GPU driver is not registered" and "the GPU
    supports no operations" are different facts and only the first is true
    on such a kernel.
    """
    try:
        return plan.tool("gpu_capability")()
    except Exception:
        return None


# What actually happens when a CPU-only operation is asked for on the GPU.
# Not a policy statement: `evaluate(graph, 'gpu', n)` raises RuntimeError
# naming `op:<name>`, because `select_backend` reports the op as a
# `gpu_blockers` entry and Runtime refuses the ineligible backend. There is
# no silent CPU fallback anywhere on this path.
_ON_CPU_ONLY_UNDER_GPU = (
    "asking for backend='gpu' on a graph that uses one of these raises "
    "RuntimeError naming 'op:<name>' -- there is no silent CPU fallback")

# What `backend='auto'` actually does. Stated precisely because the shorter
# "auto defaults to the CPU" is true on this checkout but is NOT a contract:
# planner.select_backend_dual_impl ends with
# `backend = "gpu" if gpu < cpu else "cpu"` over MEASURED costs, so auto WILL
# pick the GPU whenever a measured calibration says the GPU host cost is
# strictly lower. What auto never does is invent a GPU choice from an
# unmeasured or uncovered graph -- those resolve to the CPU with the reason
# recorded (`calibrated routing has no measured cost for op:<name>`).
_BACKEND_AUTO = (
    "backend='auto' resolves to the CPU unless a measured calibration "
    "reports a strictly lower GPU host cost; an unmeasured or uncovered "
    "graph resolves to the CPU, never to a fabricated GPU choice")


def _disclose(cpu, gpu):
    """cpu_capability() plus the GPU split, in one dict.

    The engine was ALREADY honest and still undiscoverable. `gpu_capability`
    lists the ops that execute on the device; `select_backend` names every
    other op as a `op:<name>` blocker; `evaluate(graph, 'gpu', n)` raises on a
    blocker instead of falling back silently. Nothing in the GPU driver masks
    a missing operation.

    What was missing was the PUBLIC statement. A user of `nf.*` could not
    learn which ops are on the GPU, which are CPU-only, or what asking for
    the GPU on a CPU-only op would do -- and `repr(Series)` carried no backend
    at all, so a result in hand did not say where it came from. Those four
    facts are what this dict adds; every value is read from an existing
    capability call, none of it is asserted here.
    """
    out = dict(cpu)
    all_ops = sorted(cpu.get("ops") or ())
    out["op_count"] = len(all_ops)
    if gpu is None:
        out["gpu_op_count"] = None
        out["cpu_only_op_count"] = None
        out["gpu_ops"] = None
        out["cpu_only_ops"] = None
        out["gpu_note"] = ("this kernel exposes no gpu_capability: the GPU "
                           "driver extension is not registered. The counts "
                           "above are CPU-only by construction.")
        out["on_cpu_only_under_gpu_backend"] = None
    else:
        gpu_ops = sorted(gpu.get("ops") or ())
        gpu_set = set(gpu_ops)
        cpu_only = [op for op in all_ops if op not in gpu_set]
        out["gpu_op_count"] = len(gpu_ops)
        out["cpu_only_op_count"] = len(cpu_only)
        out["gpu_ops"] = gpu_ops
        out["cpu_only_ops"] = cpu_only
        out["gpu_note"] = gpu.get("note")
        out["on_cpu_only_under_gpu_backend"] = _ON_CPU_ONLY_UNDER_GPU
    out["backend_auto"] = _BACKEND_AUTO
    return out


# --- App (one entry point) -------------------------------------------------

class App:
    """The single consumer entry point: facade + chain factory."""

    def __init__(self, kernel):
        self._kernel = kernel

    def c(self, name):
        """Column reference factory -- q.c('price')."""
        return ref(name)

    def capabilities(self):
        """cpu_capability() PLUS the GPU split -- the honest backend picture.

        Carries the unchanged cpu_capability keys (`ops`, `max_dispatch`,
        `chunkable_hints`, `note`) and adds the disclosure the public surface
        used to withhold:

          op_count / gpu_op_count / cpu_only_op_count  -- the 15-of-33 split
          gpu_ops        -- the operations that execute on the GPU
          cpu_only_ops   -- the operations that execute on the CPU
          gpu_note       -- the GPU driver's own per-op note, verbatim
          on_cpu_only_under_gpu_backend -- what asking for the GPU on a
                            CPU-only op does (raises, naming the op; no
                            silent fallback)
          backend_auto   -- what backend='auto' actually resolves to

        `gpu_capabilities()` is the same GPU driver's facts on their own.
        """
        return _disclose(plan.tool("cpu_capability")(), _gpu_capability())

    def gpu_capabilities(self):
        """gpu_capability(): ops / max_dispatch / chunkable_hints / note.

        The GPU driver's own capability record, unedited. `ops` is the set of
        operations that execute on the device; it is the same list that
        `capabilities()['gpu_ops']` reports, reachable on its own so the
        question "does this op run on the GPU?" has a public answer.
        """
        gpu = _gpu_capability()
        if gpu is None:
            raise plan.fail(
                "capabilities",
                "this kernel exposes no gpu_capability: the GPU driver "
                "extension is not registered.",
                "capabilities()['gpu_ops'] is None on such a kernel; build "
                "with Drivers/GPU to ask the question",
            )
        return gpu

    def open_stream(self, path, budget_frac=0.25, force_lazy=False):
        """Lazy budgeted block reader over an nfs-stream-v1 file (E6)."""
        return _stream.open_stream(path, budget_frac, force_lazy)

    def query(self, table):
        """Start a lazy chain from a Table (07:60 query())."""
        return Chain.from_table(self._kernel, table)

    def schema(self, table):
        """Per-column schema from Series.schema -- the honest observer.

        NOTE (DESIGN §4.3, GAP-2, unresolved): Series.schema reports
        logical='text' for a dictionary column, while column_schema() cannot
        build a text logical. This returns what the observer reports; v0 does
        not pick a winner between the two sources.
        """
        if not hasattr(table, "names"):
            raise plan.fail("schema",
                            f"schema needs a Table, got {type(table).__name__}.",
                            "schema(t) for a numfast Table")
        return {name: table.column(name).schema for name in table.names}

    def __repr__(self):
        return ("App(c=..., capabilities=..., gpu_capabilities=..., "
                "open_stream=..., query=...)")


def app():
    """The consumer facade entry point."""
    return App(plan.kernel())


def chain(table):
    """Start a lazy chain from a Table -- the 07:60 query() entry point."""
    return Chain.from_table(plan.kernel(), table)


def _table_query(self):
    """`Table.query()` -- dispatched through the CURRENT kernel alias.

    Stateless on purpose: a rebuilt kernel installs a fresh set of `_lib`
    modules, so a closure bound at load time would hand out a chain whose
    expressions belong to a different `Expr` class than `app()` returns.
    """
    import numfast as nf
    return nf.get_kernel().alias["tableexpr_chain"](self)


def _table_schema(self):
    """`Table.schema` -- same dispatch rule as `_table_query`."""
    import numfast as nf
    return nf.get_kernel().alias["tableexpr_app"]().schema(self)


def _install_table_surface():
    """Give `Table` the two consumer verbs it cannot have without a file edit.

    DESIGN GAP-2 asks for `Table.query()` / `Table.schema` explicitly "without
    changing Table": no packaged file is edited, and the two verbs are thin
    kernel-alias dispatchers, so they stay correct across kernel rebuilds.
    """
    import numfast as nf

    nf.Table.query = _table_query
    nf.Table.schema = property(_table_schema)


_install_table_surface()