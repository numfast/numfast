# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""chain.py -- Table -> query -> filter -> derive -> group -> reduce -> compile.

The 07:60 vocabulary, implemented as a lazy chain over one IR node per
column. Every op appends nodes to `jobs[]`; `compile()` makes exactly ONE
planner call over that list and executes it on the CPU oracle.

Four honest deviations, all recorded in DESIGN §2.2a / §2.2b / §3.3:

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
3. `sort()` REFUSES LOUDLY on a NULL in any sort key, for the same reason:
   `ir_sort` reads a NULL key as 0, KEEPS the row (validity=False), and
   ascending order therefore yields `[1, 2, 3, 0]` -- NULL rows first
   (GATE_semantics.md §9). Shipping that order silently is worse than
   refusing.

4. `_bind`'s memo is scoped to ONE row space. `filter`, `sort` and `limit`
   rebuild every column node into a different row space (new order or new
   count), so a cached `(expr_key) -> node` entry from before the op still
   names the PRE-op node -- the rebind would return pre-sort data with the
   post-sort column beside it, which is a wrong answer with the right shape
   and no exception (silent). The memo therefore carries a row-space
   epoch: every op that changes row order or row count bumps it, so a
   rebind in a new row space misses the cache instead of hitting it. The
   memo still does its real job inside one row space -- see `_spawn`.

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


def _kernel_series(kernel, col, values, validity):
    """Buffer + column record -> numfast Series (public boundary type)."""
    import numfast as nf
    buf = np.ascontiguousarray(np.asarray(values))
    valid = plan.as_validity(validity)
    if col.dtype == "text":
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
    return nf.Series(kernel, col.name, buf, logical, validity=valid)


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
                 bufs=None, memo=None, space=0):
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
        self._jobs.append(plan.node(op, out, *args, **kw))
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
                     self._space + 1 if new_space else self._space)

    def _col(self, name, op="query"):
        col = self._cols.get(name)
        if col is None:
            raise plan.fail(
                op, f"no column {name!r} (have {self._order}).",
                "use q.c() with one of the existing column names")
        return col

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
            self._emit("series", out, codes, "int32")
        else:
            self._emit("series", out, codes, "int32", enc_valid)
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
            self._emit("series", out, values, col.dtype)
        else:
            self._emit("series", out, values, col.dtype, col.validity)
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

    # -- the two loud guards --------------------------------------------
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

    def _refuse_null_sort_key(self, col):
        _, validity = self.material(col)
        n_null = plan.null_count(validity)
        if n_null:
            raise plan.fail(
                "sort",
                f"key column {col.name!r} has {n_null} NULL rows; ir_sort "
                "reads a NULL key as 0 and KEEPS the row (validity=False), "
                "so ascending order would return NULL rows first "
                "(GATE_semantics.md §9).",
                "filter those rows out (or fill them) before sort()")

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
                lnode, _ = self._bind(left, op)
                target = (self._bind(right, op)[0]
                          if isinstance(right, Expr) else right)
                out = self._next("m")
                self._emit("map", out, lnode, plan.MAP_FN[fn], target)
                return out, None
            if expr.kind == "cmp":
                cmp_op, left, right = expr.arg
                lnode, _ = self._bind(left, op)
                target = (self._bind(right, op)[0]
                          if isinstance(right, Expr) else right)
                out = self._next("cmp")
                self._emit("compare", out, lnode, target, _CMP_TO_NODE[cmp_op])
                return out, "bool"
            if expr.kind == "logic":
                logic_op, a, b = expr.arg
                anode, _ = self._bind(a, op)
                out = self._next("mk")
                if logic_op == "not":
                    self._emit("mask", out, anode, None, "not")
                else:
                    self._emit("mask", out, anode, self._bind(b, op)[0],
                               logic_op)
                return out, "bool"
            if expr.kind == "isin":
                return self._bind_isin(expr.name, expr.arg), "bool"
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
                self._emit(plan.TEXT_OP_NODE[text_op], out,
                           self._text_values(col),
                           *(() if arg is None else (arg,)))
                return out, ("int32" if text_op == "str_len" else "bool")
            if expr.kind == "scan":
                scan_op, arg = expr.arg
                col = self._col(expr.name, op)
                if col.dtype == "text":
                    raise plan.fail(
                        op,
                        f"{scan_op} on TEXT column {col.name!r} is meaningless "
                        "(dictionary codes are ranks, not values).",
                        f"derive a numeric column, then {scan_op}")
                out = self._next("sc")
                if scan_op == "cumsum":
                    self._emit("cumsum", out, self._need(col))
                else:
                    self._emit("shift", out, self._need(col), arg)
                return out, col.dtype
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
        return self._emit("series", self._next("isin"), mask, "bool")

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
            out = self._next(f"f_{name}")
            self._emit("filter", out, self._need(col), node)
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
        for key in keys:
            self._refuse_null_sort_key(self._col(key, "sort"))
        perm = self._next("perm")
        # descending MUST be a keyword: ir_sort(out, *keys, descending=) -- a
        # third positional is another KEY, not a flag (L5 -> GAP-7).
        self._emit("sort", perm,
                   *[self._need(self._col(k, "sort")) for k in keys],
                   descending=bool(desc))
        cols = {}
        for name, col in self._cols.items():
            out = self._next(f"srt_{name}")
            self._emit("gather", out, self._need(col), perm)
            cols[name] = Col(name, col.dtype, out, sidecar=col.sidecar)
        return self._spawn(cols, new_space=True)

    def limit(self, n, offset=0):
        self._branch()
        cols = {}
        for name, col in self._cols.items():
            out = self._next(f"lim_{name}")
            self._emit("slice", out, self._need(col), n, offset)
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
            gnode = self._next("G")
            self._emit("groupby_multi", gnode, values, knode, ops_arg,
                       result="carry")
            carry = self._buffers()[gnode + "#carry"]

        distinct = {}
        for col_name in nuniq:
            dnode = self._next("nd")
            self._emit("count_distinct", dnode,
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
        out = self._next("red")
        self._emit("reduce", out, self._need(target), op)
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


# --- App (one entry point) -------------------------------------------------

class App:
    """The single consumer entry point: facade + chain factory."""

    def __init__(self, kernel):
        self._kernel = kernel

    def c(self, name):
        """Column reference factory -- q.c('price')."""
        return ref(name)

    def capabilities(self):
        """cpu_capability(): ops / max_dispatch / chunkable_hints."""
        return plan.tool("cpu_capability")()

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
        return "App(c=..., capabilities=..., open_stream=..., query=...)"


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