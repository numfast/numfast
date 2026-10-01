# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Fused composite grouping + order-by-aggregate + limit-k, vectorized.

`composite_group_plan` pays a per-group Python cost that no amount of
vector work removes: the frozen CPU driver's composite lane materialises
every group as a nested dict cell before the caller sees a single group.
On a grouping whose group count is the same order as the row count that
dict is the whole query, and flattening it costs as much again.

This module is the other shape of the same operation. The grouping itself
is unchanged -- same `pack_keys` + `groupby_multi` nodes, same frozen
driver, same traversal -- but the groupby_multi node is asked for the
columnar result (`result='carry'`) instead of the compat dict, so the
groups arrive as parallel NumPy arrays (ukeys / counts / sums / means /
mins / maxs) and stay that way. `composite_group_fused_topk` then picks
the k requested groups with one vectorised partition plus one sort over k
elements, and only those k groups are ever turned into Python values.

The two halves are deliberately separate: planning is a pure job-graph
builder, selection is a pure read. Nothing here executes a graph and
nothing here inspects a column to decide what to do.

Genericity, stated as rules this module holds itself to:

  * `limit` is an operation parameter, read from the caller's argument.
    It is never a constant, never derived from a query, and never
    clamped to a particular value.
  * `order_by` names one of the aggregates the plan actually asked for,
    in the same `"<measure>.<op>"` / bare group-op spelling
    `composite_group_read` uses. Any requested aggregate can drive the
    order, ascending or descending.
  * The key layout is derived from argument shapes (how many key columns,
    how wide they are), never from the values in them. The one place a
    column is measured at all is computing radix widths, and that number
    is a packing parameter recorded in the plan, not a control-flow
    decision.
  * There is no branch on "groups == rows", on a tie plateau, on a
    particular column name, or on a particular result size. A grouping
    whose every group is a singleton and one whose every group is huge
    take exactly the same path through this file.

Known limit, stated rather than worked around: the radix lane needs a
key space that fits in int64, and the frozen driver raises when it does
not. Widening the key space is a driver concern, not something hidden
here.
"""

import numpy as np

_VALUE_OPS = ("sum", "count", "mean", "min", "max")
_GROUP_OPS = ("count",)

# Two int32 columns occupy exactly 64 bits, so the bitpack lane below is
# bijective for any int32 pair, negatives included.
_PACK_SHIFT = np.int64(32)
_PACK_LO_MASK = np.int64(0xFFFFFFFF)
_INT32_MAX = 2 ** 31 - 1
_INT64_MAX = 2 ** 63 - 1


def _err(what, fix, doc="specs/delta-1-fused-aggregate.md"):
    return ValueError(f"{what} Fix: {fix}. See {doc}")


def _as_int_column(name, col):
    """A key/measure column as a 1-D contiguous integer array.

    The frozen driver's `series` node is the only thing that decides the
    physical width, so this only normalises shape and refuses non-numeric
    input; narrowing to the engine's logical width stays with `series`.
    """
    arr = np.asarray(col)
    if arr.ndim != 1:
        arr = arr.ravel()
    if arr.dtype.kind not in "iub":
        raise _err(f"column '{name}' has dtype kind '{arr.dtype.kind}', not int/uint/bool.",
                   "pass integer code columns for keys and int measure columns")
    return np.ascontiguousarray(arr)


def _dtype_name(arr):
    """Physical dtype for the `series` node holding `arr`."""
    if arr.dtype.kind == "b":
        return "int32"
    if arr.dtype.kind in "iu":
        return "int32" if _INT32_MAX >= int(arr.max(initial=0)) and \
            int(arr.min(initial=0)) >= -_INT32_MAX - 1 else "int64"
    raise _err("non-integer column reached the key/measure packer.",
               "pass integer code columns for keys and int measure columns")


def _pack_layout(key_arrays):
    """Key layout for a set of key columns, from their shapes alone.

    Returns a dict describing how the driver node packs the keys and how
    to read the per-column codes back out of the packed int64 ukeys:

      mode    "none" | "pack" | "radix"
      radix   per-column code-space widths, [] unless mode == "radix"
      shift   per-column divisor, [] unless mode == "radix"
      mins    per-column rebasing offsets, [] unless mode == "radix"

    `none`  one key column is its own key space.
    `pack`  two int32 columns, bitpacked into int64. Bijective by
            construction for every int32 pair, so no range is consulted.
    `radix` any other count. Each column is rebased onto its own minimum
            and the columns are mixed-radix combined, which is the frozen
            driver's own lane: `acc = c0`, then `acc = acc*radix[i] + c_i`
            for every later column. So column i sits at weight
            `radix[i+1] * ... * radix[n-1]` and is recovered by dividing
            out that weight and reducing modulo its own radix -- the same
            reduction for every column, because the first column is never
            multiplied. Rebasing is why negative codes are fine, and why
            these columns travel as int64 rather than int32.
    """
    n = len(key_arrays)
    if n == 1:
        return {"mode": "none", "radix": [], "shift": [], "mins": []}
    if n == 2 and all(a.dtype.kind in "iu" and a.dtype.itemsize <= 4
                      for a in key_arrays):
        return {"mode": "pack", "radix": [], "shift": [], "mins": []}
    radix, mins = [], []
    for arr in key_arrays:
        lo = int(arr.min(initial=0))
        hi = int(arr.max(initial=0))
        radix.append(hi - lo + 1)
        mins.append(lo)
    # Driver lane: acc starts at c0 and each later column folds in as
    # acc = acc*radix[i] + c_i. So column i is recovered by dividing out
    # the product of every LATER radix, then reducing modulo its own.
    shift = [1] * n
    acc = 1
    for pos in range(n - 1, -1, -1):
        shift[pos] = acc
        acc *= radix[pos]
    # Largest packed value, so the int64 bound is checked on the real
    # range rather than on a loose product.
    top = sum((radix[i] - 1) * shift[i] for i in range(n))
    if top > _INT64_MAX:
        raise _err(
            f"{n} key columns need {top + 1} code combinations, "
            f"which does not fit in int64.",
            "group fewer columns at a time, or key on a single "
            "dictionary-coded column",
            doc="specs/delta-2-composite-keys.md")
    return {"mode": "radix", "radix": radix, "shift": shift, "mins": mins}


def _rebase(arr, mode, mins, pos):
    """Key column as the packed node wants to see it."""
    if mode != "radix":
        return arr
    return np.ascontiguousarray(arr.astype(np.int64) - np.int64(mins[pos]))


def composite_group_fused_plan(out, keys, values, ops, group_ops=("count",)):
    """Plan one composite grouping whose groups stay columnar.

    Same arguments and same meaning as `composite_group_plan`; the one
    difference is the requested result shape, so the driver hands back
    arrays and no group is materialised as a Python dict.

    Returns the same plan dict `composite_group_plan` returns, plus a
    "layout" entry describing how to read the per-key-column codes back
    out of the packed group keys, and a "fields" entry listing the
    aggregate field names the plan produces (the vocabulary
    `composite_group_fused_topk` orders by).
    """
    if not isinstance(keys, dict) or not keys:
        raise _err("keys must be a non-empty {name: column} dict.",
                   "pass e.g. {'a': watch_codes, 'b': ip_codes'}")
    if not isinstance(values, dict) or not values:
        raise _err("values must be a non-empty {name: column} dict.",
                   "pass e.g. {'r': is_refresh, 'w': width}")
    if not isinstance(ops, dict) or set(ops) != set(values):
        raise _err(f"ops must be {{col: (op,...)}} for exactly {list(values)}.",
                   "pass e.g. {'r': ('sum',), 'w': ('mean',)}")
    gops = tuple(group_ops)
    if any(o not in _GROUP_OPS for o in gops):
        raise _err(f"group_ops must be a subset of {list(_GROUP_OPS)}, got {list(gops)}.",
                   "count is the only aggregate a grouping yields without a "
                   "value column; every other aggregate belongs in `ops`")
    value_names = list(values)
    norm_ops = {}
    for name in value_names:
        col_ops = tuple(ops[name])
        if not col_ops:
            raise _err(f"ops for '{name}' is empty.", "request at least one aggregate")
        bad = [o for o in col_ops if o not in _VALUE_OPS]
        if bad:
            raise _err(f"unknown aggregate(s) {bad} for '{name}': use {list(_VALUE_OPS)}.",
                       "pass aggregates from sum/count/mean/min/max")
        norm_ops[name] = list(col_ops)
    carrier = value_names[0] if gops else None
    if carrier is not None:
        merged = list(norm_ops[carrier])
        for o in gops:
            if o not in merged:
                merged.append(o)
        norm_ops[carrier] = merged

    key_names = list(keys)
    key_arrays = [_as_int_column(n, keys[n]) for n in key_names]
    layout = _pack_layout(key_arrays)

    jobs = []
    for pos, (name, arr) in enumerate(zip(key_names, key_arrays)):
        packed = _rebase(arr, layout["mode"], layout["mins"], pos)
        jobs.append({"op": "series", "inputs": [],
                     "params": {"values": packed, "dtype": _dtype_name(packed)},
                     "out": name})
    for name in value_names:
        arr = _as_int_column(name, values[name])
        jobs.append({"op": "series", "inputs": [],
                     "params": {"values": arr, "dtype": _dtype_name(arr)},
                     "out": name})

    if layout["mode"] == "none":
        key_out = key_names[0]
    else:
        key_out = f"{out}#keys"
        params = {"mode": layout["mode"]}
        if layout["mode"] == "radix":
            params["radix"] = list(layout["radix"])
        jobs.append({"op": "pack_keys", "inputs": list(key_names),
                     "params": params, "out": key_out})
    jobs.append({"op": "groupby_multi", "inputs": value_names + [key_out],
                 "params": {"ops": norm_ops, "cols": value_names,
                            "result": "carry"},
                 "out": out})

    fields = []
    for name in value_names:
        for op in norm_ops[name]:
            if op == "count" and name == carrier:
                continue
            fields.append(f"{name}.{op}")
    fields.extend(gops)

    return {"jobs": jobs, "out": out, "key_out": key_out,
            "keys": key_names, "values": value_names,
            "group_carrier": carrier, "group_ops": list(gops),
            "fields": fields, "layout": layout}


def _field_vector(carry, plan, field):
    """The per-group values of one aggregate field, as a 1-D array.

    Field names are the same vocabulary `composite_group_read` produces,
    so a caller that already knows how to read a group record needs no
    new naming rules to order by one.
    """
    if field in plan.get("group_ops", ()):
        return np.asarray(carry.counts)
    name, _, op = field.rpartition(".")
    if not name or name not in plan["values"] or op not in _VALUE_OPS:
        raise _err(f"unknown order_by field '{field}'.",
                   f"order by one of the aggregates the plan produces: "
                   f"{plan.get('fields')}")
    if op == "sum":
        return np.asarray(carry.sums[name])
    if op == "mean":
        return np.asarray(carry.means(name))
    if op == "min":
        return np.asarray(carry.mins[name])
    return np.asarray(carry.maxs[name])


def _unpack_keys(packed, layout):
    """Per-key-column codes for the given packed group keys.

    Mirrors the packing the plan asked for, so a decoded key is the code
    the caller passed in, not a rebased one.
    """
    mode = layout["mode"]
    if mode == "none":
        return [np.asarray(packed).astype(np.int64)]
    p = np.asarray(packed, dtype=np.int64)
    if mode == "pack":
        # The driver masks the low column into 32 unsigned bits, so it
        # has to be reinterpreted as int32 before it is widened back --
        # otherwise a negative low code comes back as its unsigned twin.
        lo = (p & _PACK_LO_MASK).astype(np.int32).astype(np.int64)
        return [(p >> _PACK_SHIFT).astype(np.int64), lo]
    radix, shift, mins = layout["radix"], layout["shift"], layout["mins"]
    cols = []
    for pos in range(len(radix)):
        # Same reduction for every column: the packing never multiplies
        # the first column, so no column is a bare leftover quotient.
        rest = p // np.int64(shift[pos])
        cols.append((rest % np.int64(radix[pos])).astype(np.int64))
    return [c + np.int64(m) for c, m in zip(cols, mins)]

def composite_group_fused_topk(carry, plan, order_by, descending=True,
                               limit=10):
    """Pick the `limit` highest (or lowest) groups by one aggregate.

    carry:      the columnar group result the plan's graph produced
                (ukeys / counts / sums / means / mins / maxs).
    plan:       the dict `composite_group_fused_plan` returned.
    order_by:   the aggregate field to order on, in `plan["fields"]`.
    descending: True for `ORDER BY <field> DESC`, False for ASC.
    limit:      the LIMIT value. A parameter, like any other.

    Returns one record per surviving group, most-ordered first:

      {"key": (code, ...), <field>: value, ..., "count": value}

    `key` holds the per-key-column codes as the plan defined them, in
    `plan["keys"]` order. The field names are the same ones
    `composite_group_read` produces, so a caller can read either shape.

    The selection is one partition over the whole group array followed by
    one sort over the k survivors, so the cost tracks k and the group
    count, and no group outside the answer is ever converted to a Python
    value. Groups tied on the ordering aggregate have no defined order
    between them -- SQL does not give them one, and this does not invent
    one; the same input always yields the same answer, and a caller that
    needs a total order sorts the k rows itself.
    """
    if not isinstance(plan, dict) or "layout" not in plan:
        raise _err("plan is not a composite_group_fused_plan result.",
                   "pass the dict returned by composite_group_fused_plan")
    if limit is None:
        raise _err("limit is required.", "pass the query's LIMIT value")
    limit = int(limit)
    if limit < 0:
        raise _err(f"limit {limit} is negative.", "pass a non-negative LIMIT value")

    ukeys = np.asarray(carry.ukeys)
    ngroups = int(ukeys.size)
    values = _field_vector(carry, plan, order_by)

    if ngroups == 0 or limit == 0:
        chosen = np.zeros(0, dtype=np.int64)
    elif limit >= ngroups:
        chosen = np.arange(ngroups, dtype=np.int64)
        chosen = chosen[np.argsort(values[chosen], kind="stable")]
        if descending:
            chosen = chosen[::-1]
    else:
        # Partition by the negated score for a descending take, so both
        # directions run the same vectorised selection; the final order
        # over the k survivors is the only sort in the operation.
        score = -values if descending else values
        chosen = np.argpartition(score, limit - 1)[:limit]
        chosen = chosen[np.argsort(score[chosen], kind="stable")]

    key_cols = _unpack_keys(ukeys[chosen], plan["layout"])
    rows = []
    for i in range(int(chosen.size)):
        row = {}
        for pos, name in enumerate(plan["keys"]):
            row[name] = int(key_cols[pos][i])
        for field in plan["fields"]:
            if field in plan.get("group_ops", ()):
                row[field] = int(carry.counts[chosen[i]])
            else:
                name, _, op = field.rpartition(".")
                if op == "sum":
                    row[field] = int(carry.sums[name][chosen[i]])
                elif op == "mean":
                    row[field] = float(carry.means(name)[chosen[i]])
                elif op == "min":
                    row[field] = int(carry.mins[name][chosen[i]])
                else:
                    row[field] = int(carry.maxs[name][chosen[i]])
        rows.append(row)
    return {"rows": rows, "groups": ngroups, "order_by": order_by,
            "descending": bool(descending), "limit": limit}
