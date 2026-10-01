# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Single-pass composite group plan (generic: any keys, any aggregates).

One grouping traversal produces EVERY aggregate of one grouping, so the
traversal -- the expensive part -- is paid once per query instead of once
per aggregate. The plan is a pure job-graph builder: it emits `series` /
`pack_keys` / `groupby_multi` nodes and nothing else. No execution, no
aggregation, no NumPy: the traversal itself is the frozen CPU driver's
generic composite path (pack_keys tuple sidecar -> validity AND ->
lex-sorted unique tuple keys -> per-group int64 sums + counts), reused
verbatim.

Grouping keys are N integer columns (any N >= 1) packed with
`pack_keys(mode='hash')`, whose tuple sidecar is what routes the driver
to that composite traversal. The traversal already yields the per-group
row count, so `count` travels with the same pass and never needs a
second graph.

Genericity: nothing here knows about datasets, column names, column
counts, cardinalities, row counts, limits or specific queries. Every
decision is taken from the argument shapes (how many keys, which ops
were asked for), never from the data values.

Known limits of the reused traversal, stated rather than worked around:
  * the composite lane supports sum / count / mean; min / max are single
    column only in the frozen driver, so asking for them is an explicit
    error instead of a silent second traversal;
  * the driver materialises group values as a Python dict keyed by
    tuple, so per-group unpacking cost is inherent to it and is not
    hidden or special-cased here.
"""

_VALUE_OPS = ("sum", "count", "mean", "min", "max")
# Aggregates the composite traversal produces for every group on its own,
# without a value column. Only the group row count qualifies today.
_GROUP_OPS = ("count",)


def _err(what, fix, doc="specs/delta-1-fused-aggregate.md"):
    return ValueError(f"{what} Fix: {fix}. See {doc}")


def _dtype_of(name, col):
    """Physical dtype for a `series` node, derived from the column itself.

    Integer code/measure columns stay int32 (the engine's logical width),
    anything else is carried as float64. No coercion happens here: the
    caller owns narrowing, the plan only records the dtype it saw.
    """
    dt = getattr(col, "dtype", None)
    if dt is None:
        return "int32"
    kind = getattr(dt, "kind", "")
    if kind in "iub":
        return "int32"
    if kind == "f":
        return "float64"
    raise _err(f"column '{name}' has dtype kind '{kind}', not int/uint/bool/float.",
               "pass integer code columns for keys and int/float measure columns")


def composite_group_plan(out, keys, values, ops, group_ops=("count",)):
    """Plan one composite grouping that computes all of its aggregates.

    keys:      {name: 1-D integer column}, >= 1 grouping column.
    values:    {name: 1-D int/float column}, >= 1 measure column.
    ops:       {value name: non-empty subset of sum/count/mean/min/max},
               keys exactly equal to `values`.
    group_ops: group-level aggregates the traversal yields for free
               (only "count" today), i.e. aggregates that need no value
               column at all -- COUNT(*) is the motivating case.

    Returns a plan dict:
      jobs           -- list of IR job dicts, one groupby_multi carrying
                        every requested aggregate
      out            -- name of the resulting group record
      key_out        -- name of the packed key node
      values         -- measure column names, in order
      group_carrier  -- measure column that carries the group-level ops
      group_ops      -- the group-level ops that were requested

    Insertion order of `values` decides the carrier deterministically; no
    content of any column influences the plan.
    """
    if not isinstance(keys, dict) or not keys:
        raise _err("keys must be a non-empty {name: column} dict.",
                   "pass e.g. {'a': watch_codes, 'b': ip_codes}")
    if not isinstance(values, dict) or not values:
        raise _err("values must be a non-empty {name: column} dict.",
                   "pass e.g. {'r': is_refresh, 'w': width}")
    if not isinstance(ops, dict) or set(ops) != set(values):
        raise _err(f"ops must be {{col: (op,...)}} for exactly {list(values)}.",
                   "pass e.g. {'r': ('sum',), 'w': ('mean',)}")
    gops = tuple(group_ops)
    if any(o not in _GROUP_OPS for o in gops):
        raise _err(f"group_ops must be a subset of {list(_GROUP_OPS)}, got {list(gops)}.",
                   "count is the only aggregate the traversal yields without a "
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
        unsupported = [o for o in col_ops if o in ("min", "max")]
        if unsupported:
            raise _err(f"composite grouping does not support {unsupported}.",
                       "min/max are single-column only in the composite lane; "
                       "use them through a single-column grouping",
                       doc="specs/delta-2-composite-keys.md")
        norm_ops[name] = list(col_ops)
    # Group-level aggregates ride on one measure column: the frozen
    # composite groupby_multi materialises group values inside a measure
    # column's cell, and the traversal computes them once regardless of
    # which cell asks for them. First column in caller order = carrier.
    carrier = value_names[0] if gops else None
    if carrier is not None:
        merged = list(norm_ops[carrier])
        for o in gops:
            if o not in merged:
                merged.append(o)
        norm_ops[carrier] = merged

    key_names = list(keys)
    key_out = f"{out}#keys"
    jobs = [{"op": "series", "inputs": [],
             "params": {"values": keys[n], "dtype": _dtype_of(n, keys[n])},
             "out": n} for n in key_names]
    jobs.extend({"op": "series", "inputs": [],
                 "params": {"values": values[n], "dtype": _dtype_of(n, values[n])},
                 "out": n} for n in value_names)
    jobs.append({"op": "pack_keys", "inputs": key_names,
                 "params": {"mode": "hash"}, "out": key_out})
    jobs.append({"op": "groupby_multi", "inputs": value_names + [key_out],
                 "params": {"ops": norm_ops, "cols": value_names}, "out": out})
    return {"jobs": jobs, "out": out, "key_out": key_out,
            "keys": key_names, "values": value_names,
            "group_carrier": carrier, "group_ops": list(gops)}
