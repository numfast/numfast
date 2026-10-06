# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Shift-packed composite grouping: every key column in ONE int64.

`composite_group_fused_plan` hands the packing to the frozen driver's
`pack_keys`, which has to keep a tuple sidecar for a wide composite and
materialises the group values as Python dict cells before the caller can
look at one group. On a grouping whose group count is the same order as
the row count, that per-group Python object is the whole cost of the
query.

This module is the third shape of the same operation. Nothing is asked of
the driver: the grouping is done here, with vector primitives only, and it
never leaves the array domain.

    1. measure    -- one observed min/max per key column
    2. pack       -- every column rebased onto its own minimum, then
                     concatenated into a single non-negative int64
    3. order      -- one argsort of that int64
    4. boundary   -- one comparison of the sorted key against itself
    5. counts     -- one diff of the boundary positions
    6. aggregates -- one gather plus one `reduceat` per measure column,
                     and only for the aggregates that were asked for
    7. order by   -- one partition over the group vectors, then one sort
                     over the k survivors

The k groups that survive the partition are the only rows that become
Python values. Everything before that is an array.

Genericity, stated as rules this module holds itself to:

  * `limit` is an operation parameter, read from the caller's argument.
    It is never a constant and never clamped.
  * Number of key columns, the ops map and `dtype.kind` are the only
    arguments that select work. No branch on a column name, a table, a
    query number, on "groups == rows", on a tie plateau, on a result size
    or on dataset scale.
  * The one place a column is measured is step 1, and the numbers it
    produces are packing parameters (bit widths and rebasing offsets),
    recorded in the plan -- not a control-flow decision. The single
    exception is the packing-width refusal below, which is a bound on the
    packing, not a heuristic.

Overflow safety
---------------
A packed key must be *provably* injective over signed inputs, so:

  * each column is rebased onto its own observed minimum, so every code
    is non-negative and no column is ever masked into an unsigned twin of
    itself -- the failure mode where a negative low code comes back as
    its unsigned twin (code -39 read back as 4294967257) cannot occur,
    because there is no mask and no signed/unsigned reinterpretation
    anywhere in this path;
  * each column gets `bit_length(max - min)` bits, which is exactly enough
    for every code in that column and never one bit more, so the field
    widths are minimal and the total is as small as the data allows;
  * the total must fit under `PACK_BITS_MAX`, so the packed value stays in
    `[0, 2**63)` -- non-negative by construction, so the decoder's shift
    is a logical one and the largest shift never reaches the sign bit;
  * if the total does not fit, the module REFUSES: it does not truncate
    widths, does not hash, and does not collide. `composite_group_shift_plan`
    reports `"lane": "tuple"` and carries the existing composite-tuple
    graph instead, so the query still runs -- correct, and slower -- and
    the caller can see which lane was taken.

Injectivity follows, not merely approximately: the map
`value_i -> value_i - min_i` is injective per column, a tuple of
non-negative integers with fixed field widths has exactly one packing, and
the fixed widths are wide enough for every code. Therefore equal packed
values imply equal value tuples, and a decoder that shifts out each
field and adds the minimum back recovers the original value.

Known limit, stated rather than worked around: `kind='stable'` on an
int64 key is what the measured numbers were taken with, and its speed
depends on how the keys are distributed -- a wide field makes it
noticeably slower than `quicksort` on uniform random keys and
noticeably faster on clustered ones. It is a constant here, not a
measured branch.
"""

import time

import numpy as np

# Same Extension, own _lib: the fallback reuses the graph this extension
# already planned before the packed lane existed. Imported here, not inside
# the branch, because the loader unmounts `_lib` once the entry module has
# been executed -- by the time a call arrives, the import can no longer be
# resolved.
from _lib.plan import composite_group_plan

_VALUE_OPS = ("sum", "count", "mean", "min", "max")
_GROUP_OPS = ("count",)

# int64 carries 63 value bits. Every code here is non-negative (each
# column is rebased onto its own minimum), so a packed key is a plain
# unsigned quantity that happens to live in an int64: at most 63 field
# bits puts it at 2**63 - 1, exactly int64's largest value, the largest
# left shift lands one below the sign bit, and no shift can overflow.
# Nothing in this path is ever a signed/unsigned reinterpretation, which
# is what makes 63 safe where 64 would not be.
PACK_BITS_MAX = 63


def _err(what, fix, doc=""):
    return ValueError(f"{what} Fix: {fix}."
           + (f" See {doc}" if doc else ""))


def _norm_ops(values, ops, group_ops):
    """Validated ops map, carrier merge and group-op list.

    Same contract as `composite_group_fused_plan`: one carrier column
    carries the group-level aggregates so the traversal computes them
    once, and a field is named "<measure>.<op>" or by the bare group-op
    name. Insertion order of `values` picks the carrier, so the choice is
    deterministic and nothing about the data takes part in it.
    """
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
    return norm_ops, carrier, list(gops)


def _as_int_column(name, col):
    """A key column as a flat integer array.

    Only integer columns can be packed losslessly into an int64 code, so
    this refuses anything else instead of rounding it into the key space.
    """
    arr = np.asarray(col)
    if arr.ndim != 1:
        arr = arr.ravel()
    if arr.dtype.kind not in "iub":
        raise _err(f"column '{name}' has dtype kind '{arr.dtype.kind}', not int/uint/bool.",
                   "pass integer code columns for keys")
    return arr


def shift_layout(key_arrays):
    """Field widths and rebasing offsets for a set of key columns.

    The widths are a property of the columns' observed ranges, so this is
    the one measurement the lane takes. It returns, in key-column order:

      mins       per-column rebasing offset (the observed minimum)
      bits       per-column field width, `bit_length(max - min)`
      shifts     per-column left shift, i.e. the width of every LATER
                 field -- so column 0 is the most significant one and
                 the last column sits in the low bits
      total_bits sum of the widths
      fits       whether the total fits the int64 budget

    A constant column gets width 0: it has one value, so it needs no
    field and dropping it costs nothing. Column order is the driver's own
    `acc = acc*radix[i] + c_i` order -- column i is weighted by the
    product of the LATER radices -- with each radix rounded up to a power
    of two, so a field width is the mixed-radix weight written in binary.
    """
    mins, bits = [], []
    for arr in key_arrays:
        if arr.size:
            lo = int(arr.min())
            hi = int(arr.max())
        else:
            lo = hi = 0
        mins.append(lo)
        bits.append(int(hi - lo).bit_length())
    shifts = [0] * len(bits)
    acc = 0
    for pos in range(len(bits) - 1, -1, -1):
        shifts[pos] = acc
        acc += bits[pos]
    total = int(sum(bits))
    return {"mins": mins, "bits": bits, "shifts": shifts,
            "total_bits": total, "fits": bool(total <= PACK_BITS_MAX)}


def _pack(key_arrays, layout):
    """Rebase every key column onto its own minimum, then concatenate.

    One buffer per column is allocated and reused as the shift temporary,
    so the pack costs one allocation per column and no chain of temporaries.
    """
    n = int(key_arrays[0].size) if key_arrays else 0
    packed = None
    for pos, arr in enumerate(key_arrays):
        if layout["bits"][pos] == 0:
            continue
        code = np.array(arr, dtype=np.int64, copy=True)
        np.subtract(code, np.int64(layout["mins"][pos]), out=code)
        shift = layout["shifts"][pos]
        if shift:
            np.left_shift(code, np.int64(shift), out=code)
        if packed is None:
            packed = code
        else:
            np.bitwise_or(packed, code, out=packed)
    if packed is None:
        packed = np.zeros(n, dtype=np.int64)
    return packed


def _decode(packed, layout):
    """Per-key-column values back out of packed keys, one column per call.

    The mirror of `_pack`: shift the field down, mask it, add the
    rebasing offset back. A width-0 field decodes to its single possible
    value, which is exactly the offset.
    """
    masks = [(1 << b) - 1 for b in layout["bits"]]
    cols = []
    for pos, shift in enumerate(layout["shifts"]):
        code = (packed >> np.int64(shift)) & np.int64(masks[pos])
        cols.append(code + np.int64(layout["mins"][pos]))
    return cols


def composite_group_shift_plan(out, keys, values, ops, group_ops=("count",)):
    """Plan a shift-packed composite grouping that never materialises a group.

    Same arguments and same meaning as `composite_group_plan` and
    `composite_group_fused_plan`; the third and last shape of the same
    operation.

    Returns a plan dict:

      lane       "shift" -- this lane runs it
                 "tuple" -- the key space does not fit int64, so the
                            existing composite-tuple graph is carried in
                            "jobs" instead and the caller runs that
      reason     why "tuple" was chosen, "" for "shift"
      jobs       the existing composite-tuple graph; empty on "shift"
      keys       key column names, in packing order
      values     measure column names
      ops        normalised {measure: [op, ...]}
      agg_ops    {measure: [op, ...]} -- the aggregates to compute, after
                 the group-level count and the carrier's own count have
                 been merged into one field
      fields     the aggregate field names this plan produces
      layout     widths / offsets / shifts, from the observed key ranges
      key_arrays the key columns in `keys` order
      value_arrays the measure columns in `values` order

    `composite_group_shift_run` executes the plan; `composite_group_shift_topk`
    reads the result. The two lanes are decided here and nowhere else, and
    the caller always sees which one it got.
    """
    if not isinstance(keys, dict) or not keys:
        raise _err("keys must be a non-empty {name: column} dict.",
                   "pass e.g. {'a': first_code, 'b': second_code}")
    if not isinstance(values, dict) or not values:
        raise _err("values must be a non-empty {name: column} dict.",
                   "pass e.g. {'r': is_refresh, 'w': width}")
    if not isinstance(ops, dict) or set(ops) != set(values):
        raise _err(f"ops must be {{col: (op,...)}} for exactly {list(values)}.",
                   "pass e.g. {'r': ('sum',), 'w': ('mean',)}")
    norm_ops, carrier, gops = _norm_ops(values, ops, group_ops)

    key_names = list(keys)
    key_arrays = [_as_int_column(n, keys[n]) for n in key_names]
    sizes = {int(a.size) for a in key_arrays}
    if len(sizes) != 1:
        raise _err(f"key columns have different lengths: {sorted(sizes)}.",
                   "every key column must have one row per row of the table")
    value_names = list(values)
    value_arrays = [_as_int_column(n, values[n]) for n in value_names]
    vsizes = {int(a.size) for a in value_arrays}
    if len(vsizes) != 1 or vsizes != sizes:
        raise _err(f"measure columns have lengths {sorted(vsizes)} against "
                   f"key length {sorted(sizes)}.",
                   "every key and measure column must have the same rows")

    layout = shift_layout(key_arrays)
    if layout["fits"]:
        lane, reason, jobs = "shift", "", []
    else:
        # Refuse rather than truncate. The widths stay exactly what the
        # data needs; the query goes down the existing composite-tuple
        # path instead, which is exact and slower. Not pack_keys(mode=
        # 'hash') as the fast lane -- this is the fallback, and it is the
        # graph this extension already planned before the packed lane
        # existed.
        lane = "tuple"
        reason = (f"{len(key_names)} key columns need {layout['total_bits']} bits, "
                  f"over the {PACK_BITS_MAX}-bit int64 packing budget")
        jobs = composite_group_plan(out, keys, values, ops, group_ops)["jobs"]

    # The carrier's own `count` and the group-level `count` are the same
    # number, so it is produced once and named once, exactly as the other
    # lanes in this extension name it.
    agg_ops = {}
    for name in value_names:
        keep = [op for op in norm_ops[name]
                if not (op == "count" and name == carrier and "count" in gops)]
        if keep:
            agg_ops[name] = keep
    fields = []
    for name in value_names:
        for op in agg_ops.get(name, ()):
            fld = f"{name}.{op}"
            if fld not in fields:
                fields.append(fld)
    for op in gops:
        if op not in fields:
            fields.append(op)

    return {"lane": lane, "reason": reason, "jobs": jobs, "out": out,
            "keys": key_names, "values": value_names, "ops": norm_ops,
            "agg_ops": agg_ops, "fields": fields, "layout": layout,
            "key_arrays": key_arrays, "value_arrays": value_arrays,
            "group_ops": gops}


def _now_ms():
    return time.perf_counter() * 1000.0


def composite_group_shift_run(plan, stages=None):
    """Execute a shift-packed plan; return the groups as arrays.

    plan:   the dict `composite_group_shift_plan` returned.
    stages: optional dict. When a dict is passed, the elapsed
            milliseconds of each stage are recorded in it, keyed by stage
            name -- a measurement hook, off by default and never part of
            a decision.

Returns, for lane "shift":

      lane      "shift"
      n         rows
      groups    number of distinct keys
      fields    {field name: 1-D array over groups}
      counts    rows per group
      sorted_keys
                the packed int64 key of every row, in group order, so
                group g is `sorted_keys[starts[g]:starts[g + 1]]`
      starts    first position of each group -- group g's own packed key
                is `sorted_keys[starts[g]]`
      keys      key column names, in packing order
      layout    the widths / offsets / shifts the keys were packed with

    Nothing in the result is a per-group Python object: every group is
    still an element of an array, and the key columns stay packed because
    decoding is only worth doing for the k groups that survive the
    selection. `composite_group_shift_topk` is what turns those k groups
    into values.

    For lane "tuple" the keys did not fit int64; nothing is executed here
    and the existing composite-tuple graph comes back under "jobs" for
    the caller to run the way it always has.
    """
    if not isinstance(plan, dict) or "lane" not in plan:
        raise _err("plan is not a composite_group_shift_plan result.",
                   "pass the dict returned by composite_group_shift_plan")
    if plan["lane"] != "shift":
        return {"lane": plan["lane"], "reason": plan["reason"],
                "jobs": plan["jobs"], "groups": None}

    timing = stages if isinstance(stages, dict) else None

    t = _now_ms()
    packed = _pack(plan["key_arrays"], plan["layout"])
    if timing is not None:
        timing["pack_ms"] = _now_ms() - t

    n = int(packed.size)
    if n == 0:
        empty64 = np.zeros(0, dtype=np.int64)
        out = {"lane": "shift", "n": 0, "groups": 0,
               "fields": {f: empty64 for f in plan["fields"]},
               "counts": empty64, "sorted_keys": packed, "starts": empty64,
               "keys": plan["keys"], "layout": plan["layout"]}
        if timing is not None:
            for stage in ("order_ms", "boundary_ms", "counts_ms", "aggregate_ms"):
                timing.setdefault(stage, 0.0)
        return out

    t = _now_ms()
    order = np.argsort(packed, kind="stable")
    if timing is not None:
        timing["order_ms"] = _now_ms() - t

    t = _now_ms()
    ordered = packed[order]
    changed = np.empty(n, dtype=bool)
    changed[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=changed[1:])
    starts = np.flatnonzero(changed)
    if timing is not None:
        timing["boundary_ms"] = _now_ms() - t

    t = _now_ms()
    counts = np.diff(starts, append=n).astype(np.int64)
    ngroups = int(starts.size)
    if timing is not None:
        timing["counts_ms"] = _now_ms() - t

    t = _now_ms()
    fields = {}
    for op in plan["group_ops"]:
        fields[op] = counts
    for pos, name in enumerate(plan["values"]):
        want = plan["agg_ops"].get(name)
        if not want:
            continue
        base = plan["value_arrays"][pos]
        # int32 logical values, int64 accumulators; a float measure stays
        # float64 so the sum keeps the caller's precision.
        acc_dt = np.float64 if base.dtype.kind == "f" else np.int64
        sorted_vals = np.asarray(base, dtype=acc_dt)[order]
        if "sum" in want or "mean" in want:
            total = np.add.reduceat(sorted_vals, starts)
            if "sum" in want:
                fields[f"{name}.sum"] = total
            if "mean" in want:
                fields[f"{name}.mean"] = (total.astype(np.float64)
                                         / counts.astype(np.float64))
        if "min" in want:
            fields[f"{name}.min"] = np.minimum.reduceat(sorted_vals, starts)
        if "max" in want:
            fields[f"{name}.max"] = np.maximum.reduceat(sorted_vals, starts)
        if "count" in want:
            fields[f"{name}.count"] = counts
    if timing is not None:
        timing["aggregate_ms"] = _now_ms() - t

    return {"lane": "shift", "n": n, "groups": ngroups, "fields": fields,
            "counts": counts, "sorted_keys": ordered, "starts": starts,
            "keys": plan["keys"], "layout": plan["layout"]}


def _scalar(arr, i):
    """One group value as a Python scalar of the aggregate's own kind."""
    val = arr[i]
    return float(val) if np.asarray(arr).dtype.kind == "f" else int(val)


def composite_group_shift_topk(result, order_by, limit, descending=True):
    """Pick the `limit` highest (or lowest) groups by one aggregate.

    result:      what `composite_group_shift_run` returned.
    order_by:    an aggregate field name, one of the result's fields.
    limit:       the query's LIMIT value, and a required argument -- this
                 function has no default k, so k is always the caller's
                 number and never a constant baked into the operation.
    descending:  True for `ORDER BY <field> DESC`, False for ASC.

    Returns the same record `composite_group_fused_topk` returns:

      {"rows": [{"key": ..., <field>: value, ...}, ...],
       "groups": <group count>, "order_by": ..., "descending": ...,
       "limit": ..., "lane": ...}

    The selection is one partition over the whole group vector followed by
    one sort over the k survivors, so the cost tracks the group count and
    k, and only k groups are ever decoded. Groups tied on the ordering
    aggregate have no defined order between them -- SQL does not give
    them one and this does not invent one; the same input always yields
    the same answer.
    """
    if not isinstance(result, dict) or result.get("lane") != "shift":
        raise _err("result is not a composite_group_shift_run result.",
                   "only the packed lane carries columnar groups; run the "
                   "plan's 'jobs' for the composite-tuple fallback")
    if limit is None:
        raise _err("limit is required.", "pass the query's LIMIT value")
    limit = int(limit)
    if limit < 0:
        raise _err(f"limit {limit} is negative.", "pass a non-negative LIMIT value")
    fields = result["fields"]
    if order_by not in fields:
        raise _err(f"unknown order_by field '{order_by}'.",
                   f"order by one of the aggregates the plan produced: "
                   f"{list(fields)}")

    values = np.asarray(fields[order_by])
    ngroups = int(values.size)
    if ngroups == 0 or limit == 0:
        chosen = np.zeros(0, dtype=np.int64)
    elif limit >= ngroups:
        chosen = np.arange(ngroups, dtype=np.int64)
        chosen = chosen[np.argsort(values[chosen], kind="stable")]
        if descending:
            chosen = chosen[::-1]
    else:
        # One partition either way: the descending take negates its score
        # first, so both directions run the same vectorised selection and
        # the only sort is the final one over the k survivors.
        score = -values if descending else values
        chosen = np.argpartition(score, limit - 1)[:limit]
        chosen = chosen[np.argsort(score[chosen], kind="stable")]

    pk = result["sorted_keys"][result["starts"][chosen]]
    key_cols = _decode(pk, result["layout"])
    rows = []
    for i in range(int(chosen.size)):
        row = {"key": tuple(int(c[i]) for c in key_cols)}
        for pos, name in enumerate(result["keys"]):
            row[name] = int(key_cols[pos][i])
        for field, arr in fields.items():
            row[field] = _scalar(arr, chosen[i])
        rows.append(row)
    return {"rows": rows, "groups": ngroups, "order_by": order_by,
            "descending": bool(descending), "limit": limit,
            "lane": result["lane"]}
