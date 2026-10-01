# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Read a composite group record back as a flat per-group field map.

`composite_group_plan` produces a nested driver record
`{group_key: {measure: {op: value}}}`; the group-level aggregates sit
inside the carrier column's cell. This flattens it to
`{group_key: {"<measure>.<op>": value, ..., "<group_op>": value}}` so a
consumer never needs to know where the group-level values were parked.

Pure read: no aggregation, no data inspection, no branching on content.
"""


def _err(what, fix):
    return ValueError(f"{what} Fix: {fix}.")


def composite_group_read(record, plan):
    """Flatten a composite group record using the plan that produced it.

    record: the value the groupby_multi node produced ({key: {col: {op}}}).
    plan:   the dict returned by composite_group_plan.

    Returns {group_key: {field: value}}, where field is "<measure>.<op>"
    for every requested measure aggregate and the bare group op name
    ("count") for every requested group-level aggregate.
    """
    if not isinstance(plan, dict) or "values" not in plan:
        raise _err("plan is not a composite_group_plan result.",
                   "pass the dict returned by composite_group_plan")
    if not isinstance(record, dict):
        raise _err("record is not a group record dict.",
                   "pass the value of the plan's 'out' buffer")
    value_names = list(plan["values"])
    carrier = plan.get("group_carrier")
    gops = list(plan.get("group_ops") or ())
    out = {}
    for key, cell in record.items():
        fields = {}
        for name in value_names:
            for op, val in cell[name].items():
                fields[f"{name}.{op}"] = val
        if gops:
            src = cell[carrier]
            for op in gops:
                fields[op] = src[op]
        out[key] = fields
    return out
