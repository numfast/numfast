# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.cpu import cpu_capability_impl as cpu_capability
from _lib.cpu import cpu_execute_impl as _cpu_execute_impl

_box = {}


def cpu_execute(nodes):
    a = _box["kernel"].alias
    try:
        _gdh = a["grouped_distinct_hash"]
    except Exception:
        _gdh = None
    try:
        _ghm = a["GroupedHashMiss"]
    except Exception:
        _ghm = None
    return _cpu_execute_impl(nodes, a["accum_dtype"], a["canonical_dtype"], a["format_error"],
                             a["plan_groupby"], a["plan_pack"], a["check_int32_range"],
                             _gdh, _ghm)


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("CPU", {})["version"] = "0.1.0"


PUBLIC = {"cpu_execute": cpu_execute, "cpu_capability": cpu_capability}
