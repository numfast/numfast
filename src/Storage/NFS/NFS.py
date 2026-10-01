# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.nfs import load_impl as _load_impl
from _lib.nfs import persist_impl as _persist_impl

_box = {}


def persist_table(table, path):
    a = _box["kernel"].alias
    return _persist_impl(table, path, a["canonical_dtype"], a["format_error"],
                         a["check_int32_range"])


def load_table(path):
    a = _box["kernel"].alias
    return _load_impl(path, a["canonical_dtype"], a["format_error"])


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("NFS", {})["version"] = "0.1.0"


PUBLIC = {"persist_table": persist_table, "load_table": load_table}
