# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.stream import build_impl as _build_impl
from _lib.stream import open_impl as _open_impl
from _lib.stream import plan_impl as _plan_impl
from _lib.stream import read_block_impl as _read_impl

_box = {}


def nfs_stream_build(csv_path, out_path, block_rows=10000000, limit_rows=None, resume=True,
                     max_blocks=None):
    a = _box["kernel"].alias
    return _build_impl(csv_path, out_path, block_rows, a["resident_prepare"],
                       a.get("format_error"), limit_rows, resume, max_blocks)


def nfs_stream_open(path, budget_frac=0.25, force_lazy=False):
    return _open_impl(path, budget_frac, force_lazy)


def nfs_stream_read_block(handle, idx):
    return _read_impl(handle, idx)


def nfs_stream_plan(handle, budget_frac=0.10):
    return _plan_impl(handle, budget_frac)


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("NfsStream", {})["version"] = "0.1.0"
    kernel.metadata["NfsStream"]["format"] = "nfs-stream-v1"


PUBLIC = {"nfs_stream_build": nfs_stream_build, "nfs_stream_open": nfs_stream_open,
          "nfs_stream_read_block": nfs_stream_read_block,
          "nfs_stream_plan": nfs_stream_plan}
