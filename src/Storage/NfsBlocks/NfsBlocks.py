# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.blocks import build_impl as _build_impl
from _lib.blocks import open_impl as _open_impl
from _lib.blocks import read_block_impl as _read_impl

_box = {}


def nfs_build(csv_path, out_dir, block_rows=10000000, limit_rows=None, resume=True):
    a = _box["kernel"].alias
    return _build_impl(csv_path, out_dir, block_rows, a["resident_prepare"],
                       a.get("format_error"), limit_rows, resume)


def nfs_open(out_dir, cache_blocks=3):
    return _open_impl(out_dir, cache_blocks)


def nfs_read_block(handle, idx):
    return _read_impl(handle, idx)


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("NfsBlocks", {})["version"] = "0.1.0"
    kernel.metadata["NfsBlocks"]["format"] = "nfs-block-v1"


PUBLIC = {"nfs_build": nfs_build, "nfs_open": nfs_open, "nfs_read_block": nfs_read_block}
