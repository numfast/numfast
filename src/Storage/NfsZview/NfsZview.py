# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.zview import read_block_impl as _read_impl

_box = {}


def nfs_zview_read_block(handle, idx, mode="hybrid"):
    a = _box["kernel"].alias
    return _read_impl(handle, idx, mode=mode,
                      fallback=a.get("nfs_stream_read_block"))


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("NfsZview", {})["version"] = "0.1.0"
    kernel.metadata["NfsZview"]["format"] = "nfs-stream-v1"
    kernel.metadata["NfsZview"]["reader"] = "zview"


PUBLIC = {"nfs_zview_read_block": nfs_zview_read_block}
