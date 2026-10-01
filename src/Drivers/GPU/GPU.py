# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.gpu import gpu_capability_impl as gpu_capability
from _lib.gpu import gpu_execute_impl as _gpu_execute_impl
from _lib.gpu import gpu_vram_budget as _gpu_vram_budget
from _lib.gpu import gpu_chunked_execute as _gpu_chunked_execute

_box = {}


def gpu_execute(nodes):
    a = _box["kernel"].alias
    return _gpu_execute_impl(nodes, a["canonical_dtype"], a["format_error"])


def gpu_vram_budget():
    return _gpu_vram_budget()


def gpu_chunked_execute(nodes, n, chunk_size):
    a = _box["kernel"].alias
    return _gpu_chunked_execute(nodes, n, chunk_size, a["format_error"])


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("GPU", {})["version"] = "0.2.0"


PUBLIC = {"gpu_execute": gpu_execute, "gpu_capability": gpu_capability,
          "gpu_vram_budget": gpu_vram_budget,
          "gpu_chunked_execute": gpu_chunked_execute}
