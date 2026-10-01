# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.adjacency import adjacency_flat as adjacency_flat
from _lib.adjacency import adjacency_slice as adjacency_slice
from _lib.segmented import segmented_reduce as segmented_reduce

_box = {}


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("Segmented", {})["version"] = "0.1.0"
    kernel.metadata["Segmented"]["types"] = [
        "segmented_reduce", "adjacency_slice", "adjacency_flat",
    ]
    kernel.metadata["Segmented"]["max_dispatch_n"] = 4194240


PUBLIC = {"segmented_reduce": segmented_reduce,
          "adjacency_slice": adjacency_slice,
          "adjacency_flat": adjacency_flat}
