# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.bitmask_sweep import bitmask_sweep as bitmask_sweep
from _lib.bitmask_sweep import bitmask_sweep_available as bitmask_sweep_available

_box = {}


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("BitmaskSweep", {})["version"] = "0.1.0"
    kernel.metadata["BitmaskSweep"]["types"] = [
        "bitmask_sweep", "bitmask_sweep_available",
    ]


PUBLIC = {"bitmask_sweep": bitmask_sweep, "bitmask_sweep_available": bitmask_sweep_available}
