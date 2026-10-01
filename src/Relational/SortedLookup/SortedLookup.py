# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.sorted_lookup_i64 import isin_i64 as isin_i64
from _lib.sorted_lookup_i64 import lookup_i64 as lookup_i64
from _lib.sorted_lookup_i64 import lookup_positions as lookup_positions_i64

_box = {}


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("SortedLookup", {})["version"] = "0.1.0"
    kernel.metadata["SortedLookup"]["types"] = ["lookup_positions_i64", "isin_i64", "lookup_i64"]


PUBLIC = {"lookup_positions_i64": lookup_positions_i64, "isin_i64": isin_i64, "lookup_i64": lookup_i64}
