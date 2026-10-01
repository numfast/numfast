# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.distinct import count_distinct as count_distinct

_box = {}


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("DistinctCount", {})["version"] = "0.1.0"
    kernel.metadata["DistinctCount"]["types"] = [
        "count_distinct",
    ]


PUBLIC = {"count_distinct": count_distinct}
