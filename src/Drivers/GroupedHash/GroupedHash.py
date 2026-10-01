# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.grouped_hash import GroupedHashMiss as GroupedHashMiss
from _lib.grouped_hash import grouped_distinct_hash as grouped_distinct_hash

_box = {}


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("GroupedHash", {})["version"] = "0.1.0"
    kernel.metadata["GroupedHash"]["types"] = [
        "grouped_distinct_hash", "GroupedHashMiss",
    ]


PUBLIC = {"grouped_distinct_hash": grouped_distinct_hash,
          "GroupedHashMiss": GroupedHashMiss}
