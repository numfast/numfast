# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.pair_insert import pair_insert as pair_insert
from _lib.pair_insert import pair_insert_available as pair_insert_available

_box = {}


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("PairInsert", {})["version"] = "0.1.0"
    kernel.metadata["PairInsert"]["types"] = [
        "pair_insert", "pair_insert_available",
    ]


PUBLIC = {"pair_insert": pair_insert, "pair_insert_available": pair_insert_available}
