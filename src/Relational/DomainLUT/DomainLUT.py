# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""D-scale dictionary predicate LUTs and their row-side gather mask."""

from _lib.row_mask import codes_lut_mask as codes_lut_mask
from _lib.text_lut import dict_contains_lut as dict_contains_lut
from _lib.text_lut import dict_equal_lut as dict_equal_lut
from _lib.text_lut import dict_not_contains_lut as dict_not_contains_lut
from _lib.text_lut import dict_not_equal_lut as dict_not_equal_lut


def setup(kernel):
    kernel.metadata.setdefault("DomainLUT", {})["version"] = "0.1.0"
    kernel.metadata["DomainLUT"]["types"] = [
        "dict_contains_lut", "dict_equal_lut", "dict_not_equal_lut",
        "dict_not_contains_lut", "codes_lut_mask",
    ]


PUBLIC = {
    "dict_contains_lut": dict_contains_lut,
    "dict_equal_lut": dict_equal_lut,
    "dict_not_equal_lut": dict_not_equal_lut,
    "dict_not_contains_lut": dict_not_contains_lut,
    "codes_lut_mask": codes_lut_mask,
}
