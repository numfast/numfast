# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Single-pass composite grouping: one traversal for every aggregate."""

from _lib.fused import composite_group_fused_plan as composite_group_fused_plan
from _lib.fused import composite_group_fused_topk as composite_group_fused_topk
from _lib.plan import composite_group_plan as composite_group_plan
from _lib.read import composite_group_read as composite_group_read
from _lib.shifted import composite_group_shift_plan as composite_group_shift_plan
from _lib.shifted import composite_group_shift_run as composite_group_shift_run
from _lib.shifted import composite_group_shift_topk as composite_group_shift_topk


def setup(kernel):
    kernel.metadata.setdefault("CompositeGroup", {})["version"] = "0.1.0"
    kernel.metadata["CompositeGroup"]["types"] = [
        "composite_group_plan", "composite_group_read",
        "composite_group_fused_plan", "composite_group_fused_topk",
        "composite_group_shift_plan", "composite_group_shift_run",
        "composite_group_shift_topk",
    ]


PUBLIC = {"composite_group_plan": composite_group_plan,
          "composite_group_read": composite_group_read,
          "composite_group_fused_plan": composite_group_fused_plan,
          "composite_group_fused_topk": composite_group_fused_topk,
          "composite_group_shift_plan": composite_group_shift_plan,
          "composite_group_shift_run": composite_group_shift_run,
          "composite_group_shift_topk": composite_group_shift_topk}
