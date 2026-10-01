# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""GroupBy member-marker (SPEC 12): no code, Planner intention only."""

_box = {}


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("GroupBy", {})["version"] = "0.1.0"


PUBLIC = {}
