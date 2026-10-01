# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Series/Table over the packaged kernel (boundary convenience, not execution).

Thin wrappers: every op lowers to IR jobs (ir_series/ir_map/ir_compare/
ir_filter/ir_sort+ir_gather/ir_reduce) and runs compile -> optimize ->
evaluate(backend="cpu"). Mass compute belongs to explicit kernel paths;
these helpers are the verified from_numpy -> op -> to_numpy round-trip.

Internal contract: numfast._lib (Series/Table internals) is NOT part of
the public API and carries no backward-compat promise; the physical
import is NOT blocked, but new code must use only the public nf.*
names listed in numfast.__all__.
"""

from .series import Series
from .table import Table

__all__ = ["Series", "Table"]
