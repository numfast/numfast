# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Exports only (no code)."""
try:
    from .adjacency import adjacency_flat, adjacency_slice  # noqa: F401
except ImportError:
    pass
try:
    from .segmented import MAX_DISPATCH_N, segmented_reduce  # noqa: F401
except ImportError:
    pass

__all__ = ["MAX_DISPATCH_N", "adjacency_flat", "adjacency_slice",
           "segmented_reduce"]
