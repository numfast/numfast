"""ZindiGaps Extension — GPU searchsorted + backward-anchor features.

Public API (alias -> mod):
    anchors       — history build for train/test frames
    searchsorted  — GPU binary search in sorted (cell, month) index
"""
from ZindiGaps._lib.anchors import anchors, searchsorted


def setup(kernel):
    """Register in kernel metadata."""
    kernel.metadata.setdefault("ZindiGaps", {})
    kernel.metadata["ZindiGaps"]["version"] = "0.1.0"
    kernel.metadata["ZindiGaps"]["types"] = ["anchors", "searchsorted"]


__all__ = ["anchors", "searchsorted"]