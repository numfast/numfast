"""GroupBy Extension — Builder entry point."""
try:
    from GroupBy._lib.groupby import GroupBy, groupby, groupby_array, SCAN_LIMIT, SORT_LIMIT, GROUPBY_LIMIT
except ImportError:
    from _lib.groupby import GroupBy, groupby, groupby_array, SCAN_LIMIT, SORT_LIMIT, GROUPBY_LIMIT

__all__ = ["GroupBy", "groupby", "groupby_array", "SCAN_LIMIT", "SORT_LIMIT", "GROUPBY_LIMIT"]

def setup(kernel):
    """Register in kernel metadata."""
    kernel.metadata.setdefault("GroupBy", {})
    kernel.metadata["GroupBy"]["version"] = "0.1.0"
    kernel.metadata["GroupBy"]["types"] = ["groupby"]
