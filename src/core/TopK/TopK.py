"""TopK Extension — Builder entry point."""
try:
    from TopK._lib.topk import TopK, topk, topk_array, SCAN_LIMIT, SORT_LIMIT
except ImportError:
    from _lib.topk import TopK, topk, topk_array, SCAN_LIMIT, SORT_LIMIT

__all__ = ["TopK", "topk", "topk_array", "SCAN_LIMIT", "SORT_LIMIT"]

def setup(kernel):
    """Register in kernel metadata."""
    kernel.metadata.setdefault("TopK", {})
    kernel.metadata["TopK"]["version"] = "0.1.0"
    kernel.metadata["TopK"]["types"] = ["TopK"]
