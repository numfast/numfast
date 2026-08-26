"""ISA Extension -- compile-time lowering to Compute primitives.

Usage:
    from ISA._lib.lowering import lower, format_jobs

    ast = parse('SMA(close, 14)')
    d = ast_to_dict(ast)
    graph, out = lower(d)
    jobs = format_jobs(graph, out)
"""

from _lib.lowering import lower, format_jobs


def setup(kernel):
    """Register in kernel metadata."""
    kernel.metadata.setdefault("ISA", {})
    kernel.metadata["ISA"]["version"] = "0.1.0"
