"""Filter Extension -- Builder entry point.

Filter(src, cond) = Gather(src, indices where cond !=0)
indices = Scan(Compare(cond,0)→0/1) stable order, chunkable 4_194_240
Select(cond,a,b) = Mask alias.
"""

from _lib.filter import Filter, filter_array
from _lib.select import Select, select_array

__all__ = ["Filter", "filter_array", "Select", "select_array"]

def setup(kernel):
    """Register Filter in kernel metadata."""
    kernel.metadata.setdefault("Filter", {})
    kernel.metadata["Filter"]["version"] = "0.1.0"
    kernel.metadata["Filter"]["types"] = ["Filter", "Select"]
    kernel.metadata["Filter"]["alias"] = "Filter"
    kernel.metadata["Filter"]["description"] = "Filter=Gather(Scan(Compare)); Select=Mask; chunkable 4194240"
    try:
        kernel.Filter = Filter
        kernel.Select = Select
    except Exception:
        pass
