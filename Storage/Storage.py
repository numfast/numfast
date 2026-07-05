"""Storage Extension — Column, PackingPlan, Table, Optimize.

Builder entry point. Delegates to _lib modules.
Written from scratch, not copied from MVP.
"""

from ._lib.column import Column, ColumnType
from ._lib.packing import PackingPlan, ColumnLayout, compute_packing_plan
from ._lib.accessor import ColumnAccessor
from ._lib.table import Table
from ._lib.optimize import optimize as _optimize

# Builder aliases
make_column = Column
make_table = Table
compute_plan = compute_packing_plan
optimize_table = _optimize
column_layout = ColumnLayout
packing_plan = PackingPlan

def setup(kernel):
    """Register Storage types in kernel metadata."""
    kernel.metadata.setdefault("Storage", {})
    kernel.metadata["Storage"]["version"] = "1.0.0"
    kernel.metadata["Storage"]["types"] = ["Column", "ColumnType", "PackingPlan", "ColumnAccessor", "Table"]
