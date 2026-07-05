"""NumFast kernel — internal singleton context.

Users should use the top-level API (nf.Table, nf.create_table).
Context is only needed for advanced multi-context scenarios.
"""

from .Storage import (
    Table as _Table,
    Column as _Column,
    ColumnType as _ColumnType,
    PackingPlan as _PackingPlan,
    ColumnLayout as _ColumnLayout,
    ColumnAccessor as _ColumnAccessor,
    compute_packing_plan as _compute_packing_plan,
)


class Context:
    """NumFast execution context.

    Manages storage, memory, and compute resources.
    In typical usage this is hidden behind the top-level API.

    Advanced usage — creating independent table stores:

        ctx = nf.Context()
        table = ctx.create_table(schema=[...], data={...})
    """

    def __init__(self):
        pass

    def create_table(self, columns=None, data=None, **kwargs):
        """Create a new packed table.

        Args:
            columns: list of Column definitions
            data: dict of column_name → list of values
            **kwargs: additional options (reserved for future use)

        Returns:
            Table instance
        """
        return _Table(columns, data)

    @property
    def Table(self):
        """Access the Table class directly."""
        return _Table

    @property
    def Column(self):
        """Access the Column class directly."""
        return _Column

    @property
    def ColumnType(self):
        """Access the ColumnType enum directly."""
        return _ColumnType

    @property
    def PackingPlan(self):
        return _PackingPlan

    @property
    def ColumnLayout(self):
        return _ColumnLayout


# Singleton kernel — powers the top-level API
_kernel = Context()


def create_table(columns=None, data=None, **kwargs):
    """Create a packed table. Convenience wrapper around Context.create_table.

    Usage:
        table = nf.create_table(schema=[...], data={...})
    """
    return _kernel.create_table(columns=columns, data=data, **kwargs)
