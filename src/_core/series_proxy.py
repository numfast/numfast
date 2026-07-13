"""SeriesProxy — lightweight proxy for one column of a Table.

A SeriesProxy holds only (table, column_name).
All operations delegate to the table.
SeriesProxy does NOT store data.
"""


class SeriesProxy:
    """Proxy for a single column of a Table.
    
    Every Series in NumFast is a SeriesProxy.
    There are no 'standalone' series — only table columns.
    
    Args:
        table: Table instance containing the data
        column_name: name of the column this proxy points to
    
    Usage:
        proxy = SeriesProxy(table, "Close")
        val = proxy[0]          # table.get("Close", 0)
        proxy.optimize()         # table.optimize()
    """
    
    def __init__(self, table, column_name):
        self._table = table
        self._column = column_name
        self._name = column_name
    
    # ── Identity ────────────────────────────────────────
    
    @property
    def table(self):
        """The underlying Table."""
        return self._table
    
    @property
    def name(self):
        """Column name in the underlying Table."""
        return self._column
    
    @property
    def num_rows(self):
        """Number of rows in the underlying Table."""
        return self._table.num_rows
    
    # ── Data access — delegates to Table ─────────────────
    
    def get(self, i):
        """Read value at row i."""
        return self._table.get(self._column, i)
    
    def set(self, i, value):
        """Write value at row i."""
        self._table.set(self._column, i, value)
    
    def __getitem__(self, i):
        """Indexing: proxy[i] → table.get(column, i)"""
        return self.get(i)
    
    def __setitem__(self, i, value):
        self.set(i, value)
    
    def __len__(self):
        return self.num_rows
    
    # ── Operations — delegate to Table ───────────────────
    
    def optimize(self):
        """Optimize the underlying table's packing."""
        return self._table.optimize()
    
    def statistics(self):
        """Compute column statistics (min, max, mean, std)."""
        values = [self.get(i) for i in range(self.num_rows)]
        n = len(values)
        if n == 0:
            return {}
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        return {
            "min": min(values),
            "max": max(values),
            "mean": mean,
            "std": variance ** 0.5,
            "count": n,
        }
    
    # ── Display ─────────────────────────────────────────
    
    def __repr__(self):
        return f"SeriesProxy({self._column}, rows={self.num_rows})"
    
    def __str__(self):
        return f"<Series: {self._column} ({self.num_rows} rows)>"
