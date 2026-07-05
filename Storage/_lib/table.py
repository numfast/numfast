"""Table: row-struct container with packed u32 buffer."""

import numpy as np
from .accessor import ColumnAccessor
from .packing import compute_packing_plan


class Table:
    """A table stored as packed rows of u32 values.
    
    Each row is a fixed-size array of u32 parts.
    The PackingPlan describes how columns map to bits within parts.
    ColumnAccessor handles individual column read/write within a row.
    """
    
    _next_id = 1
    
    def __init__(self, columns, data=None):
        self._table_id = f"tbl_{Table._next_id}"
        Table._next_id += 1
        
        self.columns = {col.name: col for col in columns}
        self.column_list = columns
        self.packing_plan = compute_packing_plan(columns)
        
        # Build accessors from packing plan
        self.accessors = {}
        for col in columns:
            layout = self.packing_plan.get_layout(col.name)
            self.accessors[col.name] = ColumnAccessor(layout)
        
        # Determine row count from data
        num_rows = 0
        if data:
            for col_name in data:
                num_rows = max(num_rows, len(data[col_name]))
        
        self.num_rows = num_rows
        self.num_parts = self.packing_plan.num_parts
        self.buffer = np.zeros(num_rows * self.num_parts, dtype=np.uint32)
        
        if data:
            self._load(data)
    
    def _load(self, data):
        """Populate buffer from Python dict of lists."""
        for col_name, values in data.items():
            if col_name not in self.accessors:
                continue
            accessor = self.accessors[col_name]
            for i, val in enumerate(values):
                row = self._get_row(i)
                accessor.set(row, val)
    
    def _get_row(self, i):
        """Return writable view of row i as a u32 slice."""
        start = i * self.num_parts
        return self.buffer[start:start + self.num_parts]
    
    def get(self, col_name, i):
        """Read value at column col_name, row i."""
        if col_name not in self.accessors:
            raise KeyError(f"Column '{col_name}' not found")
        row = self._get_row(i)
        return self.accessors[col_name].get(row)
    
    def set(self, col_name, i, value):
        """Write value at column col_name, row i."""
        if col_name not in self.accessors:
            raise KeyError(f"Column '{col_name}' not found")
        row = self._get_row(i)
        self.accessors[col_name].set(row, value)
    
    def info(self):
        """Print table layout info."""
        print(f"Table {self._table_id}: {self.num_rows} rows, "
              f"{self.packing_plan.num_parts} parts/row, "
              f"{self.packing_plan.bytes_per_row} B/row")
        print(self.packing_plan)
    
    def series(self, column_name=None):
        """Return a SeriesProxy for a column of this table.
        
        Args:
            column_name: column name (defaults to first column)
        
        Returns:
            SeriesProxy instance
        """
        from numfast._core.series_proxy import SeriesProxy
        if column_name is None:
            column_name = self.column_list[0].name
        return SeriesProxy(self, column_name)

    def optimize(self):
        """Create a new Table with minimal bitness for each column.
        
        Create → Swap → Free pattern.
        Returns a new Table; caller should drop reference to old one.
        """
        from .optimize import optimize
        return optimize(self)
