"""Optimize: create a new Table with reduced bitness per column."""

from .column import Column, ColumnType


def optimize(table):
    """Rebuild table with minimal bitness for scaled columns.
    
    Reads all data, computes per-column range,
    creates new Column definitions with optimized scale/offset,
    and builds a new Table.
    
    Only affects SCALED columns. INT8/16/32 and ENUM are passed through.
    
    Args:
        table: Table instance to optimize
    
    Returns:
        New Table with potentially narrower packing
    """
    data = {}
    stats = {}
    
    # Read all data and compute stats
    for col in table.column_list:
        values = [table.get(col.name, i) for i in range(table.num_rows)]
        data[col.name] = values
        stats[col.name] = {'min': min(values), 'max': max(values)}
    
    # Build new column definitions
    new_columns = []
    for col in table.column_list:
        s = stats[col.name]
        
        if col.dtype == ColumnType.SCALED:
            val_range = s['max'] - s['min']
            if val_range == 0:
                val_range = 1
            
            # Try bit widths from 16 down to 8 (higher = more precision)
            chosen = None
            for bits in [16, 12, 8]:
                max_int = (1 << (bits - 1)) - 1
                if val_range <= max_int:
                    new_scale = val_range / max_int
                    chosen = Column(
                        name=col.name,
                        dtype=ColumnType.SCALED,
                        scale=new_scale,
                        offset=float(s['min']),
                    )
                    break
            
            if chosen is None:
                new_columns.append(col)
            else:
                new_columns.append(chosen)
        else:
            # Pass through non-scaled columns unchanged
            new_columns.append(col)
    
    # Build new table (Create → Swap → Free)
    from .table import Table
    return Table(new_columns, data)
