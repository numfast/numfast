"""Column: logical description of a table column."""

from enum import Enum
from typing import Optional


class ColumnType(Enum):
    SCALED = "scaled"
    ENUM = "enum"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"


class Column:
    """Logical column description.
    
    Args:
        name: column name
        dtype: ColumnType
        scale: multiplier for scaled encoding
        offset: addend for scaled encoding
        nullable: whether column can contain None
        categories: for ENUM type, list of string values
    
    Note:
        _bit_width is computed in __init__ and is internal.
    """
    
    def __init__(
        self,
        name: str,
        dtype: ColumnType,
        scale: float = 1.0,
        offset: float = 0.0,
        nullable: bool = False,
        categories: Optional[list] = None,
    ):
        self.name = name
        self.dtype = dtype
        self.scale = scale
        self.offset = offset
        self.nullable = nullable
        self.categories = categories
        self._compute_bit_width()
    
    def _compute_bit_width(self):
        if self.dtype == ColumnType.ENUM and self.categories is not None:
            n = len(self.categories)
            bits = 1
            while (1 << bits) < n + (1 if self.nullable else 0):
                bits += 1
            self._bit_width = bits
        elif self.dtype == ColumnType.SCALED:
            self._bit_width = 16
        elif self.dtype == ColumnType.INT32:
            self._bit_width = 32
        elif self.dtype == ColumnType.INT16:
            self._bit_width = 16
        elif self.dtype == ColumnType.INT8:
            self._bit_width = 8
        else:
            self._bit_width = 16
    
    def __repr__(self):
        return f"Column({self.name}, {self.dtype.value}, bits={self._bit_width})"
