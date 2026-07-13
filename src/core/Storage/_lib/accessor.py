"""ColumnAccessor: read/write a single column from a packed u32 row.

Knows nothing about ColumnType. Works only from ColumnLayout parameters.
"""

import numpy as np


class ColumnAccessor:
    """Reads and writes one column within a packed row (u32 array).
    
    Uses multi-part assembly for cross-boundary fields.
    Sign extension for signed layouts.
    Category lookup for enum layouts.
    Scale/offset for physical→logical conversion.
    """
    
    def __init__(self, layout):
        self.layout = layout
        self.parts = layout.parts
        self.bit_width = layout.bit_width
        self.signed = layout.signed
        self.scale = layout.scale
        self.offset = layout.offset
        self.categories = layout.categories
    
    def get(self, row: np.ndarray):
        """Extract logical value from a packed row (u32 slice)."""
        # Multi-part assembly: read chunks from each part, shift into place
        extracted = 0
        shift = 0
        for part_idx, part_offset, part_bits in self.parts:
            part = int(row[part_idx])
            chunk = (part >> part_offset) & ((1 << part_bits) - 1)
            extracted |= (chunk << shift)
            shift += part_bits
        
        # Sign extension for signed types
        if self.signed and (extracted >> (self.bit_width - 1)):
            extracted -= (1 << self.bit_width)
        
        # Enum lookup
        if self.categories is not None:
            if 0 <= extracted < len(self.categories):
                return self.categories[extracted]
            return f"<unknown:{extracted}>"
        
        # Scale/offset for numeric values
        return float(extracted) * self.scale + self.offset
    
    def set(self, row: np.ndarray, value):
        """Encode logical value into packed row (u32 slice)."""
        # Convert logical value to raw integer
        if self.categories is not None:
            if value in self.categories:
                raw = self.categories.index(value)
            else:
                raise ValueError(f"Unknown enum value '{value}'")
        elif self.scale != 0:
            raw = int(round((value - self.offset) / self.scale))
        else:
            raw = int(value)
        
        # Range check with two's complement for signed
        max_val = (1 << self.bit_width) - 1
        if self.signed and raw < 0:
            raw = (1 << self.bit_width) + raw  # two's complement
        if raw < 0 or raw > max_val:
            raise ValueError(
                f"Value {value} (raw={raw}) exceeds bit_width {self.bit_width} "
                f"for signed={self.signed}"
            )
        
        # Multi-part write: write each chunk to its part
        shift = 0
        for part_idx, part_offset, part_bits in self.parts:
            chunk = (raw >> shift) & ((1 << part_bits) - 1)
            mask = np.uint32(((1 << part_bits) - 1) << part_offset)
            row[part_idx] = (row[part_idx] & ~mask) | np.uint32(chunk << part_offset)
            shift += part_bits
