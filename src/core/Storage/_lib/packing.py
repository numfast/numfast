"""PackingPlan: bit-layout description for a table row."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ColumnLayout:
    """Physical layout of one column within a packed row."""
    name: str
    bit_offset: int
    bit_width: int
    scale: float = 1.0
    offset: float = 0.0
    signed: bool = False
    categories: Optional[list] = None
    parts: tuple = ()


@dataclass(frozen=True)
class PackingPlan:
    """Immutable plan describing bit layout of all columns."""
    columns: tuple
    num_parts: int
    bytes_per_row: int
    
    def get_layout(self, name: str) -> Optional[ColumnLayout]:
        for col in self.columns:
            if col.name == name:
                return col
        return None
    
    def __repr__(self):
        parts = []
        for c in self.columns:
            parts.append(f"  {c.name}: offset={c.bit_offset} bits={c.bit_width} "
                         f"parts={len(c.parts)} scale={c.scale} offset={c.offset}")
        return f"PackingPlan({self.num_parts} parts, {self.bytes_per_row}B/row):\n" + "\n".join(parts)


def compute_packing_plan(columns):
    """Compute bit layout with cross-boundary split.
    
    Args:
        columns: list of Column objects
    
    Returns:
        PackingPlan describing the bit layout
    """
    PART_BITS = 32
    layouts = []
    next_bit = 0
    
    for col in columns:
        bw = col._bit_width
        remaining = bw
        parts = []
        bit_start = next_bit
        
        while remaining > 0:
            part_idx = next_bit // PART_BITS
            part_offset = next_bit % PART_BITS
            available = PART_BITS - part_offset
            chunk = min(remaining, available)
            parts.append((part_idx, part_offset, chunk))
            next_bit += chunk
            remaining -= chunk
        
        layouts.append(ColumnLayout(
            name=col.name,
            bit_offset=bit_start,
            bit_width=bw,
            scale=col.scale,
            offset=col.offset,
            signed=(col.dtype.name in ("SCALED", "INT8", "INT16")),
            categories=col.categories if col.dtype.name == "ENUM" else None,
            parts=tuple(parts),
        ))
    
    num_parts = (next_bit + PART_BITS - 1) // PART_BITS
    bytes_per_row = num_parts * 4
    return PackingPlan(columns=tuple(layouts), num_parts=num_parts, bytes_per_row=bytes_per_row)
