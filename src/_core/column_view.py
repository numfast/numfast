# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""Zero-copy column view into bit-packed container rows.

Reads values directly from container rows using mask+shift,
without extracting all values to a Python list.
"""

from _core.container import _u32_bits_to_f32, _unpack_compressed_float
from _core.container import _unpack_scaled, _unpack_enum
from _core.backend import get_xp


class ColumnView:
    """Zero-copy view into a single column of a bit-packed container.

    Provides index access to individual values via mask+shift directly
    on the container rows. No full-column extraction to Python list.

    Args:
        rows: container rows (list of list of u32 parts)
        col_name: column name
        layout: container layout dict
    """

    def __init__(self, rows: list, col_name: str, layout: dict):
        self._rows = rows
        self._col_name = col_name
        self._layout = layout
        self._cinfo = layout["columns"][col_name]
        self._dtype = self._cinfo["dtype"]
        self._meta = self._cinfo["meta"]
        self._num_rows = len(rows)

    def __len__(self) -> int:
        return self._num_rows

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._num_rows)
            return [self._get_value(i) for i in range(start, stop, step)]
        if idx < 0:
            idx += self._num_rows
        if idx < 0 or idx >= self._num_rows:
            raise IndexError(f"index {idx} out of range [0, {self._num_rows})")
        return self._get_value(idx)

    def __iter__(self):
        for i in range(self._num_rows):
            yield self._get_value(i)

    def __repr__(self) -> str:
        return self._format_preview()

    def __str__(self) -> str:
        return self._format_preview()

    def to_list(self) -> list:
        """Extract all column values to a Python list."""
        return [self._get_value(i) for i in range(self._num_rows)]

    def to_numpy(self) -> object:
        """Extract all column values to a numpy array."""
        xp = get_xp()
        return xp.array(self.to_list())

    def _get_value(self, idx: int):
        row = self._rows[idx]
        raw = 0
        bits_collected = 0
        for part_idx, part_offset, chunk_size in self._cinfo["parts"]:
            mask = (1 << chunk_size) - 1
            chunk = (row[part_idx] >> part_offset) & mask
            raw = (raw << chunk_size) | chunk
            bits_collected += chunk_size

        dtype = self._dtype
        meta = self._meta
        if dtype == "float32":
            if meta.get("_is_compressed_float"):
                return _unpack_compressed_float(raw, meta)
            return _u32_bits_to_f32(raw)
        if dtype == "scaled":
            return _unpack_scaled(raw, meta)
        if dtype == "enum":
            return _unpack_enum(raw, meta)
        if dtype == "int64":
            if raw >= (1 << 63):
                raw -= (1 << 64)
            return raw
        return float(raw)

    def _format_preview(self) -> str:
        n = self._num_rows
        if n == 0:
            return f"ColumnView({self._col_name!r}, dtype={self._dtype}, empty)"
        if n <= 6:
            vals = [self._get_value(i) for i in range(n)]
            return f"ColumnView({self._col_name!r}, dtype={self._dtype}, [{', '.join(self._fmt(v) for v in vals)}])"
        head = [self._get_value(i) for i in range(3)]
        tail = [self._get_value(i) for i in range(n - 3, n)]
        return (
            f"ColumnView({self._col_name!r}, dtype={self._dtype}, "
            f"[{', '.join(self._fmt(v) for v in head)}, ..., "
            f"{', '.join(self._fmt(v) for v in tail)}], n={n})"
        )

    @staticmethod
    def _fmt(val) -> str:
        if isinstance(val, float):
            return f"{val:.4g}"
        return str(val)
