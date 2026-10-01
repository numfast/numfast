# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Table: ordered named columns (2D boundary over Series)."""

import numpy as np

from .series import Series


class Table:
    def __init__(self, kernel, columns: dict):
        if not columns:
            raise ValueError(
                "Table needs >=1 column. Fix: pass {name: Series}. "
                "See specs/01-public-api.md"
            )
        names = list(columns)
        n = len(columns[names[0]])
        for nm, col in columns.items():
            if not isinstance(col, Series):
                raise ValueError(
                    f"Table column '{nm}' is {type(col).__name__}, not Series. "
                    "Fix: wrap via nf.from_numpy first."
                )
            if len(col) != n:
                raise ValueError(
                    f"Table column '{nm}' length {len(col)} != {n}. "
                    "Fix: align column lengths."
                )
        self._kernel = kernel
        self._columns = dict(columns)

    @property
    def names(self):
        return list(self._columns)

    def __len__(self):
        return len(next(iter(self._columns.values())))

    @property
    def ncols(self):
        return len(self._columns)

    def __repr__(self):
        cols = ", ".join(f"{n}:{c.dtype}" for n, c in self._columns.items())
        return f"Table({len(self)}x{self.ncols} [{cols}])"

    def column(self, name):
        try:
            return self._columns[name]
        except KeyError:
            raise KeyError(
                f"Table has no column '{name}' (have {self.names}). "
                "Fix: use one of the existing names."
            ) from None

    def to_numpy(self):
        """Stack columns (C-contiguous 2D); mixed dtypes -> upcast via numpy."""
        mats = [c.to_numpy() for c in self._columns.values()]
        kinds = {m.dtype.kind for m in mats}
        if len(kinds) > 1:
            raise ValueError(
                f"Table.to_numpy needs uniform dtype kind, got {sorted(kinds)}. "
                "Fix: convert columns first or read per-column to_numpy()."
            )
        return np.ascontiguousarray(np.column_stack(mats))

    def to_pandas(self):
        """Thin wrapper over nf.to_pandas (no duplicate logic)."""
        import numfast as nf
        return nf.to_pandas(self)

    def to_arrow(self):
        """Thin wrapper over nf.to_arrow (no duplicate logic)."""
        import numfast as nf
        return nf.to_arrow(self)
