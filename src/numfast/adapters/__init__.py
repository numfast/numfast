# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Adapters: host <-> NumFast boundary (Schema layer in, engine buffers out).

Adapters NEVER execute queries: no groupby/join/reduce here, only dtype /
validity / NaN translation plus Series/Table construction. Layout inside
the wheel package keeps the wheel self-contained.
"""

# Note: .pandas/.arrow import cleanly without their optional deps;
# only the entry calls raise the controlled ImportError.
from .numpy import from_numpy, to_numpy
from .pandas import from_pandas, to_pandas
from .arrow import from_arrow, to_arrow

__all__ = ["from_numpy", "to_numpy", "from_pandas", "to_pandas",
           "from_arrow", "to_arrow"]
