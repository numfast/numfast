# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Sequence

from _core.context import check_context_compatible
from _core.series_types import SeriesKind
from .base_series import BaseSeries

_DEFAULT_COL = "_value"


class NumericSeries(BaseSeries):
    kind = SeriesKind.NUMERIC

    def __init__(self, data: Sequence, ctx: dict,
                 compression: dict | None = None):
        from _core.container import pack_rows, compute_layout

        if compression:
            col_def = {"name": _DEFAULT_COL, "dtype": "float32",
                       "compression": {"scaled": True, "bits": compression.get("bits", 16),
                                       "scale": compression.get("scale", 1.0),
                                       "offset": compression.get("offset", 0.0)}}
        else:
            col_def = {"name": _DEFAULT_COL, "dtype": "float32"}

        schema = [col_def]
        packed = pack_rows(schema, {_DEFAULT_COL: data})
        cinfo = packed["layout"]["columns"][_DEFAULT_COL]

        from _core import kernel as _kernel
        tid, _ = _kernel._register_table(packed["rows"], packed["num_parts"],
                                         packed["schema"], packed["layout"],
                                         created_by="numeric_series")

        proxy = _kernel._ProxyDict(
            _series_id=tid,
            _context_id=ctx["_id"],
            _length=packed["num_rows"],
            _table_id=tid,
            _col_name=_DEFAULT_COL,
            _col_schema=cinfo,
            _layout=packed["layout"],
            _is_hidden_series=True,
        )
        self._proxy = proxy
        self._shape = (packed["num_rows"],)

    def __getitem__(self, key):
        return self._proxy[key]

    def __contains__(self, key):
        return key in self._proxy

    def __len__(self) -> int:
        return self._proxy["_length"]

    def __add__(self, other):
        from .expr import _LazyExpr
        if isinstance(other, _LazyExpr):
            return _LazyExpr('add', self, other)
        if isinstance(other, (int, float)):
            return _LazyExpr('add', self, other)
        if not isinstance(other, NumericSeries):
            return NotImplemented
        return _LazyExpr('add', self, other)

    def __radd__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        from .expr import _LazyExpr
        return _LazyExpr('add', other, self)

    def __sub__(self, other):
        from .expr import _LazyExpr
        if isinstance(other, _LazyExpr):
            return _LazyExpr('sub', self, other)
        if isinstance(other, (int, float)):
            return _LazyExpr('sub', self, other)
        if not isinstance(other, NumericSeries):
            return NotImplemented
        return _LazyExpr('sub', self, other)

    def __rsub__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        from .expr import _LazyExpr
        return _LazyExpr('sub', other, self)

    def __mul__(self, other):
        from .expr import _LazyExpr
        if isinstance(other, _LazyExpr):
            return _LazyExpr('mul', self, other)
        if isinstance(other, (int, float)):
            return _LazyExpr('mul', self, other)
        if not isinstance(other, NumericSeries):
            return NotImplemented
        return _LazyExpr('mul', self, other)

    def __rmul__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        from .expr import _LazyExpr
        return _LazyExpr('mul', other, self)

    def __truediv__(self, other):
        from .expr import _LazyExpr
        if isinstance(other, _LazyExpr):
            return _LazyExpr('truediv', self, other)
        if isinstance(other, (int, float)):
            return _LazyExpr('truediv', self, other)
        if not isinstance(other, NumericSeries):
            return NotImplemented
        return _LazyExpr('truediv', self, other)

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        from .expr import _LazyExpr
        return _LazyExpr('pow', self, other)

    def __neg__(self):
        from .expr import _LazyExpr
        return _LazyExpr('neg', self)

    def data(self):
        return _series_data_proxy(self._proxy)

    def info(self):
        self._proxy.info()

    def release(self):
        self._proxy.release()

    def _validate(self):
        self._proxy._validate()

    def is_alive(self) -> bool:
        return self._proxy.is_alive()


def _series_data_proxy(s: dict) -> list | None:
    from _core.container import extract_column
    from _core import kernel as _kernel
    entry = _kernel._get_table_entry(s["_table_id"])
    if entry is None:
        return None
    return extract_column(entry["rows"], s["_col_name"], s["_layout"])


def _series_len(s: dict) -> int:
    return s["_length"]


def make_series(data: Sequence, ctx: dict,
                compression: dict | None = None) -> NumericSeries:
    return NumericSeries(data, ctx, compression)


def series_add(a: dict, b: dict) -> NumericSeries:
    check_context_compatible(a, b)
    ad = _series_data_proxy(a)
    bd = _series_data_proxy(b)
    if ad is None or bd is None:
        raise ValueError("Series data not found in Kernel")
    n = min(len(ad), len(bd))
    result = [ad[i] + bd[i] for i in range(n)]
    ctx = {"_id": a["_context_id"]}
    return NumericSeries(result, ctx)


def series_sub(a: dict, b: dict) -> NumericSeries:
    check_context_compatible(a, b)
    ad = _series_data_proxy(a)
    bd = _series_data_proxy(b)
    if ad is None or bd is None:
        raise ValueError("Series data not found in Kernel")
    n = min(len(ad), len(bd))
    result = [ad[i] - bd[i] for i in range(n)]
    ctx = {"_id": a["_context_id"]}
    return NumericSeries(result, ctx)
