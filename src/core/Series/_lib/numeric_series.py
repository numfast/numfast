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
        if isinstance(key, (int, slice)):
            return self.data()[key]
        return self._proxy[key]

    def __contains__(self, value):
        if isinstance(value, (int, float)):
            try:
                import numpy as np
                return bool(np.any(np.isclose(self.data(), value)))
            except Exception:
                return value in self.data()
        return value in self._proxy

    def __iter__(self):
        return iter(self.data())

    def __array__(self, dtype=None):
        import numpy as np
        arr = np.array(self.data(), dtype=np.float64)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def __len__(self) -> int:
        return self._proxy["_length"]

    @property
    def len(self) -> int:
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

    def __rtruediv__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        from .expr import _LazyExpr
        return _LazyExpr('truediv', other, self)

    def __mod__(self, other):
        from .expr import _LazyExpr
        if isinstance(other, _LazyExpr):
            return _LazyExpr('mod', self, other)
        if isinstance(other, (int, float)):
            return _LazyExpr('mod', self, other)
        if not isinstance(other, NumericSeries):
            return NotImplemented
        return _LazyExpr('mod', self, other)

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        from .expr import _LazyExpr
        return _LazyExpr('pow', self, other)

    def __neg__(self):
        from .expr import _LazyExpr
        return _LazyExpr('neg', self)

    # ── Comparisons -> 0/1 mask expressions (Compare primitive, L1) ───

    def _cmp(self, other, cmp_op):
        """Build a lazy comparison node -> mask of 0.0/1.0 f32 on compute()."""
        from .expr import _LazyExpr
        if isinstance(other, (int, float, _LazyExpr, NumericSeries)):
            return _LazyExpr('cmp', self, other, cmp_op=cmp_op)
        return NotImplemented

    def __lt__(self, other):
        return self._cmp(other, 'lt')

    def __le__(self, other):
        return self._cmp(other, 'le')

    def __gt__(self, other):
        return self._cmp(other, 'gt')

    def __ge__(self, other):
        return self._cmp(other, 'ge')

    def __eq__(self, other):
        return self._cmp(other, 'eq')

    def __ne__(self, other):
        return self._cmp(other, 'ne')

    # Identity hash preserved despite __eq__ (series stay dict/set keys).
    __hash__ = object.__hash__

    def filter(self, mask):
        """Gather-composition filter: values where mask is nonzero.

        Uses the core Filter extension (Compare+Scan+Gather semantics),
        stable order. Materializes both series at the boundary.

        Args:
            mask: NumericSeries of 0/1 markers, same length as self.

        Returns:
            NumericSeries with selected values.
        """
        from Filter._lib.filter import Filter as _filter_impl
        if not isinstance(mask, NumericSeries):
            raise TypeError("filter mask must be a NumericSeries")
        if len(mask) != len(self):
            raise ValueError("filter mask length mismatch")
        selected = _filter_impl(self.data(), mask.data())
        ctx = {"_id": self._proxy["_context_id"]}
        return NumericSeries(list(selected), ctx)

    def data(self):
        """MATERIALIZATION BOUNDARY: extract column into a Python list.

        Copies packed storage to the host. Do not use inside lazy/GPU
        pipelines; prefer expression ops and ``.compute()``.
        """
        return _series_data_proxy(self._proxy)

    def info(self):
        self._proxy.info()

    def release(self):
        self._proxy.release()

    def _validate(self):
        self._proxy._validate()

    @classmethod
    def from_packed(cls, rows, layout, ctx, col="_value"):
        """Gap-фикс S137: создание Series без pack_rows копии.

        Берёт уже packed rows/layout, регистрирует via _kernel._register_table,
        возвращает NumericSeries view (аналогично __init__ но без pack).
        """
        from _core import kernel as _kernel

        # Schema из layout.columns
        if isinstance(layout, dict) and "columns" in layout:
            schema = []
            for name, cinfo in layout["columns"].items():
                entry = {"name": name, "dtype": cinfo.get("dtype", "float32")}
                if "compression" in cinfo:
                    entry["compression"] = cinfo["compression"]
                schema.append(entry)
            # cinfo для выбранной колонки
            if col in layout["columns"]:
                cinfo = layout["columns"][col]
            else:
                # fallback: первая колонка или дефолт
                cinfo = next(iter(layout["columns"].values())) if layout["columns"] else {"name": col, "dtype": "float32"}
                # если col != найдено но ожидается _value — создаём дефолт
                if col not in layout["columns"]:
                    cinfo = {"name": col, "dtype": cinfo.get("dtype", "float32")}
            num_rows = layout.get("num_rows")
            if num_rows is None:
                # пробуем взять из cinfo
                num_rows = cinfo.get("length") or cinfo.get("num_rows") or cinfo.get("count")
            if num_rows is None:
                try:
                    num_rows = len(rows) if hasattr(rows, "__len__") else 0
                    # если rows — packed bytes/list of parts, длина != num_rows, пробуем layout
                    if isinstance(rows, (bytes, bytearray)):
                        num_rows = layout.get("num_rows", 0)
                except Exception:
                    num_rows = 0
            num_parts = layout.get("num_parts", 1)
            # rows может быть list частей
            if isinstance(rows, list) and layout.get("num_parts") is None:
                # эвристика: если rows — список частей
                if rows and isinstance(rows[0], (bytes, bytearray)):
                    num_parts = len(rows)
        else:
            # layout не в ожидаемом формате — строим минимальный
            schema = [{"name": col, "dtype": "float32"}]
            cinfo = {"name": col, "dtype": "float32"}
            try:
                num_rows = len(rows) if hasattr(rows, "__len__") else 0
            except Exception:
                num_rows = 0
            num_parts = 1
            layout = {"columns": {col: cinfo}, "num_rows": num_rows, "num_parts": num_parts}

        # Нормализуем num_rows к int
        try:
            num_rows = int(num_rows)
        except Exception:
            num_rows = 0

        tid, _ = _kernel._register_table(rows, num_parts, schema, layout, created_by="numeric_series.from_packed")

        proxy = _kernel._ProxyDict(
            _series_id=tid,
            _context_id=ctx["_id"],
            _length=num_rows,
            _table_id=tid,
            _col_name=col if col in layout.get("columns", {}) else (next(iter(layout["columns"].keys())) if layout.get("columns") else col),
            _col_schema=cinfo,
            _layout=layout,
            _is_hidden_series=True,
        )
        obj = cls.__new__(cls)
        obj._proxy = proxy
        obj._shape = (num_rows,)
        return obj

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


__all__ = ["NumericSeries", "make_series", "series_add", "series_sub"]
