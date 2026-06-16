# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Sequence

from _core.context import check_context_compatible
from _core.series_types import SeriesKind
from _core.series.base_series import BaseSeries

_DEFAULT_COL = "_value"


class NumericSeries(BaseSeries):
    """Числовая серия с битовой упаковкой, GPU-агрегацией и статистикой.

    Основной рабочий тип NumFast. Использует Adaptive Bit Packing,
    Columnar Container и JIT WGSL compute shaders.

    Пример:
        s = NumericSeries([1.0, 2.0, 3.0], ctx)
        s.info()
        s.kind  # SeriesKind.NUMERIC
    """

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

    # ── Dict-like proxy delegation ────────────────────────────────────

    def __getitem__(self, key):
        return self._proxy[key]

    def __contains__(self, key):
        return key in self._proxy

    def __len__(self) -> int:
        return self._proxy["_length"]

    def __add__(self, other):
        if not isinstance(other, (NumericSeries, dict)):
            return NotImplemented
        a_data = _series_data_proxy(self._proxy)
        b_data = _series_data_proxy(other if isinstance(other, dict)
                                    else other._proxy)
        if a_data is None or b_data is None:
            raise ValueError("Series data not found in Kernel")
        n = min(len(a_data), len(b_data))
        result = [a_data[i] + b_data[i] for i in range(n)]
        ctx = {"_id": self._proxy["_context_id"]}
        return NumericSeries(result, ctx)

    def __sub__(self, other):
        if not isinstance(other, (NumericSeries, dict)):
            return NotImplemented
        a_data = _series_data_proxy(self._proxy)
        b_data = _series_data_proxy(other if isinstance(other, dict)
                                    else other._proxy)
        if a_data is None or b_data is None:
            raise ValueError("Series data not found in Kernel")
        n = min(len(a_data), len(b_data))
        result = [a_data[i] - b_data[i] for i in range(n)]
        ctx = {"_id": self._proxy["_context_id"]}
        return NumericSeries(result, ctx)

    def data(self):
        """Извлечь данные из Kernel (CPU, через numpy)."""
        return _series_data_proxy(self._proxy)

    def info(self):
        """Показать диагностический дашборд."""
        self._proxy.info()

    def release(self):
        """Освободить ресурсы в Kernel."""
        self._proxy.release()

    def _validate(self):
        self._proxy._validate()

    def is_alive(self) -> bool:
        return self._proxy.is_alive()


# ── Внутренние функции (используются ядром и тестами) ─────────────────


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
    """Создать NumericSeries.

    Args:
        data: список float значений
        ctx: контекст
        compression: параметры сжатия (scale, offset, bits)

    Returns:
        NumericSeries
    """
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
