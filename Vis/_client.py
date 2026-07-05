"""Client-side charts.

ClientChart — полный датасет → JS, viewport локальный.
SliceChart — GPU viewport aggregation, только срез окна на клиенте.
"""

from __future__ import annotations

import time
from pathlib import Path

import anywidget
import numpy as np
import traitlets

from ._renderer import _parse_color

_ESM = Path(__file__).parent / "_client.js"


class ClientChart(anywidget.AnyWidget):
    """Chart: полный датасет → JS, viewport локальный.

    Использование:
        chart = ClientChart()
        chart.add_series(data, name="x", color="cyan")
        chart.show()
    """

    _esm = _ESM
    _window_data = traitlets.Unicode("").tag(sync=True)

    def __init__(self):
        super().__init__()
        self._series: list[dict] = []

    def add_series(self, data, name=None, color=None):
        if hasattr(data, "data") and callable(data.data):
            data = data.data()
        arr = np.asarray(data, dtype=np.float32).ravel()
        if arr.size == 0:
            raise ValueError("Empty series")
        rgba = _parse_color(color)
        if name is None:
            name = f"series_{len(self._series)}"
        self._series.append({"name": name, "color_rgba": rgba, "data": arr})
        self._send_all()
        return self

    def show(self):
        from IPython.display import display
        display(self)

    def _send_all(self):
        import base64, json

        if not self._series:
            return

        merged = np.concatenate([s["data"] for s in self._series]).tobytes()
        b64 = base64.b64encode(merged).decode("ascii")

        self._window_data = json.dumps({
            "type": "full",
            "series": [
                {"name": s["name"], "color": s["color_rgba"],
                 "length": s["data"].size}
                for s in self._series
            ],
            "data": b64,
        })


class SliceChart(anywidget.AnyWidget):
    """Chart: GPU viewport aggregation, на клиенте только срез окна.

    Данные хранятся на сервере в GPU-буферах.
    JS шлёт viewport-запрос (x_min, x_max, width_px).
    GPU compute downsamples до ~width_px точек.
    На клиент идёт только уже агрегированный viewport.

    Использование:
        chart = SliceChart()
        chart.add_series(data, name="x", color="cyan")
        chart.show()
    """

    _esm = _ESM
    _window_data = traitlets.Unicode("").tag(sync=True)

    def __init__(self, width=1000, height=260):
        super().__init__()
        self._series_info: list[dict] = []
        self._x_min = 0
        self._x_max = 0
        self._target = width
        self._agg = None
        self._gen_ms = 0.0

    # ── Lazy GPU engine ────────────────────────────────────────────────

    def _ensure_agg(self):
        if self._agg is None:
            from ._viewport_agg import ViewportAgg
            self._agg = ViewportAgg()

    # ── Public API ──────────────────────────────────────────────────────

    def add_series(self, data, name=None, color=None):
        rgba = _parse_color(color)

        # NumericSeries path (duck-type через _proxy)
        if hasattr(data, '_proxy'):
            proxy = data._proxy
            length = proxy._length
            if length == 0:
                raise ValueError("Empty series")
            self._ensure_agg()
            self._agg.attach_series_via_table(
                proxy._table_id, proxy._col_name,
                proxy._layout, length,
            )
            # Передаём gen_ms из серии (заполняется в nf.arange/sin/cos)
            if hasattr(data, '_gen_ms') and data._gen_ms > self._gen_ms:
                self._gen_ms = data._gen_ms
            if name is None:
                name = f"series_{len(self._series_info)}"
            self._series_info.append({
                "name": name, "color_rgba": rgba, "length": length,
            })
            self._x_min = 0
            self._x_max = min(1000, length)
            self._send_window()
            return self

        # Legacy .data() callable
        if hasattr(data, "data") and callable(data.data):
            data = data.data()

        arr = np.asarray(data, dtype=np.float32).ravel()
        if arr.size == 0:
            raise ValueError("Empty series")
        if name is None:
            name = f"series_{len(self._series_info)}"

        self._ensure_agg()
        self._agg.attach_series(arr)

        self._series_info.append({
            "name": name, "color_rgba": rgba, "length": arr.size,
        })
        self._x_min = 0
        self._x_max = min(1000, arr.size)
        self._send_window()
        return self

    def show(self):
        from IPython.display import display
        display(self)

    # ── Custom messages from JS ─────────────────────────────────────────

    def _handle_custom_msg(self, content, buffers=None):
        if isinstance(content, str):
            return
        typ = content.get("type")

        if typ == "init":
            if self._series_info:
                self._send_window()

        elif typ == "viewport":
            x_min = int(content.get("xMin", 0))
            x_max = int(content.get("xMax", 1000))
            cw = int(content.get("cw", 1000))
            self._target = max(10, cw)
            self._x_min = max(0, x_min)
            self._x_max = max(self._x_min + 1, x_max)
            self._send_window()

    # ── GPU viewport aggregation → send ────────────────────────────────

    def _send_window(self):
        import base64, json

        if not self._series_info:
            return

        total_len = max(s["length"] for s in self._series_info)
        x_min = max(0, min(self._x_min, total_len - 1))
        x_max = max(x_min + 1, min(self._x_max, total_len))
        if x_max <= x_min:
            return

        t0 = time.perf_counter()

        slices = self._agg.aggregate(x_min, x_max, self._target)

        y_min, y_max = float("inf"), float("-inf")
        for s in slices:
            if len(s) > 0:
                y_min = min(y_min, float(s.min()))
                y_max = max(y_max, float(s.max()))
        if y_min == float("inf"):
            y_min, y_max = 0.0, 1.0

        t_agg = time.perf_counter() - t0

        merged = np.concatenate(slices).tobytes()
        b64 = base64.b64encode(merged).decode("ascii")

        payload = json.dumps({
            "type": "slice",
            "xMin": x_min,
            "xMax": x_max,
            "yMin": y_min,
            "yMax": y_max,
            "totalLen": total_len,
            "series": [
                {"name": s["name"], "color": s["color_rgba"],
                 "length": len(sl)}
                for s, sl in zip(self._series_info, slices)
            ],
            "data": b64,
            "aggMs": round(t_agg * 1000, 1),
            "genMs": round(self._gen_ms, 1),
        })
        self._window_data = payload
