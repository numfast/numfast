import time
import struct

import numpy as np
import wgpu
from rendercanvas.jupyter import JupyterRenderCanvas, RenderCanvas

from ._shaders import VIEWPORT_VERTEX, VIEWPORT_FRAGMENT

DARK_BG = (28 / 255, 30 / 255, 34 / 255, 1.0)

# ---------------------------------------------------------------------------
# Monkey-patch: _rc_request_paint -> _rc_force_paint
# ---------------------------------------------------------------------------
_orig_request_paint = JupyterRenderCanvas._rc_request_paint


def _patched_request_paint(self):
    self._rc_force_paint()


JupyterRenderCanvas._rc_request_paint = _patched_request_paint

# ---------------------------------------------------------------------------
# Monkey-patch: force PNG instead of JPEG (54x smaller for chart images)
# ---------------------------------------------------------------------------
from jupyter_rfb._utils import array2compressed as _orig_array2compressed
from jupyter_rfb._png import array2png as _array2png


def _patched_array2compressed(array, quality=90):
    # Drop alpha for PNG
    if len(array.shape) == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    return "image/png", _array2png(array)


import jupyter_rfb._utils as _rfb_utils

_rfb_utils.array2compressed = _patched_array2compressed

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

_NAMED_COLORS = {
    "red": (1, 0, 0, 1),
    "green": (0, 1, 0, 1),
    "blue": (0, 0, 1, 1),
    "cyan": (0, 1, 1, 1),
    "magenta": (1, 0, 1, 1),
    "yellow": (1, 1, 0, 1),
    "white": (1, 1, 1, 1),
    "black": (0, 0, 0, 1),
    "orange": (1, 0.5, 0, 1),
    "purple": (0.5, 0, 0.5, 1),
    "pink": (1, 0.75, 0.8, 1),
    "gray": (0.5, 0.5, 0.5, 1),
    "grey": (0.5, 0.5, 0.5, 1),
}


def _parse_color(color):
    if color is None:
        return (0.4, 0.7, 1.0, 1.0)
    if isinstance(color, str):
        c = color.strip().lower()
        if c.startswith("#"):
            h = c.lstrip("#")
            r = int(h[0:2], 16) / 255
            g = int(h[2:4], 16) / 255
            b = int(h[4:6], 16) / 255
            a = int(h[6:8], 16) / 255 if len(h) >= 8 else 1.0
            return (r, g, b, a)
        if c in _NAMED_COLORS:
            return _NAMED_COLORS[c]
        raise ValueError(f"Unknown color: {color}")
    if isinstance(color, (tuple, list)):
        r, g, b, *a = color
        return (float(r), float(g), float(b), float(a[0] if a else 1.0))
    raise ValueError(f"Invalid color: {color}")


# ---------------------------------------------------------------------------
# Chart — Multi-series Viewport Engine
# ---------------------------------------------------------------------------

class Chart:
    REVISION = "2025-06-17/multi"

    def __init__(self, width=1000, height=600):
        self.width = width
        self.height = height
        self._series = []
        self._x_min = 0
        self._x_max = 0
        self._y_min = 0.0
        self._y_max = 1.0

        # Canvas / GPU — max_fps=60 for smoother scheduling
        self.canvas = RenderCanvas(size=(width, height), max_fps=60)
        self.context = self.canvas.get_wgpu_context()
        self.adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance"
        )
        self.device = self.adapter.request_device_sync()
        fmt = self.context.get_preferred_format(self.adapter)
        self.context.configure(device=self.device, format=fmt)
        self._fmt = fmt

        # Shared resources (created lazily)
        self._vp_buffer = None
        self._pipeline = None
        self._events_setup = False

        # Set draw function so force_draw() works
        self.canvas.request_draw(self.draw_frame)

        # ipywidgets (created lazily)
        self._scroll = None
        self._zoom = None
        self._status = None

        # Drag state
        self._drag_start = None

        # FPS measurement
        self._frame_count = 0
        self._fps_t0 = 0.0
        self._hover_text = ""

        print(f"[Chart] rev={self.REVISION}  ctx={type(self.canvas).__base__.__name__}")

    # ── Public API ──────────────────────────────────────────────────────

    def add_series(self, data, name=None, color=None):
        if hasattr(data, "data") and callable(data.data):
            data = data.data()
        arr = np.asarray(data, dtype=np.float32).ravel()
        if arr.size == 0:
            raise ValueError("Empty series")

        rgba = _parse_color(color)
        if name is None:
            name = f"series_{len(self._series)}"

        # Storage buffer
        data_buffer = self.device.create_buffer(
            label=f"data_{name}",
            size=arr.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.device.queue.write_buffer(data_buffer, 0, arr.tobytes())

        # Color uniform buffer
        color_bytes = struct.pack("ffff", *rgba)
        color_buffer = self.device.create_buffer(
            label=f"color_{name}",
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.device.queue.write_buffer(color_buffer, 0, color_bytes)

        entry = {
            "name": name,
            "data_np": arr,
            "color_rgba": rgba,
            "buffer": data_buffer,
            "color_buffer": color_buffer,
            "length": arr.size,
            "bind_group": None,
        }

        # Create pipeline + shared vp_buffer on first series
        if self._pipeline is None:
            self._vp_buffer = self.device.create_buffer(
                label="viewport_uniform",
                size=16,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            self._rebuild_pipeline()

        entry["bind_group"] = self._make_bind_group(entry)

        if not self._series:
            self._y_min = float(arr.min())
            self._y_max = float(arr.max())
        else:
            self._y_min = min(self._y_min, float(arr.min()))
            self._y_max = max(self._y_max, float(arr.max()))

        self._series.append(entry)

        if len(self._series) == 1:
            vp_size = min(1000, arr.size)
            self._x_min = 0
            self._x_max = vp_size

        self._write_vp_uniform()
        self._sync_widgets()
        self.canvas.request_draw(self.draw_frame)
        return self

    def set_series(self, data):
        return self.add_series(data, name="main", color="cyan")

    def set_viewport(self, x_min, x_max):
        total_len = max((s["length"] for s in self._series), default=0)
        x_min = max(0, int(x_min))
        x_max = min(total_len, int(x_max))
        if x_max <= x_min:
            x_max = min(x_min + 1, total_len)
        self._x_min = x_min
        self._x_max = x_max

        self._write_vp_uniform()
        self._sync_widgets()
        # Synchronous draw — bypasses scheduler, renders + sends immediately
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.force_draw()

    def set_yrange(self, y_min, y_max):
        self._y_min = float(y_min)
        self._y_max = float(y_max)
        self._write_vp_uniform()
        self.canvas.force_draw()

    def show(self):
        from IPython.display import display
        display(self._make_vbox())
        if self._series:
            self.canvas.request_draw(self.draw_frame)

    # ── Series info ─────────────────────────────────────────────────────

    @property
    def series_count(self):
        return len(self._series)

    def series_names(self):
        return [s["name"] for s in self._series]

    # ── ipywidgets ──────────────────────────────────────────────────────

    def _ensure_widgets(self):
        if self._scroll is not None:
            return
        import ipywidgets as ipw

        total_len = max((s["length"] for s in self._series), default=1)
        vp_size = max(1, self._x_max - self._x_min)

        self._scroll = ipw.IntSlider(
            value=self._x_min,
            min=0,
            max=max(0, total_len - vp_size),
            step=1,
            description="Scroll",
            continuous_update=True,
        )
        self._zoom = ipw.IntSlider(
            value=vp_size,
            min=10,
            max=max(10, total_len),
            step=1,
            description="Zoom",
            continuous_update=True,
        )
        self._status = ipw.HTML(value="", placeholder="\u200b")

        self._scroll.observe(self._on_scroll, names="value")
        self._zoom.observe(self._on_zoom, names="value")

    def _make_vbox(self):
        self._ensure_widgets()
        self._ensure_events()
        import ipywidgets as ipw
        return ipw.VBox([self.canvas, self._scroll, self._zoom, self._status])

    def _on_scroll(self, change):
        self.set_viewport(change["new"], change["new"] + self._zoom.value)

    def _on_zoom(self, change):
        self.set_viewport(self._x_min, self._x_min + change["new"])

    def _sync_widgets(self):
        if self._scroll is None:
            return
        total_len = max((s["length"] for s in self._series), default=1)
        vp_size = self._x_max - self._x_min
        with self._scroll.hold_trait_notifications():
            self._scroll.max = max(0, total_len - vp_size)
            self._scroll.value = self._x_min
        with self._zoom.hold_trait_notifications():
            self._zoom.max = max(10, total_len)
            self._zoom.value = vp_size

    # ── Mouse / event handlers ─────────────────────────────────────────

    def _ensure_events(self):
        if self._events_setup:
            return
        self.canvas.add_event_handler(self._on_wheel, "wheel")
        self.canvas.add_event_handler(self._on_pointer_move, "pointer_move")
        self.canvas.add_event_handler(self._on_pointer_down, "pointer_down")
        self.canvas.add_event_handler(self._on_pointer_up, "pointer_up")
        self.canvas.add_event_handler(self._on_double_click, "double_click")
        self._events_setup = True

    def _on_wheel(self, event):
        dy = event.get("dy", 0)
        if dy == 0:
            return
        vp_size = self._x_max - self._x_min
        total_len = max((s["length"] for s in self._series), default=0)
        if total_len <= 0 or vp_size <= 0:
            return

        factor = 1 / 1.1 if dy < 0 else 1.1
        new_size = max(10, int(vp_size * factor))

        cx = event.get("x", self.width / 2)
        cursor_ratio = cx / max(self.width, 1)
        cursor_idx = self._x_min + int(cursor_ratio * vp_size)

        new_x_min = cursor_idx - int(cursor_ratio * new_size)
        self.set_viewport(new_x_min, new_x_min + new_size)

    def _on_pointer_move(self, event):
        buttons = event.get("buttons", ())

        if 1 in buttons or 2 in buttons:
            if self._drag_start is not None:
                x = event.get("x", 0)
                dx = self._drag_start["x"] - x
                vp_size = self._x_max - self._x_min
                shift = int(dx / max(self.width, 1) * vp_size)
                new_min = self._drag_start["x_min"] + shift
                self.set_viewport(new_min, new_min + vp_size)
            return

        vp_size = self._x_max - self._x_min
        if vp_size <= 0 or not self._series:
            return
        x = event.get("x", 0)
        ratio = x / max(self.width, 1)
        idx = self._x_min + int(ratio * vp_size)
        total_len = max((s["length"] for s in self._series))
        idx = max(0, min(idx, total_len - 1))

        parts = [f"#{idx}"]
        for s in self._series:
            if idx < s["length"]:
                parts.append(f"{s['name']}={s['data_np'][idx]:.4f}")
        self._hover_text = "  ".join(parts)
        if self._status is not None:
            self._status.value = self._hover_text

    def _on_pointer_down(self, event):
        button = event.get("button", -1)
        if button in (0, 1):
            self._drag_start = {
                "x": event.get("x", 0),
                "x_min": self._x_min,
            }

    def _on_pointer_up(self, event):
        self._drag_start = None

    def _on_double_click(self, event):
        total_len = max((s["length"] for s in self._series), default=0)
        vp_size = min(1000, total_len)
        self.set_viewport(0, vp_size)

    # ── Internal: GPU ──────────────────────────────────────────────────

    def _make_bind_group(self, entry):
        return self.device.create_bind_group(
            layout=self._pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": self._vp_buffer,
                                             "offset": 0, "size": 16}},
                {"binding": 1, "resource": {"buffer": entry["buffer"],
                                             "offset": 0,
                                             "size": entry["length"] * 4}},
                {"binding": 2, "resource": {"buffer": entry["color_buffer"],
                                             "offset": 0, "size": 16}},
            ],
        )

    def _rebuild_pipeline(self):
        device = self.device
        shader = device.create_shader_module(
            code=VIEWPORT_VERTEX + VIEWPORT_FRAGMENT
        )
        self._pipeline = device.create_render_pipeline(
            label="viewport_pipeline",
            layout="auto",
            vertex=wgpu.VertexState(module=shader, entry_point="vs_main"),
            primitive=wgpu.PrimitiveState(
                topology=wgpu.PrimitiveTopology.line_strip,
            ),
            fragment=wgpu.FragmentState(
                module=shader,
                entry_point="fs_main",
                targets=[wgpu.ColorTargetState(format=self._fmt)],
            ),
        )

    def _write_vp_uniform(self):
        if self._vp_buffer is None:
            return
        data = struct.pack(
            "<IIff", self._x_min, self._x_max, self._y_min, self._y_max
        )
        self.device.queue.write_buffer(self._vp_buffer, 0, data)

    def draw_frame(self):
        # FPS measurement
        t = time.perf_counter()
        if self._frame_count == 0:
            self._fps_t0 = t
        self._frame_count += 1
        elapsed = t - self._fps_t0
        if elapsed >= 1.0:
            fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_t0 = t
            if self._status is not None:
                line = f"FPS: {fps:.1f}"
                if self._hover_text:
                    line += f"  |  {self._hover_text}"
                self._status.value = line

        # Render
        device = self.device
        ctx = self.context
        vp_size = self._x_max - self._x_min
        if vp_size <= 0 or not self._series:
            return

        texture = ctx.get_current_texture()
        if texture is None:
            return
        view = texture.create_view()

        encoder = device.create_command_encoder(label="frame_encoder")
        pass_ = encoder.begin_render_pass(
            label="main_pass",
            color_attachments=[
                wgpu.RenderPassColorAttachment(
                    view=view,
                    load_op=wgpu.LoadOp.clear,
                    store_op=wgpu.StoreOp.store,
                    clear_value=DARK_BG,
                )
            ],
        )

        pass_.set_pipeline(self._pipeline)
        for entry in self._series:
            series_vp_size = max(0, min(vp_size, entry["length"] - self._x_min))
            if series_vp_size <= 0:
                continue
            pass_.set_bind_group(0, entry["bind_group"])
            pass_.draw(series_vp_size)

        pass_.end()
        device.queue.submit([encoder.finish()])
