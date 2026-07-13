/**
 * Client-side chart: поддерживает два режима.
 *
 * 1) «full-data» — полный датасет получен один раз, viewport локальный.
 * 2) «server-gpu» — данные на GPU сервера, виджет получает только срез окна.
 *
 * Режим определяется автоматически по первому полученному пакету:
 *   type="full"  → full-data
 *   type="slice" → server-gpu
 */

function render({ model, el }) {
  // ── Состояние ────────────────────────────────────────────────────────
  let serverGpu = false;     // true — режим server-gpu
  let seriesList = [];       // [{name, color, data:Float32Array}, ...]
  let dataLen = 0;           // общая длина (для full-data)
  let visW = 1000;           // точек в видимом окне
  let scrollPos = 0;         // смещение
  let dragStart = null;
  let rafId = null;
  let frameDirty = true;
  let canvas = null;
  let ctx = null;
  let zoomSlider = null;
  let scrollSlider = null;
  let infoEl = null;
  let scrollLbl = null;
  let zoomLbl = null;
  let valLbl = null;
  let pending = false;       // ожидание ответа от сервера
  let tReq = 0;              // метка времени последнего запроса
  let timingEl = null;

  // ── DOM ──────────────────────────────────────────────────────────────
  function buildDOM() {
    el.style.background = "#0d1117";
    el.style.padding = "0";
    el.style.fontFamily = "monospace";

    canvas = document.createElement("canvas");
    canvas.style.display = "block";
    canvas.style.cursor = "crosshair";
    el.appendChild(canvas);

    const ctrl = document.createElement("div");
    ctrl.style.display = "flex";
    ctrl.style.flexWrap = "wrap";
    ctrl.style.alignItems = "center";
    ctrl.style.gap = "8px";
    ctrl.style.padding = "6px 0";

    zoomLbl = document.createElement("span");
    zoomLbl.style.color = "#8b949e";
    zoomLbl.style.fontSize = "12px";
    zoomLbl.style.minWidth = "70px";
    zoomLbl.textContent = "100%";
    ctrl.appendChild(zoomLbl);

    zoomSlider = document.createElement("input");
    zoomSlider.type = "range";
    zoomSlider.min = "0";
    zoomSlider.max = "100";
    zoomSlider.value = "100";
    zoomSlider.style.flex = "1";
    zoomSlider.style.minWidth = "60px";
    zoomSlider.style.height = "6px";
    zoomSlider.style.accentColor = "#58a6ff";
    ctrl.appendChild(zoomSlider);

    scrollLbl = document.createElement("span");
    scrollLbl.style.color = "#8b949e";
    scrollLbl.style.fontSize = "12px";
    scrollLbl.style.minWidth = "80px";
    scrollLbl.textContent = "0 / 0";
    ctrl.appendChild(scrollLbl);

    scrollSlider = document.createElement("input");
    scrollSlider.type = "range";
    scrollSlider.min = "0";
    scrollSlider.max = "0";
    scrollSlider.value = "0";
    scrollSlider.style.flex = "1";
    scrollSlider.style.minWidth = "60px";
    scrollSlider.style.height = "6px";
    scrollSlider.style.accentColor = "#58a6ff";
    ctrl.appendChild(scrollSlider);

    el.appendChild(ctrl);

    infoEl = document.createElement("div");
    infoEl.style.color = "#8b949e";
    infoEl.style.fontSize = "12px";
    infoEl.style.display = "flex";
    infoEl.style.gap = "16px";
    infoEl.style.padding = "2px 0";
    valLbl = document.createElement("span");
    valLbl.textContent = "\u2014";
    infoEl.appendChild(valLbl);
    timingEl = document.createElement("span");
    timingEl.style.color = "#484f58";
    timingEl.style.fontSize = "11px";
    timingEl.style.marginLeft = "auto";
    timingEl.textContent = "\u2014";
    infoEl.appendChild(timingEl);
    el.appendChild(infoEl);

    resizeCanvas();
    setupEvents();
    setupSliders();
  }

  function resizeCanvas() {
    const w = model.get("width") || 1000;
    const h = model.get("height") || 260;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx._w = w;
    ctx._h = h;
  }

  // ── Получение данных ─────────────────────────────────────────────────
  model.on("change:_window_data", () => {
    const raw = model.get("_window_data");
    if (!raw) return;
    try {
      const msg = JSON.parse(raw);
      if (msg.type === "full") acceptFull(msg);
      else if (msg.type === "slice") acceptSlice(msg);
    } catch (e) { console.error("decode error", e); }
  });

  function b64ToF32(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }

  function acceptFull(msg) {
    serverGpu = false;
    const allData = b64ToF32(msg.data);
    seriesList = [];
    let offset = 0;
    for (const s of msg.series) {
      const data = allData.slice(offset, offset + s.length);
      offset += s.length;
      seriesList.push({ name: s.name, color: rgba2css(s.color), data });
    }
    dataLen = Math.max(...msg.series.map(s => s.length), 0);
    resizeCanvas();
    initViewport();
    frameDirty = true;
    scheduleRender();
  }

  function acceptSlice(msg) {
    serverGpu = true;
    pending = false;
    const allData = b64ToF32(msg.data);
    scrollPos = msg.xMin;
    visW = msg.xMax - msg.xMin;
    seriesList = [];
    let offset = 0;
    for (const s of msg.series) {
      const data = allData.slice(offset, offset + s.length);
      offset += s.length;
      seriesList.push({ name: s.name, color: rgba2css(s.color), data });
    }
    dataLen = msg.totalLen || 0;

    // update sliders to match server viewport
    zoomSlider.value = sldZoomFromVisW();
    const maxScroll = Math.max(0, dataLen - visW);
    scrollSlider.max = maxScroll;
    scrollSlider.value = Math.min(scrollPos, maxScroll);
    scrollLbl.textContent = scrollPos + " / " + maxScroll;
    zoomLbl.textContent = Math.round(((canvas._w || 800) / visW) * 100) + "%";

    if (valLbl) valLbl.textContent = seriesList.length + "s  " + msg.xMin + "\u2013" + msg.xMax;
    const source = msg.xMax - msg.xMin;
    const visible = seriesList[0]?.data.length || 0;
    const ratio = visible > 0 ? (source / visible).toFixed(1) : "?";
    const netMs = performance.now() - tReq;
    const aggMs = msg.aggMs || 0;
    const genMs = msg.genMs || 0;
    timingEl.textContent = [
      "source: " + source,
      "visible: " + visible,
      "ratio: " + ratio + "x",
      "|  gen: " + genMs + "ms",
      "agg: " + aggMs + "ms",
      "net: " + netMs.toFixed(1) + "ms",
    ].join("  ");
    frameDirty = true;
    scheduleRender();
  }

  // ── Viewport init ────────────────────────────────────────────────────
  function initViewport() {
    if (dataLen <= 0) return;
    visW = Math.min(1000, dataLen);
    scrollPos = 0;
    updateSliders();
  }

  // ── Sliders ──────────────────────────────────────────────────────────
  function setupSliders() {
    zoomSlider.addEventListener("input", () => {
      const center = scrollPos + visW / 2;
      visW = visWFromSldZoom();
      const maxS = Math.max(0, dataLen - visW);
      scrollPos = Math.max(0, Math.min(maxS, Math.round(center - visW / 2)));
      updateSliders();
      if (serverGpu) requestFrame();
      else { frameDirty = true; scheduleRender(); }
    });
    scrollSlider.addEventListener("input", () => {
      scrollPos = parseInt(scrollSlider.value) || 0;
      updateSliders();
      if (serverGpu) requestFrame();
      else { frameDirty = true; scheduleRender(); }
    });
  }

  function sldZoomFromVisW() {
    const minW = Math.min(canvas._w || 800, dataLen || 1);
    const z = (visW - minW) / Math.max(1, (dataLen || 1) - minW);
    const clamped = Math.max(0, Math.min(1, 1 - Math.sqrt(z)));
    return Math.round(clamped * 100);
  }

  function visWFromSldZoom() {
    const z = parseFloat(zoomSlider.value) / 100;
    const minW = Math.min(canvas._w || 800, dataLen || 1);
    const n = dataLen || 1;
    return Math.round(minW + (n - minW) * Math.pow(1 - z, 2));
  }

  function applyZoom() {
    if (dataLen <= 0) return;
    const center = scrollPos + visW / 2;
    const z = parseFloat(zoomSlider.value) / 100;
    const minW = Math.min(canvas._w || 800, dataLen);
    visW = Math.round(minW + (dataLen - minW) * Math.pow(1 - z, 2));
    visW = Math.max(minW, Math.min(dataLen, visW));
    const maxS = Math.max(0, dataLen - visW);
    scrollPos = Math.max(0, Math.min(maxS, Math.round(center - visW / 2)));
    updateSliders();
  }

  function updateSliders() {
    if (!scrollSlider || !zoomSlider) return;
    const maxScroll = Math.max(0, dataLen - visW);
    scrollSlider.max = maxScroll;
    scrollSlider.value = Math.min(scrollPos, maxScroll);
    scrollLbl.textContent = scrollPos + " / " + maxScroll;
    const pct = Math.round(((canvas._w || 800) / visW) * 100);
    zoomLbl.textContent = Math.min(pct, 100) + "%";
  }

  // ── Server-GPU: запрос кадра ─────────────────────────────────────────
  function requestFrame() {
    if (pending) return;
    pending = true;
    tReq = performance.now();
    const { pixW } = dim();
    model.send({
      type: "viewport",
      xMin: scrollPos,
      xMax: scrollPos + visW,
      cw: pixW,
      ch: canvas._h || 260,
    });
    valLbl.textContent = ".";
  }

  // ── Mouse ────────────────────────────────────────────────────────────
  function setupEvents() {
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      if (!seriesList.length) return;
      const { plotLeft, pixW } = dim();
      const frac = Math.max(0, Math.min(1, (e.offsetX - plotLeft) / pixW));
      const oldIdx = scrollPos + frac * visW;
      const factor = e.deltaY < 0 ? 1 / 1.1 : 1.1;
      const newW = Math.max(pixW, Math.min(dataLen, Math.round(visW * factor)));
      const maxS = Math.max(0, dataLen - newW);
      scrollPos = Math.max(0, Math.min(maxS, Math.round(oldIdx - frac * newW)));
      visW = newW;
      updateSliders();
      if (serverGpu) requestFrame();
      else { frameDirty = true; scheduleRender(); }
    }, { passive: false });

    canvas.addEventListener("pointerdown", (e) => {
      if (e.button === 0) {
        dragStart = { x: e.clientX, sp: scrollPos };
        canvas.setPointerCapture(e.pointerId);
      }
    });

    canvas.addEventListener("pointermove", (e) => {
      if (dragStart) {
        const dx = dragStart.x - e.clientX;
        const pw = dim().pixW;
        const shift = Math.round((dx / pw) * visW);
        const maxS = Math.max(0, (dataLen || 1) - visW);
        scrollPos = Math.max(0, Math.min(maxS, dragStart.sp + shift));
        scrollSlider.value = scrollPos;
        scrollLbl.textContent = scrollPos + " / " + maxS;
        if (serverGpu) {
          requestFrame();
        } else {
          frameDirty = true;
          scheduleRender();
        }
      } else {
        // Hover — локально (только если есть данные)
        if (!seriesList.length) return;
        const pw = dim().pixW;
        const pl = dim().plotLeft;
        const px = e.offsetX - pl;
        if (serverGpu) {
          const n = seriesList[0]?.data.length || 1;
          const ci = Math.max(0, Math.min(Math.round((px / pw) * (n - 1)), n - 1));
          const parts = ["#" + ci];
          for (const s of seriesList) {
            if (ci < s.data.length) parts.push(s.name + "=" + s.data[ci].toFixed(4));
          }
          valLbl.textContent = parts.join("  ");
        } else {
          const idx = scrollPos + Math.round((px / pw) * visW);
          if (idx >= 0 && idx < dataLen) {
            const parts = ["#" + idx];
            for (const s of seriesList) {
              if (idx < s.data.length) parts.push(s.name + "=" + s.data[idx].toFixed(4));
            }
            valLbl.textContent = parts.join("  ");
          }
        }
      }
    });

    canvas.addEventListener("pointerup", () => { dragStart = null; });
    canvas.addEventListener("pointerleave", () => { dragStart = null; });

    canvas.addEventListener("dblclick", () => {
      if (!seriesList.length) return;
      if (serverGpu) {
        scrollPos = 0;
        visW = Math.min(1000, dataLen);
        updateSliders();
        zoomSlider.value = sldZoomFromVisW();
        requestFrame();
      } else {
        initViewport();
        frameDirty = true;
        scheduleRender();
      }
    });
  }

  function dim() {
    const w = ctx._w || 800;
    const h = ctx._h || 260;
    const plotLeft = 56;
    return { w, h, plotLeft, pixW: w - plotLeft, plotH: h - 16 - 16 };
  }

  // ── Render ───────────────────────────────────────────────────────────
  function scheduleRender() {
    if (rafId !== null) return;
    rafId = requestAnimationFrame(() => {
      rafId = null;
      if (frameDirty) draw();
    });
  }

  function draw() {
    frameDirty = false;
    const { w, h, plotLeft, pixW, plotH } = dim();
    if (pixW <= 0 || plotH <= 0) return;

    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, w, h);

    if (!seriesList.length) {
      ctx.fillStyle = "#8b949e";
      ctx.font = "14px monospace";
      ctx.fillText(pending ? "loading..." : "waiting...", 10, 20);
      return;
    }

    const end = Math.min(scrollPos + visW, dataLen || visW);
    const visible = end - scrollPos;
    if (visible <= 0) return;

    let mn = Infinity, mx = -Infinity;
    for (const s of seriesList) {
      for (let i = 0; i < s.data.length; i++) {
        const v = s.data[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    const range = mx - mn || 1;

    function yPx(v) { return 16 + (1 - (v - mn) / range) * plotH; }

    ctx.fillStyle = "#8b949e";
    ctx.font = "10px monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i <= 5; i++) {
      const val = mn + (range * i) / 5;
      const y = yPx(val);
      ctx.fillText(val.toFixed(4), plotLeft - 6, y);
      ctx.strokeStyle = "#1a1f2e";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plotLeft, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    for (const s of seriesList) {
      ctx.beginPath();
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      let started = false;

      if (serverGpu) {
        // Сервер уже сжал данные до cw точек — равномерно растягиваем по ширине
        const n = s.data.length;
        const div = Math.max(1, n - 1);
        for (let i = 0; i < n; i++) {
          const x = plotLeft + (i / div) * pixW;
          const y = yPx(s.data[i]);
          if (!started) { ctx.moveTo(x, y); started = true; }
          else ctx.lineTo(x, y);
        }
      } else {
        const step = Math.max(1, Math.floor(visible / pixW));
        for (let px = 0; px < pixW; px++) {
          const idx = scrollPos + px * step;
          if (idx >= s.data.length) break;
          const y = yPx(s.data[idx]);
          const x = plotLeft + px;
          if (!started) { ctx.moveTo(x, y); started = true; }
          else ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    ctx.strokeStyle = "#30363d";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotLeft, 16);
    ctx.lineTo(plotLeft, 16 + plotH);
    ctx.stroke();

    // Обновляем тайминг с render
    if (serverGpu) {
      const drawMs = performance.now() - tReq;
      const cur = timingEl.textContent;
      const idx = cur.indexOf("render:");
      if (idx >= 0) {
        timingEl.textContent = cur.replace(/render: [\d.]+ms/, "render: " + drawMs.toFixed(1) + "ms");
      } else {
        timingEl.textContent += "  render: " + drawMs.toFixed(1) + "ms";
      }
    }
  }

  // ── Helpers ──────────────────────────────────────────────────────────
  function rgba2css(c) {
    if (Array.isArray(c)) {
      const [r, g, b, a] = c;
      return `rgba(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0},${a})`;
    }
    return String(c);
  }

  // ── Init ─────────────────────────────────────────────────────────────
  buildDOM();

  const existing = model.get("_window_data");
  if (existing) {
    try {
      const msg = JSON.parse(existing);
      if (msg.type === "full") acceptFull(msg);
      else if (msg.type === "slice") acceptSlice(msg);
    } catch (e) { /* ignore */ }
  }

  // Запрашиваем первый кадр (на случай server-gpu: show() был вызван до add_series)
  if (serverGpu || !existing) {
    model.send({ type: "init" });
  }

  return () => {
    if (rafId !== null) cancelAnimationFrame(rafId);
  };
}

export default { render };
