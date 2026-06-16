# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""Dual-Mode Diagnostic Visualizer.

Auto-detects Jupyter (HTML/CSS grid) vs CLI (ANSI ASCII art).
Renders segmented bit-level memory maps with color-coded columns.
"""

import math

# ── color palette (6 cycling colors) ──────────────────────────────────
# each entry: (html_hex, ansi_bg_code, ansi_fg_code)

_COLORS = [
    ("#4CAF50", 22, 10),   # green
    ("#2196F3", 19, 12),   # blue
    ("#FF9800", 208, 214), # orange
    ("#9C27B0", 93, 99),   # purple
    ("#F44336", 124, 196), # red
    ("#00BCD4", 44, 50),   # teal
]


def _detect_env() -> str:
    """'jupyter' if inside Jupyter notebook, else 'cli'."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return "jupyter"
    except ImportError:
        pass
    return "cli"


def _col_color(idx: int) -> tuple:
    return _COLORS[idx % len(_COLORS)]


# ── helpers ────────────────────────────────────────────────────────────

def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.2f} MB"


def _fmt_ratio(ratio: float) -> str:
    return f"{ratio:.2f}x"


def _fmt_pct(v: float) -> str:
    return f"{v:.1f}%"


# ── display entry point ───────────────────────────────────────────────

def render(info: dict) -> None:
    """Detect environment and render info dict accordingly."""
    env = _detect_env()
    if env == "jupyter":
        _render_html(info)
    else:
        _render_cli(info)


# ═══════════════════════════════════════════════════════════════════════
# HTML renderer
# ═══════════════════════════════════════════════════════════════════════

def _render_html(info: dict) -> None:
    from IPython.display import display, HTML

    is_series = "_series_col" in info
    cols = info["columns"]
    pmap = info["packing_map"]
    html = _build_html(info, is_series, cols, pmap)
    display(HTML(html))


def _build_html(info: dict, is_series: bool, cols: list, pmap: list) -> str:
    p = _HtmlBuilder()
    p.writeln("<style>")
    p.writeln("  .nf-info { display: flex; flex-direction: column; font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace; font-size: 12px; line-height: 1.4; color: #e0e0e0; background: #1e1e2e; padding: 1.2em; border-radius: 0.6em; max-width: 1000px; }")
    p.writeln("  .nf-info h3 { margin: 0 0 0.6em 0; font-weight: 500; color: #ffd700; font-size: 1.1em; }")
    p.writeln("  .nf-info .meta { display: grid; grid-template-columns: 1fr 1fr; gap: 0.2em 1.8em; margin-bottom: 0.9em; }")
    p.writeln("  .nf-info .meta-item { display: flex; justify-content: space-between; }")
    p.writeln("  .nf-info .meta-key { color: #888; }")
    p.writeln("  .nf-info .meta-val { color: #fff; }")
    p.writeln("  .nf-info .part-row { display: flex; align-items: center; gap: 0.6em; margin: 0.2em 0; }")
    p.writeln("  .nf-info .part-label { color: #888; min-width: 4.5em; font-size: 0.9em; }")
    p.writeln("  .nf-info .bit-bar { display: flex; height: 2em; border-radius: 0.3em; overflow: hidden; flex: 1; }")
    p.writeln("  .nf-info .bit-seg { display: flex; align-items: center; justify-content: center; font-size: 0.7em; color: rgba(255,255,255,0.85); min-width: 0; position: relative; transition: 0.1s; }")
    p.writeln("  .nf-info .bit-seg:hover { filter: brightness(1.3); }")
    p.writeln("  .nf-info .bit-seg.free { background: #2a2a3e; color: #555; }")
    p.writeln("  .nf-info .col-table { width: 100%; table-layout: fixed; border-collapse: collapse; margin-top: 0.4em; font-size: 0.8em; }")
    p.writeln("  .nf-info .col-table th { color: #888; text-align: left; padding: 0.1em 0.3em; border-bottom: 1px solid #333; border-right: 1px solid #333; }")
    p.writeln("  .nf-info .col-table th:last-child { border-right: none; }")
    p.writeln("  .nf-info .col-table td { padding: 0.1em 0.3em; border-bottom: 1px solid #2a2a3e; border-right: 1px solid #2a2a3e; vertical-align: middle; text-align: left; overflow-wrap: anywhere; }")
    p.writeln("  .nf-info .col-table td:last-child { border-right: none; }")
    p.writeln("  .nf-info .col-table tr:hover td { background: #252540; }")
    p.writeln("  .nf-info .col-dot { display: inline-block; width: 0.6em; height: 0.6em; border-radius: 0.1em; margin-right: 0.3em; vertical-align: middle; }")
    p.writeln("  .nf-info .metrics { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.6em; margin: 0.6em 0 0 0; }")
    p.writeln("  .nf-info .metric-card { background: #252540; border-radius: 0.3em; padding: 0.4em 0.8em; text-align: center; }")
    p.writeln("  .nf-info .metric-card .val { font-size: 1.1em; font-weight: 600; }")
    p.writeln("  .nf-info .metric-card .lbl { font-size: 0.8em; color: #888; margin-top: 0.1em; }")
    p.writeln("</style>")

    p.writeln('<div class="nf-info">')
    title = "Series Info" if is_series else "Table Info"
    p.writeln(f"<h3>▎{title}</h3>")

    # meta grid
    p.writeln('<div class="meta">')
    p.meta("Table ID", info["table_id"])
    p.meta("Rows", f"{info['num_rows']:,}")
    p.meta("Parts/row", str(info["num_parts"]))
    p.meta("Physical mem", _fmt_bytes(info["memory_total"]))
    if is_series:
        p.meta("Column", info["_series_col"])
        p.meta("Length", str(info["_series_length"]))
        p.meta("Category", "Standalone Series: Managed via Isolated Mono-Layout Table")

    else:
        p.meta("Category", "Multi-Column Dataframe")
        p.meta("Columns", str(len(cols)))
    p.writeln("</div>")

    # ── column index + split detection (before bit-bar) ────────────
    col_idx = {col["name"]: i for i, col in enumerate(cols)}
    col_part_count: dict[str, int] = {}
    for part in pmap:
        for s in part["slots"]:
            col_part_count[s["name"]] = col_part_count.get(s["name"], 0) + 1
    split_cols = {name for name, cnt in col_part_count.items() if cnt > 1}

    # bit-level memory map
    p.writeln("<b style='color:#aaa;font-size:1.2em;'>Bit-level memory map:</b>")
    for part in pmap:
        pi = part["part_idx"]
        slots = part["slots"]
        p.writeln('<div class="part-row">')
        p.writeln(f'<span class="part-label">part{pi}</span>')
        p.writeln('<div class="bit-bar">')
        pos = 0
        for si, s in enumerate(slots):
            if s["start"] > pos:
                free_n = s["start"] - pos
                label = "FREE" if free_n > 4 else ""
                p.writeln(f'<div class="bit-seg free" style="flex-grow:{free_n};" title="{free_n} free bits">{label}</div>')
            seg_n = s["end"] - s["start"]
            ci = col_idx[s["name"]]
            html_c, _, _ = _col_color(ci)
            label = s["name"] if seg_n > 16 else ""
            p.writeln(f'<div class="bit-seg" style="flex-grow:{seg_n};background:{html_c};" title="{s["name"]} [{s["start"]}:{s["end"]})">{label}</div>')
            pos = s["end"]
        if pos < 32:
            free_n = 32 - pos
            label = "FREE" if free_n > 4 else ""
            p.writeln(f'<div class="bit-seg free" style="flex-grow:{free_n};" title="{free_n} free bits">{label}</div>')
        p.writeln("</div></div>")

    # columns table
    p.writeln('<table class="col-table">')
    p.writeln('<colgroup><col style="width:4%"><col style="width:18%"><col style="width:12%"><col style="width:5%"><col style="width:29%"><col style="width:25%"><col style="width:7%"></colgroup>')
    p.writeln("<tr><th>#</th><th>Column</th><th>Dtype</th><th>Bits</th><th>Packing</th><th>Compression</th><th>Memory</th></tr>")
    for i, col in enumerate(cols):
        html_c, _, _ = _col_color(i)
        split_mark = " ↗" if col["name"] in split_cols else ""
        p.writeln(f"<tr>"
                  f"<td>{i}</td>"
                  f'<td><span class="col-dot" style="background:{html_c};"></span>{col["name"]}{split_mark}</td>'
                  f"<td>{col['dtype']}</td>"
                  f"<td>{col['size_bits']}</td>"
                  f'<td>{col["packing_str"]}</td>'
                  f'<td>{col["compression_str"]}</td>'
                  f"<td>{_fmt_bytes(col['memory'])}</td>"
                  f"</tr>")
    p.writeln("</table>")

    # metrics cards
    mem = info["memory_total"]
    logical_mem = info.get("uncompressed_memory", 0)
    ratio = info.get("compression_ratio", 1.0)
    saved = info.get("space_saved", 0.0)
    p.writeln('<div class="metrics">')
    p.metric_card(_fmt_bytes(logical_mem) if logical_mem else _fmt_bytes(mem), "Uncompressed")
    p.metric_card(_fmt_bytes(mem), "Physical")
    p.metric_card(_fmt_ratio(ratio), "Compression Ratio" if ratio != 1.0 else "Logical Width")
    p.writeln("</div>")
    if saved > 0:
        p.writeln(f"<div style='color:#4CAF50;font-size:1.2em;margin-top:0.4em;'>Space saved: {_fmt_pct(saved)}</div>")

    p.writeln("</div>")
    return p.result


# ═══════════════════════════════════════════════════════════════════════
# CLI / ANSI renderer
# ═══════════════════════════════════════════════════════════════════════

def _render_cli(info: dict) -> None:
    print(_build_cli(info), end="")


def _build_cli(info: dict) -> str:
    lines = []
    a = _Ansi

    is_series = "_series_col" in info
    cols = info["columns"]
    pmap = info["packing_map"]

    # header
    title = " Series Info " if is_series else " Table Info "
    lines.append(f"{a.BOLD}{a.YELLOW}═══{title}{'═' * (56 - len(title))}{a.RESET}\n")

    # meta
    meta_lines = [
        ("Table ID", info["table_id"]),
        ("Rows", f"{info['num_rows']:,}"),
        ("Parts/row", str(info["num_parts"])),
        ("Physical", _fmt_bytes(info["memory_total"])),
    ]
    if is_series:
        meta_lines.insert(0, ("Column", f"{info['_series_col']} ({info['_series_length']} elems)"))
        meta_lines.insert(1, ("Category", "Standalone Series: Managed via Isolated Mono-Layout Table"))
    else:
        meta_lines.append(("Category", "Multi-Column Dataframe"))

    for key, val in meta_lines:
        lines.append(f"  {a.DIM}{key:<12}{a.RESET} {val}\n")

    # bit-level memory map
    lines.append(f"\n  {a.BOLD}Bit-level memory map:{a.RESET}\n")
    col_idx = {}
    for part in pmap:
        pi = part["part_idx"]
        slots = part["slots"]
        bar_chars = []
        pos = 0
        for s in slots:
            ci = len(col_idx)
            if s["name"] not in col_idx:
                col_idx[s["name"]] = ci
            ansi_bg = _col_color(col_idx[s["name"]])[1]
            if s["start"] > pos:
                free_n = s["start"] - pos
                bar_chars.append(f"{_ANSI_BG_RESET}{' ' * free_n}")
            seg_n = s["end"] - s["start"]
            bar_chars.append(f"\033[48;5;{ansi_bg}m{' ' * seg_n}")
            pos = s["end"]
        if pos < 32:
            bar_chars.append(f"{_ANSI_BG_RESET}{' ' * (32 - pos)}")

        bar = "".join(bar_chars)
        lines.append(f"  {a.DIM}part{pi:<5}{a.RESET} │{bar}{_ANSI_BG_RESET}│\n")

    lines.append(f"  {a.DIM}{'─' * 65}{a.RESET}\n")

    # legend and columns
    lines.append(f"  {'#':>3}  {'Column':<12} {'Dtype':<14} {'Bits':<6} {'Packing':<24} {'Memory':>8}\n")
    lines.append(f"  {a.DIM}{'───':>3}  {'────────────':<12} {'──────────────':<14} {'──────':<6} {'────────────────────────':<24} {'────────':>8}{a.RESET}\n")

    for i, col in enumerate(cols):
        col_idx[col["name"]] = i
        ansi_bg = _col_color(i)[1]
        dot = f"\033[48;5;{ansi_bg}m  {_ANSI_BG_RESET}"
        lines.append(
            f"  {i:>3}  {dot} {col['name']:<10} {col['dtype']:<14} "
            f"{col['size_bits']:<6} {a.DIM}{col['packing_str']:<24}{a.RESET} "
            f"{_fmt_bytes(col['memory']):>8}\n"
        )

    # metrics
    mem = info["memory_total"]
    logical_mem = info.get("uncompressed_memory", 0)
    ratio = info.get("compression_ratio", 1.0)
    saved = info.get("space_saved", 0.0)
    lines.append(f"\n  {a.BOLD}Metrics:{a.RESET}\n")
    lines.append(f"  {a.DIM}Uncompressed:{a.RESET} {_fmt_bytes(logical_mem) if logical_mem else '—':>8}   ")
    lines.append(f"{a.DIM}Physical:{a.RESET} {_fmt_bytes(mem):>8}   ")
    if ratio != 1.0:
        lines.append(f"{a.DIM}Ratio:{a.RESET} {_fmt_ratio(ratio):>6}   ")
        lines.append(f"{a.DIM}Saved:{a.RESET} {_fmt_pct(saved)}\n")
    else:
        lines.append(f"{a.DIM}Width:{a.RESET} {info.get('avg_bits_per_row', 32):.0f} bit/row\n")

    lines.append(f"{a.RESET}")
    return "".join(lines)


# ── ANSI escape helpers ──────────────────────────────────────────────

class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    YELLOW = "\033[33m"

_ANSI_BG_RESET = "\033[49m"


# ── HTML string builder ──────────────────────────────────────────────

class _HtmlBuilder:
    def __init__(self):
        self.lines = []

    def writeln(self, s: str = ""):
        self.lines.append(s)

    def meta(self, key: str, val: str):
        self.lines.append(f'<div class="meta-item"><span class="meta-key">{key}</span><span class="meta-val">{val}</span></div>')

    def metric_card(self, val: str, lbl: str):
        self.lines.append(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>')

    @property
    def result(self) -> str:
        return "\n".join(self.lines)
