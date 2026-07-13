# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

import itertools
import time


class _ProxyDict(dict):
    """Dict with attribute-style access and automatic container lifecycle.

    Stores a copy of _generation_id at creation time.
    On __del__ (refcount GC), releases the container.
    info() is a bound method — no dynamic lambdas.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._generation_id = 0
        tid = self.get("_table_id")
        if tid is not None:
            entry = TABLE_REGISTRY.get(tid)
            if entry is not None:
                self._generation_id = entry.get("generation_id", 0)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __del__(self):
        tid = self.get("_table_id")
        if tid is not None:
            _release_table(tid)

    def _validate(self):
        tid = self.get("_table_id")
        if tid is None:
            raise RuntimeError("Proxy missing _table_id")
        entry = TABLE_REGISTRY.get(tid)
        if entry is None:
            raise RuntimeError(f"Underlying container '{tid}' was released")
        if entry.get("generation_id", -1) != getattr(self, "_generation_id", -1):
            raise RuntimeError(f"Container '{tid}' was recreated (generation mismatch)")

    def is_alive(self) -> bool:
        tid = self.get("_table_id")
        if tid is None:
            return False
        entry = TABLE_REGISTRY.get(tid)
        if entry is None:
            return False
        return entry.get("generation_id", -1) == getattr(self, "_generation_id", -2)

    def info(self):
        """Display structured info about this proxy's container."""
        entry = None
        tid = self.get("_table_id")
        if tid is not None:
            entry = TABLE_REGISTRY.get(tid)
        if entry is None:
            print(f"<Container '{tid}' not found>")
            return
        info_dict = _build_table_info(entry)
        if self.get("_series_id") or self.get("_is_hidden_series"):
            info_dict["_series_col"] = self.get("_col_name")
            info_dict["_series_length"] = self.get("_length")
            del info_dict["columns"][:]
            info_dict["columns"] = []
        _display(info_dict)

    def release(self):
        tid = self.get("_table_id")
        if tid is not None:
            _release_table(tid)


_tid_counter = itertools.count(1)
_sid_counter = itertools.count(1)
_tid_generation = itertools.count(1)

SERIES_REGISTRY: dict[str, dict] = {}
CHUNK_SIZE = 1000
ALIGNMENT = 256


def configure(*, chunk_size: int = 1000) -> None:
    global CHUNK_SIZE
    CHUNK_SIZE = chunk_size


def _calc_chunk_size() -> int:
    return (CHUNK_SIZE // ALIGNMENT) * ALIGNMENT


def _padded_size(n: int) -> int:
    return ((n + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT


def _is_scaled_chunk(chunk: dict) -> bool:
    meta = chunk.get("compression")
    return meta is not None and meta.get("type") == "scaled"


def _is_enum_chunk(chunk: dict) -> bool:
    meta = chunk.get("compression")
    return meta is not None and meta.get("type") == "enum"


def _slice_into_chunks(data: list, compression: dict | None = None) -> list[dict]:
    if not data:
        return []
    chunk_size = _calc_chunk_size()
    if chunk_size == 0:
        chunk_size = ALIGNMENT
    chunks = []
    for start in range(0, len(data), chunk_size):
        segment = data[start:start + chunk_size]
        valid = len(segment)
        padded = _padded_size(valid)
        if valid < padded:
            if segment and isinstance(segment[0], int):
                segment = list(segment) + [0] * (padded - valid)
            else:
                segment = list(segment) + [0.0] * (padded - valid)
        chunks.append({
            "buf": segment,
            "size": padded,
            "valid_elements": valid,
            "compression": compression,
        })
    return chunks


def register_series(data: list, context_id: str, compression: dict | None = None) -> str:
    sid = f"ser_{next(_sid_counter)}"
    chunks = _slice_into_chunks(data, compression)
    SERIES_REGISTRY[sid] = {
        "id": sid,
        "context_id": context_id,
        "length": len(data),
        "chunks": chunks,
        "compression": compression,
    }
    return sid


def get_series(series_id: str) -> dict | None:
    return SERIES_REGISTRY.get(series_id)


def remove_series(series_id: str) -> bool:
    if series_id in SERIES_REGISTRY:
        del SERIES_REGISTRY[series_id]
        return True
    return False


def get_chunks(series_id: str) -> list[dict] | None:
    entry = SERIES_REGISTRY.get(series_id)
    if entry is None:
        return None
    return entry["chunks"]


def series_data(series_id: str) -> list | None:
    entry = SERIES_REGISTRY.get(series_id)
    if entry is None:
        return None
    result = []
    comp = entry.get("compression")
    for c in entry["chunks"]:
        raw = c["buf"][:c["valid_elements"]]
        if comp and comp.get("type") == "scaled":
            from _core.compression import unpack_scaled
            raw = unpack_scaled(raw, comp)
        elif comp and comp.get("type") == "enum":
            from _core.compression import unpack_enum
            raw = unpack_enum(raw, comp)
        result.extend(raw)
    return result


TABLE_REGISTRY: dict[str, dict] = {}


def _register_table(rows: list, num_parts: int, schema: list,
                    layout: dict, created_by: str = "table") -> tuple:
    tid = f"tbl_{next(_tid_counter)}"
    num_rows = len(rows)
    byte_size = num_rows * num_parts * 4
    entry = {
        "id": tid,
        "rows": rows,
        "num_parts": num_parts,
        "num_rows": num_rows,
        "schema": schema,
        "layout": layout,
        "refcount": 1,
        "bytes": byte_size,
        "created_by": created_by,
        "last_access_ts": time.time(),
        "state": "LIVE",
        "generation_id": next(_tid_generation),
    }
    TABLE_REGISTRY[tid] = entry
    return tid, entry


def _get_table_entry(table_id: str) -> dict | None:
    return TABLE_REGISTRY.get(table_id)


def _release_table(table_id: str) -> bool:
    """Decrement refcount. If 0, mark ORPHAN (not dead yet — GC will sweep)."""
    entry = TABLE_REGISTRY.get(table_id)
    if entry is None:
        return False
    entry["refcount"] -= 1
    entry["last_access_ts"] = time.time()
    if entry["refcount"] <= 0 and entry.get("state") == "LIVE":
        entry["state"] = "ORPHAN"
        # Invalidate moments cache entries referencing this table
        _invalidate_moments_for(table_id, entry.get("generation_id", 0))
    return True


def _invalidate_moments_for(table_id: str, generation_id: int):
    """Remove MOMENTS cache entries matching (table_id, generation_id)."""
    from Stats._lib.stats_lib import MOMENTS
    drop = [k for k in MOMENTS
            if isinstance(k, tuple) and len(k) >= 2
            and k[0] == table_id and k[1] == generation_id]
    for k in drop:
        del MOMENTS[k]


def gc():
    """Sweep ORPHAN entries. Call explicitly or on allocation failure."""
    now = time.time()
    drop = []
    for tid, entry in list(TABLE_REGISTRY.items()):
        if entry.get("state") == "ORPHAN" and entry["refcount"] <= 0:
            entry["state"] = "DEAD"
            drop.append(tid)
    for tid in drop:
        del TABLE_REGISTRY[tid]
    return len(drop)


def release(proxy_or_tid):
    """Explicit release: decrements refcount. Accepts proxy dict or table_id str.

    Using this as the primary release mechanism is preferred over relying
    on __del__, which may be delayed or skipped in some Python environments
    (Jupyter history, exception frames, reference cycles).
    """
    if isinstance(proxy_or_tid, str):
        return _release_table(proxy_or_tid)
    if isinstance(proxy_or_tid, dict):
        tid = proxy_or_tid.get("_table_id")
        if tid is not None:
            return _release_table(tid)
    raise TypeError("Expected a table_id string or a proxy dict")


def clear_all():
    global _tid_counter, _sid_counter, _tid_generation
    SERIES_REGISTRY.clear()
    TABLE_REGISTRY.clear()
    _tid_counter = itertools.count(1)
    _sid_counter = itertools.count(1)
    _tid_generation = itertools.count(1)


def memory_info() -> dict:
    """Diagnostics: returns buffer counts and memory usage."""
    entries = list(TABLE_REGISTRY.values())
    live = sum(1 for e in entries if e.get("state") == "LIVE" and e["refcount"] > 0)
    orphan = sum(1 for e in entries if e.get("state") == "ORPHAN")
    total_buffers = len(entries)
    used_bytes = sum(e["bytes"] for e in entries)
    largest = max((e["bytes"] for e in entries), default=0)
    total_logical = 0
    for e in entries:
        ncols = len(e.get("schema", []))
        total_logical += e["num_rows"] * ncols * 4
    packed_bytes = sum(e["num_rows"] * sum(
        max(1, (c["size"] + 7) // 8) for c in e.get("layout", {}).get("columns", {}).values()
    ) for e in entries)
    saved_pct = 0.0
    if total_logical > 0:
        saved_pct = (1 - used_bytes / total_logical) * 100
    return {
        "buffers": total_buffers,
        "live": live,
        "orphan": orphan,
        "used_bytes": used_bytes,
        "packed_bytes": packed_bytes,
        "saved_pct": round(saved_pct, 1),
        "largest_buffer_bytes": largest,
    }


def debug_buffers() -> list[dict]:
    """Returns structured list of all registered buffers for debugging."""
    return [
        {
            "id": e["id"],
            "state": e.get("state", "UNKNOWN"),
            "bytes": e["bytes"],
            "refs": e["refcount"],
            "type": e.get("created_by", "?"),
            "gen": e.get("generation_id", 0),
        }
        for e in sorted(TABLE_REGISTRY.values(), key=lambda x: x["id"])
    ]


# ── Legacy info() helper (for backward compat with non-proxy access) ─────


def info(obj: dict | str) -> None:
    """Inspect a container or proxy.

    Accepts proxy dicts (with info() method) or raw table_id strings.
    """
    if isinstance(obj, str):
        entry = _get_table_entry(obj)
        if entry is None:
            print(f"<Table '{obj}' not found>")
            return
        _display(_build_table_info(entry))
        return

    if hasattr(obj, "info") and callable(obj.info) and obj.info.__func__ is not info:
        if isinstance(obj, _ProxyDict):
            obj.info()
            return

    # Fallback: infer from dict fields
    if obj.get("_series_id") or obj.get("_is_hidden_series"):
        tid = obj.get("_table_id")
        if tid is None:
            print("<Series proxy missing _table_id>")
            return
        entry = _get_table_entry(tid)
        if entry is None:
            print(f"<Table '{tid}' not found>")
            return
        info_dict = _build_table_info(entry)
        info_dict["_series_col"] = obj.get("_col_name")
        info_dict["_series_length"] = obj.get("_length")
        _display(info_dict)
        return

    if obj.get("_table_id"):
        entry = _get_table_entry(obj["_table_id"])
        if entry is None:
            print(f"<Table '{obj['_table_id']}' not found>")
            return
        _display(_build_table_info(entry))
        return

    print(f"<Unknown object: {type(obj).__name__}>")


# ── Info / diagnostics internals ────────────────────────────────────────


def _build_table_info(entry: dict) -> dict:
    """Собирает структурированные данные о таблице для отображения."""
    layout = entry["layout"]
    schema = entry["schema"]
    num_rows = entry["num_rows"]
    num_parts = entry["num_parts"]
    memory_total = num_rows * num_parts * 4

    columns = []
    logical_bits_total = 0
    for col in schema:
        name = col["name"]
        cinfo = layout["columns"][name]
        dtype = cinfo["dtype"]
        size = cinfo["size"]
        parts = cinfo["parts"]
        meta = cinfo.get("meta", {})

        packing_parts = []
        for p_idx, p_off, p_size in parts:
            packing_parts.append(f"part{p_idx}[{p_off}:{p_off + p_size})")
        packing_str = " + ".join(packing_parts)

        comp_parts = []
        if meta.get("_is_compressed_float"):
            comp_parts.append("compressed")
        if "scale" in meta:
            comp_parts.append(f"scale={meta['scale']}")
        if "offset" in meta:
            comp_parts.append(f"offset={meta['offset']}")
        comp_str = ", ".join(comp_parts) or "none"

        bytes_per_elem = max(1, (size + 7) // 8)
        col_memory = num_rows * bytes_per_elem
        logical_bits_total += size

        columns.append({
            "name": name,
            "dtype": dtype,
            "size_bits": size,
            "packing_str": packing_str,
            "compression_str": comp_str,
            "memory": col_memory,
        })

    # packing map
    parts_count = num_parts
    part_slots: list[list[dict]] = [[] for _ in range(parts_count)]
    for col in schema:
        name = col["name"]
        cinfo = layout["columns"][name]
        for p_idx, p_off, p_size in cinfo["parts"]:
            part_slots[p_idx].append({
                "name": name,
                "start": p_off,
                "end": p_off + p_size,
            })

    packing_map = []
    for pi, slots in enumerate(part_slots):
        used = sorted(slots, key=lambda x: x["start"])
        packing_map.append({
            "part_idx": pi,
            "slots": used,
        })

    # metrics
    physical_width = num_parts * 32
    avg_bits_per_row = logical_bits_total / max(1, len(columns))
    uncompressed_memory = num_rows * len(columns) * 4
    ratio = uncompressed_memory / max(1, memory_total)
    space_saved = (1 - memory_total / max(1, uncompressed_memory)) * 100

    return {
        "table_id": entry["id"],
        "num_rows": num_rows,
        "num_parts": num_parts,
        "memory_total": memory_total,
        "columns": columns,
        "packing_map": packing_map,
        "uncompressed_memory": uncompressed_memory,
        "compression_ratio": ratio,
        "space_saved": space_saved,
        "avg_bits_per_row": avg_bits_per_row,
        "physical_width": physical_width,
        "logical_bits_total": logical_bits_total,
        "num_columns": len(columns),
    }


def _display(info: dict) -> None:
    from _core.visualizer import render as _viz_render
    _viz_render(info)


def _fmt_bytes(n: int) -> str:
    """Форматирует байты в человеко-читаемый вид."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / 1024 ** 2:.2f} MB"
