"""Phase 4 - ROLLINGSTDDEV consumer verification (S102c).

Spec: ROLLINGSTDDEV_SLICE_SPEC.md sec.9 (S102c).

Consumer map (by ISA lowering, lowering.py):
  SMA(data, 14)   -> RollingSum(period=14) -> MapBinary(DIV 14)   [NO RollingStdDev]
  ATR(h,l,c,14)   -> TrueRange -> RollingSum(period=14) -> MapBinary(DIV 14) [NO RollingStdDev]
  EMA(data,14)    -> StateKernel_Single(mode=0) [NO RollingStdDev]
  RSI(data,14)    -> StateKernel_Dual + StateKernel_Single x2 + Map/MapBinary/Shift [NO RollingStdDev]
  Trading: BollingerBands(Close, period) -> own kernel (inline SMA+StdDev, D8 family) KEEP
           CCI(High,Low,Close, period) -> own kernel (inline, D8) KEEP
           No live consumers of Compute RollingStdDev primitive - 0 live.

Graph assertions: lower(ast_to_dict(parse(expr))) for SMA/ATR/EMA/RSI must NOT contain RollingStdDev node (regression protection).
Trading kernels BollingerBands/CCI must remain KEEP (D8 family) - deletion map D12 shows 0 live consumers.

Evidence: evidence/rollingstddev_slice_phase4/consumers.json

IMPORTANT: this file is ASCII-only (no Cyrillic) because
tests/test_backend.py::test_no_cupy_in_test_files reads tests/*.py with
open() in locale encoding (cp1251 on this host).
"""

import json
import os
import pathlib
import sys

import numpy as np

_PATH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _sub in (".", "src/core", "src/math"):
    _p = os.path.join(_PATH_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AST._lib.parser import parse
from AST._lib.ast_nodes import ast_to_dict

CHECKPOINT_SHA = "a60e39b0c49434231433faac671369b8f7e34c2f"
SLICE = "rollingstddev"

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "rollingstddev_slice_phase4")

try:
    import wgpu
    _adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
except Exception:  # noqa: BLE001
    _adapter = None
GPU_AVAILABLE = _adapter is not None
ADAPTER_INFO = dict(_adapter.info) if _adapter is not None else {}


def _versions():
    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import wgpu
        v["wgpu"] = wgpu.__version__
    except Exception:  # noqa: BLE001
        v["wgpu"] = "n/a"
    v["gpu"] = "n/a"
    if _adapter is not None:
        info = dict(_adapter.info)
        v["gpu"] = f"{info.get('device', 'unknown')} ({info.get('backend_type', 'unknown')})"
    return v


def _graph_for(expr):
    try:
        from ISA._lib.lowering import lower, format_jobs
    except Exception as e:
        # fallback: import via Compute if ISA not mountable directly
        raise
    ast = parse(expr)
    d = ast_to_dict(ast)
    graph, out = lower(d)
    jobs = format_jobs(graph, out)
    return graph, out, jobs


def _rolling_stddev_nodes(graph):
    return [nd for nd in graph if nd["op"] == "RollingStdDev"]


def test_consumer_graph_sma_no_rollstd():
    graph, out, jobs = _graph_for("SMA(data, 14)")
    rs = _rolling_stddev_nodes(graph)
    assert rs == [], f"SMA must NOT use RollingStdDev, got {rs}"
    assert any(nd["op"] == "RollingSum" for nd in graph), "SMA must use RollingSum"


def test_consumer_graph_atr_no_rollstd():
    graph, out, jobs = _graph_for("ATR(h, l, c, 14)")
    rs = _rolling_stddev_nodes(graph)
    assert rs == [], f"ATR must NOT use RollingStdDev, got {rs}"


def test_consumer_graph_ema_no_rollstd():
    graph, out, jobs = _graph_for("EMA(data, 14)")
    rs = _rolling_stddev_nodes(graph)
    assert rs == [], f"EMA must NOT use RollingStdDev, got {rs}"
    assert any(nd["op"] == "StateKernel_Single" for nd in graph)


def test_consumer_graph_rsi_no_rollstd():
    graph, out, jobs = _graph_for("RSI(data, 14)")
    rs = _rolling_stddev_nodes(graph)
    assert rs == [], f"RSI must NOT use RollingStdDev, got {rs}"
    assert any(nd["op"] == "StateKernel_Dual" for nd in graph)


def test_trading_bollingerbands_cci_keep():
    # D8 family kernels must remain - not deleted, not migrated to RollingStdDev
    # Check that Trading kernels exist and are KEEP
    import importlib.util
    import pathlib as pl
    bb_desc = pl.Path(_PATH_ROOT) / "src" / "Trading" / "_lib" / "bollingerbands" / "descriptor.py"
    cci_desc = pl.Path(_PATH_ROOT) / "src" / "Trading" / "_lib" / "cci" / "descriptor.py"
    assert bb_desc.exists(), "BollingerBands descriptor must exist (D8 KEEP)"
    assert cci_desc.exists(), "CCI descriptor must exist (D8 KEEP)"
    # verify they don't use RollingStdDev primitive internally (they inline)
    bb_text = bb_desc.read_text(encoding="utf-8")
    cci_text = cci_desc.read_text(encoding="utf-8")
    assert "RollingStdDev" not in bb_text, "BollingerBands must inline, not use RollingStdDev primitive"
    assert "RollingStdDev" not in cci_text, "CCI must inline, not use RollingStdDev primitive"


def test_consumers_evidence_saved():
    """Write consumers.json (0 live consumers) and delete_map_d12.json."""
    graph_sma, out_sma, jobs_sma = _graph_for("SMA(data, 14)")
    graph_atr, out_atr, jobs_atr = _graph_for("ATR(h, l, c, 14)")
    graph_ema, out_ema, jobs_ema = _graph_for("EMA(data, 14)")
    graph_rsi, out_rsi, jobs_rsi = _graph_for("RSI(data, 14)")

    consumer_map = {
        "SMA": {
            "uses_rollingstddev": False,
            "graph": graph_sma,
            "note": "RollingSum + MapBinary(DIV) - RollingStdDev NOT used",
        },
        "ATR": {
            "uses_rollingstddev": False,
            "graph": graph_atr,
            "note": "TrueRange + RollingSum + MapBinary(DIV) - RollingStdDev NOT used",
        },
        "EMA": {
            "uses_rollingstddev": False,
            "graph": graph_ema,
            "note": "StateKernel_Single(mode=0) - RollingStdDev NOT used",
        },
        "RSI": {
            "uses_rollingstddev": False,
            "graph": graph_rsi,
            "note": "StateKernel_Dual + StateKernel_Single x2 + Map/MapBinary/Shift - RollingStdDev NOT used",
        },
        "BollingerBands": {
            "uses_rollingstddev": False,
            "status": "KEEP (D8 family)",
            "location": "src/Trading/_lib/bollingerbands",
            "note": "inline SMA+StdDev, NOT migrated to Compute RollingStdDev primitive",
        },
        "CCI": {
            "uses_rollingstddev": False,
            "status": "KEEP (D8 family)",
            "location": "src/Trading/_lib/cci",
            "note": "inline MeanDeviation, D8 family - KEEP",
        },
    }

    evidence = {
        "phase": 4,
        "slice": SLICE,
        "checkpoint_sha": CHECKPOINT_SHA,
        "versions": _versions(),
        "result": {
            "consumer_map": consumer_map,
            "live_consumers": 0,
            "live_consumers_list": [],
            "trading_keep": ["BollingerBands", "CCI"],
            "trading_family": "D8",
            "gpu_available": GPU_AVAILABLE,
            "note": "graph assertions via lower(ast_to_dict(parse(expr))): "
                    "SMA/ATR/EMA/RSI -> NO RollingStdDev node "
                    "(resolver regression protection, 0 live consumers). "
                    "Trading BollingerBands/CCI remain KEEP D8-family "
                    "(inline kernels, NOT using Compute RollingStdDev primitive). "
                    "D12 delete map: 0 live consumers -> no deletion, KEEP.",
        },
        "evidence_schema": "v1",
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / "consumers.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    assert os.path.getsize(path) > 0, "consumers.json written empty"

    # delete_map_d12.json
    delete_map = {
        "phase": 4,
        "slice": SLICE,
        "checkpoint_sha": CHECKPOINT_SHA,
        "versions": _versions(),
        "result": {
            "candidate": "RollingStdDev inline duplicates",
            "decision": "KEEP",
            "reason": "D12 trading KEEP: BollingerBands and CCI are D8-family inline kernels, NOT duplicates of Compute RollingStdDev primitive. 0 live consumers of Compute RollingStdDev via ISA lowering (SMA/ATR/EMA/RSI do not use it). No deletion.",
            "live_consumers": 0,
            "live_consumers_list": [],
            "trading_keep": ["BollingerBands", "CCI"],
            "trading_family": "D8",
            "audit": "READ-only. No code in Trading modified; Compute RollingStdDev primitive remains for future use but has 0 live ISA consumers in this slice.",
        },
        "evidence_schema": "v1",
    }
    path2 = EVIDENCE_DIR / "delete_map_d12.json"
    path2.write_text(json.dumps(delete_map, indent=2, ensure_ascii=False),
                     encoding="utf-8")
    assert os.path.getsize(path2) > 0, "delete_map_d12.json written empty"
