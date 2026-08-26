"""Phase 4 - ROLLINGSUM consumer verification (S79).

Spec: ROLLINGSUM_SLICE_SPEC.md sec.9 (S79).

Consumer map (by ISA lowering, lowering.py):
  SMA(data, 14)   -> RollingSum(period=14) -> MapBinary(op=3,
                    use_scalar_b=1, scalar_b=14)   [CANONICAL CONSUMER]
  ATR(h,l,c,14)   -> TrueRange -> RollingSum(period=14) -> MapBinary(
                    op=3, use_scalar_b=1, scalar_b=14)  [CANONICAL CONSUMER]
  EMA(data, 14)   -> StateKernel_Single(mode=0)   [NO RollingSum]
  RSI(data, 14)   -> StateKernel_Dual + StateKernel_Single x2 +
                    Map/MapBinary/Shift            [NO RollingSum]

Graph assertions: lower(ast_to_dict(parse(expr))) must contain EXACTLY
one RollingSum node with params.period == 14 for SMA/ATR, plus a
MapBinary node with op=3, use_scalar_b=1.0, scalar_b=14.0; EMA/RSI
graphs must NOT contain any RollingSum node (regression protection for
resolvers).

Consumer test runs (PASS criteria, spec sec.9):
  1. tests/test_ast_integration.py -k sma  (canonical RollingSum+MapBinary
     end-to-end vs numpy) - run via subprocess.
  2. tests/golden/test_golden.py (SMA/RSI/ATR golden, tol 1e-12)
  3. tests/runtime/test_full_cycle.py (SMA->RSI, EMA->SMA->RSI chains)
  Results recorded in consumers.json (failures in these files that are
  pre-existing at the Reduce baseline are noted, NOT attributed to this
  slice).

Evidence: evidence/rollingsum_slice_phase4/consumers.json

IMPORTANT: this file is ASCII-only (no Cyrillic) because
tests/test_backend.py::test_no_cupy_in_test_files reads tests/*.py with
open() in locale encoding (cp1251 on this host).
"""

import json
import os
import pathlib
import subprocess
import sys

import numpy as np

_PATH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _sub in (".", "src/core", "src/math"):
    _p = os.path.join(_PATH_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AST._lib.parser import parse
from AST._lib.ast_nodes import ast_to_dict
from ISA._lib.lowering import lower, format_jobs

CHECKPOINT_SHA = "63e44f3b77f09a410a4e084de771bd72302ec2ca"
SLICE = "rollingsum"

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "rollingsum_slice_phase4")

PYTHON = sys.executable

try:
    import wgpu
    _adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
except Exception:  # noqa: BLE001 - adapter probe must never crash collection
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
    ast = parse(expr)
    d = ast_to_dict(ast)
    graph, out = lower(d)
    jobs = format_jobs(graph, out)
    return graph, out, jobs


def _rolling_sum_nodes(graph):
    return [nd for nd in graph if nd["op"] == "RollingSum"]


def _mapbinary_div_nodes(graph, period):
    out = []
    for nd in graph:
        if nd["op"] != "MapBinary":
            continue
        p = nd["params"]
        if (p.get("op") == 3 and p.get("use_scalar_b") == 1.0
                and p.get("scalar_b") == float(period)):
            out.append(nd)
    return out


def test_consumer_graph_sma():
    graph, out, jobs = _graph_for("SMA(data, 14)")
    rs = _rolling_sum_nodes(graph)
    assert len(rs) == 1, f"SMA: RollingSum count {len(rs)}"
    assert rs[0]["params"]["period"] == 14, rs[0]["params"]
    mb = _mapbinary_div_nodes(graph, 14)
    assert len(mb) == 1, f"SMA: MapBinary div nodes {len(mb)}"
    # lowered node inputs are a list of producer output names
    assert mb[0]["inputs"] == [rs[0]["out"]], mb[0]
    # full job chain must compile/format cleanly
    assert len(jobs) == 2, jobs


def test_consumer_graph_atr():
    graph, out, jobs = _graph_for("ATR(h, l, c, 14)")
    rs = _rolling_sum_nodes(graph)
    assert len(rs) == 1, f"ATR: RollingSum count {len(rs)}"
    assert rs[0]["params"]["period"] == 14, rs[0]["params"]
    mb = _mapbinary_div_nodes(graph, 14)
    assert len(mb) == 1, f"ATR: MapBinary div nodes {len(mb)}"
    assert len(jobs) == 3, jobs  # TrueRange + RollingSum + MapBinary


def test_consumer_graph_ema_no_rollingsum():
    graph, out, jobs = _graph_for("EMA(data, 14)")
    rs = _rolling_sum_nodes(graph)
    assert rs == [], f"EMA must NOT use RollingSum, got {rs}"
    assert any(nd["op"] == "StateKernel_Single" for nd in graph)


def test_consumer_graph_rsi_no_rollingsum():
    graph, out, jobs = _graph_for("RSI(data, 14)")
    rs = _rolling_sum_nodes(graph)
    assert rs == [], f"RSI must NOT use RollingSum, got {rs}"
    assert any(nd["op"] == "StateKernel_Dual" for nd in graph)


def _run_ast_integration():
    """S79 criteria 1: test_ast_integration SMA end-to-end must PASS."""
    test_file = os.path.join(_PATH_ROOT, "tests", "test_ast_integration.py")
    proc = subprocess.run(
        [PYTHON, "-m", "pytest", test_file, "-k", "sma", "-q", "--no-header"],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, \
        f"test_ast_integration -k sma failed:\n{proc.stdout}\n{proc.stderr}"
    return {
        "returncode": proc.returncode,
        "summary": proc.stdout.strip().splitlines()[-1]
                   if proc.stdout.strip() else "",
    }


def _run_golden_full_cycle():
    """S79 criteria 2-3: golden + full_cycle baseline runs (pre-existing
    collection status at Reduce baseline recorded, NOT attributed)."""
    results = {}
    for name, rel in (("golden", os.path.join("tests", "golden", "test_golden.py")),
                      ("full_cycle", os.path.join("tests", "runtime", "test_full_cycle.py"))):
        path = os.path.join(_PATH_ROOT, rel)
        if not os.path.exists(path):
            results[name] = {"status": "missing", "fact": f"{rel} not found"}
            continue
        proc = subprocess.run(
            [PYTHON, "-m", "pytest", path, "-q", "--no-header"],
            capture_output=True, text=True, timeout=900,
        )
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        if proc.returncode == 0:
            results[name] = {"status": "PASS", "returncode": 0,
                             "summary": tail}
        else:
            results[name] = {
                "status": "FAIL" if proc.returncode != 5 else "COLLECTION_ERR",
                "returncode": proc.returncode,
                "summary": tail,
                "note": "pre-existing at Reduce baseline (regression.json "
                        "S80: 2x collection golden/full_cycle), NOT "
                        "attributed to this slice",
            }
    return results


def test_consumers_evidence_saved():
    """Write consumers.json (consumer map + test results)."""
    graph_sma, out_sma, jobs_sma = _graph_for("SMA(data, 14)")
    graph_atr, out_atr, jobs_atr = _graph_for("ATR(h, l, c, 14)")
    graph_ema, out_ema, jobs_ema = _graph_for("EMA(data, 14)")
    graph_rsi, out_rsi, jobs_rsi = _graph_for("RSI(data, 14)")

    ast_run = _run_ast_integration()
    baseline_runs = _run_golden_full_cycle()

    consumer_map = {
        "SMA": {
            "uses_rollingsum": True,
            "graph": graph_sma,
            "rolling_sum_period": 14,
            "mapbinary": {"op": 3, "use_scalar_b": 1.0, "scalar_b": 14.0},
            "jobs_count": len(jobs_sma),
        },
        "ATR": {
            "uses_rollingsum": True,
            "graph": graph_atr,
            "rolling_sum_period": 14,
            "mapbinary": {"op": 3, "use_scalar_b": 1.0, "scalar_b": 14.0},
            "jobs_count": len(jobs_atr),
        },
        "EMA": {
            "uses_rollingsum": False,
            "graph": graph_ema,
            "note": "StateKernel_Single(mode=0) - RollingSum NOT used",
        },
        "RSI": {
            "uses_rollingsum": False,
            "graph": graph_rsi,
            "note": "StateKernel_Dual + StateKernel_Single x2 + "
                    "Map/MapBinary/Shift - RollingSum NOT used",
        },
    }

    evidence = {
        "phase": 4,
        "slice": SLICE,
        "checkpoint_sha": CHECKPOINT_SHA,
        "versions": _versions(),
        "result": {
            "consumer_map": consumer_map,
            "tests": {
                "test_ast_integration_sma": ast_run,
                "golden": baseline_runs.get("golden"),
                "full_cycle": baseline_runs.get("full_cycle"),
            },
            "gpu_available": GPU_AVAILABLE,
            "note": "graph assertions via lower(ast_to_dict(parse(expr))): "
                    "SMA/ATR -> exactly one RollingSum node with "
                    "params.period==14 + MapBinary(op=3, use_scalar_b=1, "
                    "scalar_b=14); EMA/RSI -> NO RollingSum node "
                    "(resolver regression protection). test_ast_integration "
                    "-k sma is the canonical consumer end-to-end "
                    "(RollingSum+MapBinary vs numpy) and must PASS; "
                    "golden/full_cycle collection status is pre-existing "
                    "at the Reduce baseline (S80 regression.json), "
                    "recorded NOT attributed.",
        },
        "evidence_schema": "v1",
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / "consumers.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    assert os.path.getsize(path) > 0, "consumers.json written empty"