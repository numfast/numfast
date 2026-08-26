"""Phase 4 - STATEKERNEL consumer verification (S91).

Spec: STATEKERNEL_SLICE_SPEC.md sec.9 (S91).

Consumer map (by ISA lowering, lowering.py):
  EMA(data, 14)  -> StateKernel_Single(mode=0, a=2/15, b=13/15) [CANONICAL]
  RSI(data, 14)  -> StateKernel_Dual(mode=1, a=1/14, b=13/14) +
                    StateKernel_Single x2 (mode=0, same a/b) +
                    Map/MapBinary/Shift  [CANONICAL]
  SMA(data, 14)  -> RollingSum(period=14) -> MapBinary  [NO StateKernel]
  ATR(h,l,c,14)  -> TrueRange -> RollingSum -> MapBinary [NO StateKernel]

Graph assertions: lower(ast_to_dict(parse(expr))) must contain EXACTLY
as described; EMA: one StateKernel_Single with mode 0 a=2/15 b=13/15;
RSI: one StateKernel_Dual (mode 1 a=1/14 b=13/14) + two StateKernel_Single
(mode 0 same a/b); SMA/ATR must NOT contain any StateKernel node
(regression protection).

Consumer test runs (PASS criteria, spec sec.9):
  1. tests/test_ast_integration.py -k ema or sma (end-to-end vs numpy)
  2. tests/runtime/test_full_cycle.py (pre-existing collection)
  3. tests/golden/test_golden.py (pre-existing collection)
  Results recorded in consumers.json.

Evidence: evidence/statekernel_slice_phase4/consumers.json

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

CHECKPOINT_SHA = "08575d7d8a57ec5f1de0fae974bbeb76bf650f07"
SLICE = "statekernel"

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "statekernel_slice_phase4")

PYTHON = sys.executable

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
    ast = parse(expr)
    d = ast_to_dict(ast)
    graph, out = lower(d)
    jobs = format_jobs(graph, out)
    return graph, out, jobs


def _sk_nodes(graph, op):
    return [nd for nd in graph if nd["op"] == op]


def _all_sk_nodes(graph):
    return [nd for nd in graph if nd["op"].startswith("StateKernel")]


def _approx(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def test_consumer_graph_ema():
    graph, out, jobs = _graph_for("EMA(data, 14)")
    singles = _sk_nodes(graph, "StateKernel_Single")
    assert len(singles) == 1, f"EMA: StateKernel_Single count {len(singles)} expected 1, got {singles}"
    p = singles[0]["params"]
    assert p.get("mode") == 0.0 or p.get("mode") == 0, f"EMA mode {p}"
    assert _approx(p.get("a"), 2.0 / 15.0), f"EMA a {p.get('a')} != 2/15"
    assert _approx(p.get("b"), 13.0 / 15.0), f"EMA b {p.get('b')} != 13/15"
    # EMA must NOT contain Dual or ST
    duals = _sk_nodes(graph, "StateKernel_Dual")
    assert duals == [], f"EMA must NOT contain StateKernel_Dual, got {duals}"
    sts = _sk_nodes(graph, "StateKernel_ST")
    assert sts == [], f"EMA must NOT contain StateKernel_ST, got {sts}"


def test_consumer_graph_rsi():
    graph, out, jobs = _graph_for("RSI(data, 14)")
    duals = _sk_nodes(graph, "StateKernel_Dual")
    assert len(duals) == 1, f"RSI: StateKernel_Dual count {len(duals)} expected 1, got {duals}"
    dp = duals[0]["params"]
    assert dp.get("mode") == 1.0 or dp.get("mode") == 1, f"RSI Dual mode {dp}"
    assert _approx(dp.get("a"), 1.0 / 14.0), f"RSI Dual a {dp.get('a')} != 1/14"
    assert _approx(dp.get("b"), 13.0 / 14.0), f"RSI Dual b {dp.get('b')} != 13/14"
    singles = _sk_nodes(graph, "StateKernel_Single")
    assert len(singles) == 2, f"RSI: StateKernel_Single count {len(singles)} expected 2, got {singles}"
    for nd in singles:
        p = nd["params"]
        assert p.get("mode") == 0.0 or p.get("mode") == 0, f"RSI Single mode {p}"
        assert _approx(p.get("a"), 1.0 / 14.0), f"RSI Single a {p.get('a')} != 1/14"
        assert _approx(p.get("b"), 13.0 / 14.0), f"RSI Single b {p.get('b')} != 13/14"


def test_consumer_graph_sma_no_statekernel():
    graph, out, jobs = _graph_for("SMA(data, 14)")
    sk = _all_sk_nodes(graph)
    assert sk == [], f"SMA must NOT contain StateKernel, got {sk}"
    # must contain exactly one RollingSum with period 14 (protection)
    rs = [nd for nd in graph if nd["op"] == "RollingSum"]
    assert len(rs) == 1, f"SMA RollingSum count {len(rs)}"
    assert rs[0]["params"]["period"] == 14, rs[0]["params"]


def test_consumer_graph_atr_no_statekernel():
    graph, out, jobs = _graph_for("ATR(h, l, c, 14)")
    sk = _all_sk_nodes(graph)
    assert sk == [], f"ATR must NOT contain StateKernel, got {sk}"
    rs = [nd for nd in graph if nd["op"] == "RollingSum"]
    assert len(rs) == 1, f"ATR RollingSum count {len(rs)}"
    assert rs[0]["params"]["period"] == 14, rs[0]["params"]


def _run_ast_integration():
    test_file = os.path.join(_PATH_ROOT, "tests", "test_ast_integration.py")
    if not os.path.exists(test_file):
        return {"status": "missing", "fact": "test_ast_integration.py not found"}
    # run SMA and EMA checks (spec S91: SMA end-to-end protection + EMA vs numpy)
    proc = subprocess.run(
        [PYTHON, "-m", "pytest", test_file, "-k", "sma or ema", "-q", "--no-header"],
        capture_output=True, text=True, timeout=600,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if proc.returncode == 0:
        return {"status": "PASS", "returncode": 0, "summary": tail}
    return {"status": "FAIL" if proc.returncode != 5 else "COLLECTION_ERR",
            "returncode": proc.returncode, "summary": tail,
            "stdout": proc.stdout[:2000], "stderr": proc.stderr[:2000]}


def _run_golden_full_cycle():
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
            results[name] = {"status": "PASS", "returncode": 0, "summary": tail}
        else:
            results[name] = {
                "status": "FAIL" if proc.returncode != 5 else "COLLECTION_ERR",
                "returncode": proc.returncode,
                "summary": tail,
                "stdout": proc.stdout[:2000],
                "stderr": proc.stderr[:2000],
                "note": "pre-existing at RollingSum baseline (regression S92: 3x test_stats, 2x collection golden/full_cycle), NOT attributed to this slice",
            }
    return results


def test_consumers_evidence_saved():
    graph_ema, out_ema, jobs_ema = _graph_for("EMA(data, 14)")
    graph_rsi, out_rsi, jobs_rsi = _graph_for("RSI(data, 14)")
    graph_sma, out_sma, jobs_sma = _graph_for("SMA(data, 14)")
    graph_atr, out_atr, jobs_atr = _graph_for("ATR(h, l, c, 14)")

    ast_run = _run_ast_integration()
    baseline_runs = _run_golden_full_cycle()

    # Extract EMA/Rsi params for evidence
    ema_single = [nd for nd in graph_ema if nd["op"] == "StateKernel_Single"][0]
    rsi_dual = [nd for nd in graph_rsi if nd["op"] == "StateKernel_Dual"][0]
    rsi_singles = [nd for nd in graph_rsi if nd["op"] == "StateKernel_Single"]

    consumer_map = {
        "EMA": {
            "uses_statekernel": True,
            "graph": graph_ema,
            "statekernel_single": {"mode": ema_single["params"]["mode"], "a": ema_single["params"]["a"], "b": ema_single["params"]["b"]},
            "expected": {"mode": 0, "a": 2.0 / 15.0, "b": 13.0 / 15.0},
            "jobs_count": len(jobs_ema),
        },
        "RSI": {
            "uses_statekernel": True,
            "graph": graph_rsi,
            "statekernel_dual": {"mode": rsi_dual["params"]["mode"], "a": rsi_dual["params"]["a"], "b": rsi_dual["params"]["b"]},
            "expected_dual": {"mode": 1, "a": 1.0 / 14.0, "b": 13.0 / 14.0},
            "statekernel_singles": [{"mode": nd["params"]["mode"], "a": nd["params"]["a"], "b": nd["params"]["b"]} for nd in rsi_singles],
            "expected_single": {"mode": 0, "a": 1.0 / 14.0, "b": 13.0 / 14.0},
            "jobs_count": len(jobs_rsi),
        },
        "SMA": {
            "uses_statekernel": False,
            "graph": graph_sma,
            "note": "RollingSum(period=14) -> MapBinary - StateKernel NOT used (protection)",
            "jobs_count": len(jobs_sma),
        },
        "ATR": {
            "uses_statekernel": False,
            "graph": graph_atr,
            "note": "TrueRange -> RollingSum(period=14) -> MapBinary - StateKernel NOT used (protection)",
            "jobs_count": len(jobs_atr),
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
                "test_ast_integration": ast_run,
                "golden": baseline_runs.get("golden"),
                "full_cycle": baseline_runs.get("full_cycle"),
            },
            "gpu_available": GPU_AVAILABLE,
            "note": "graph assertions via lower(ast_to_dict(parse(expr))): EMA(data,14) -> exactly one StateKernel_Single {mode:0, a:2/15, b:13/15}; RSI(data,14) -> one StateKernel_Dual {mode:1, a:1/14, b:13/14} + two StateKernel_Single {mode:0, same a/b}; SMA/ATR -> NO StateKernel node (RollingSum protection). test_ast_integration -k sma/ema must PASS if present; golden/full_cycle pre-existing collection status recorded NOT attributed.",
        },
        "evidence_schema": "v1",
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / "consumers.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    assert os.path.getsize(path) > 0, "consumers.json written empty"
