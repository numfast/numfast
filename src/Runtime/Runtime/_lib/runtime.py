# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Runtime: ExecutionGraph -> Driver, dual-driver dispatch (DELTA-6 + 04R).

SPEC-DELTA-04R: select_backend is called EXACTLY ONCE per ExecutionGraph
here (per-op routing is forbidden). execution_info carries the full
EXPLAIN surface: backend/graph/state/profile/estimated/gpu breakdown/edges.
"""

import numpy as np


def evaluate_impl(graph, backend, n, select_backend, cpu_execute,
                  gpu_execute, chunk_plan, hints=None, profile=None):
    """evaluate(graph, backend, n, hints, profile) -> {result, execution_info}.

    Graph-level chunked GPU execution: when chunk_plan returns num_chunks > 1
    for GPU backend, execute graph in chunks with deterministic merge.
    Contract: backend="gpu" = GPU whole OR GPU chunked. NEVER GPU->CPU fallback.
    Override (P5): cpu->CPU always; gpu->GPU or explicit error; auto->
    capability->coverage->cost (select_backend owns it, called exactly once).
    hints/profile are additive (default warm/host): old callers unaffected.
    """
    if backend not in ("auto", "cpu", "gpu"):
        raise ValueError(
            f"evaluate: unknown backend '{backend}': use auto/cpu/gpu. "
            "Fix: pass backend='auto', 'cpu' or 'gpu'."
        )
    hints = dict(hints or {})
    sel = select_backend(graph, n, profile, hints)  # one decision/graph (04R)
    if backend == "gpu" and not sel.get("gpu_eligible", False):
        missing = ", ".join(sel.get("gpu_blockers", ["?"]))
        raise RuntimeError(
            f"backend='gpu' ineligible: ops not on GPU: {missing}. "
            "Fix: use backend='cpu' (or 'auto') for this graph. "
            "See specs/delta-6-runtime-gpu-groupby.md"
        )
    actual = backend if backend in ("cpu", "gpu") else sel["backend"]
    if backend in ("cpu", "gpu"):
        reason = (f"explicit backend='{backend}'"
                  + ("" if actual == sel["backend"]
                     else f" (planner preferred '{sel['backend']}': "
                          f"{sel['reason']}>)"))
    else:
        reason = sel["reason"]
    res_from = str(hints.get("residence_from",
                             hints.get("residence", "host"))).lower()
    res_from = {"cpu": "host", "gpu": "resident"}.get(res_from, res_from)
    if res_from not in ("host", "resident"):
        res_from = "host"
    res_to = "resident" if actual == "gpu" else "host"
    if res_from == res_to:
        crossings = []
    elif res_to == "resident":
        crossings = ["h2d"]
    else:
        crossings = ["d2h"]
    residence = {"from": res_from, "to": res_to, "crossings": crossings,
                 "ping_pong": False,
                 "note": "single backend per graph: at most one crossing"}

    # Compute per-node chunk plans (VRAM-aware for GPU)
    plans = {nd["out"]: chunk_plan(nd["kernel_id"], n, backend=actual,
                                   graph=graph["nodes"])
             for nd in graph["nodes"]}
    num_chunks = max((p["num_chunks"] for p in plans.values()), default=1)

    if actual == "gpu" and num_chunks > 1:
        # GPU chunked execution path
        chunk_size = min(p.get("chunk_size", n)
                         for p in plans.values() if p["num_chunks"] > 1)
        chunk_size = max(1, int(chunk_size))

        import time as _time
        t0 = _time.perf_counter()
        result = gpu_execute(graph["nodes"], n, chunk_size)
        exec_ms = (_time.perf_counter() - t0) * 1000

        if isinstance(result, np.generic):
            result = result.item()

        est = sel.get("estimate") or {}
        info = {
            "requested": backend,
            "actual": actual,
            "reason": reason + f" (GPU chunked: {num_chunks} chunks)",
            "dispatches": len(graph["nodes"]),
            "h2d": num_chunks,
            "d2h": num_chunks,
            "num_chunks": num_chunks,
            "chunk_size": chunk_size,
            "chunk_plan": plans,
            "cost_estimate": sel["cost_estimate"],
            "estimated_cost": sel["cost_estimate"],
            "profile": sel["profile"],
            "gpu_eligible": sel.get("gpu_eligible"),
            "gpu_blockers": sel.get("gpu_blockers", []),
            "coverage": sel.get("coverage", {"ok": True, "blockers": []}),
            "estimate": est,
            "placement": "gpu_chunked",
            "residence": {**residence, "to": "resident",
                          "note": residence["note"] + "; chunked: H2D/D2H per chunk"},
            "edges": [{"out": nd["out"], "inputs": list(nd["inputs"]),
                       "kernel": nd.get("kernel_id", nd.get("op", "?"))}
                      for nd in graph["nodes"]],
        }
        return {"result": result, "execution_info": info}

    if num_chunks != 1:
        raise RuntimeError(
            f"chunked execution not implemented: chunk_plan={plans}. "
            "Fix: reduce n to fit backend_limit or use a chunkable op path."
        )
    run = gpu_execute if actual == "gpu" else cpu_execute
    bufs = run(graph["nodes"])
    last = graph["outputs"][-1] if graph["outputs"] else graph["nodes"][-1]["out"]
    result = bufs[last]
    if isinstance(result, np.generic):
        result = result.item()
    est = sel.get("estimate") or {}
    info = {
        "requested": backend,
        "actual": actual,
        "reason": reason,
        "dispatches": len(graph["nodes"]),
        "h2d": 1,
        "d2h": 1,
        "num_chunks": 1,
        "chunk_plan": plans,
        "cost_estimate": sel["cost_estimate"],
        "estimated_cost": sel["cost_estimate"],
        "profile": sel["profile"],
        "gpu_eligible": sel.get("gpu_eligible"),
        "gpu_blockers": sel.get("gpu_blockers", []),
        "coverage": sel.get("coverage", {"ok": True, "blockers": []}),
        "estimate": est,
        "placement": "resident" if actual == "gpu" else "host",
        "residence": residence,
        "edges": [{"out": nd["out"], "inputs": list(nd["inputs"]),
                   "kernel": nd.get("kernel_id", nd.get("op", "?"))}
                  for nd in graph["nodes"]],
    }
    return {"result": result, "execution_info": info}
