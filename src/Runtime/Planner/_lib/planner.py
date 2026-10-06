# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Planner: compile jobs -> ExecutionGraph, capability + calibrated STUB (spec 03).

Capability (ops, chunkable hints, dispatch/buffer limits) is read from the
driver capability() — single source, no Planner-side duplicate table.
No real calibration here (nf.calibrate later). Numbers below are documented
placeholders; the FORMULA structure is normative, never hardcoded thresholds.
"""

import json
import math

from _lib.calibrate import build_context as _build_context
from _lib.calibrate import check_coverage as _check_coverage
from _lib.calibrate import check_eligibility as _check_eligibility
from _lib.calibrate import estimate_cost_v1 as _estimate_cost_v1
from _lib.calibrate import estimate_graph as _estimate_graph
from _lib.calibrate import format_explain as _format_explain
from _lib.calibrate import profile_status as _profile_status
from _lib.calibrate import propagate_residence as _propagate_residence
from _lib.calibrate import resolve_routing_profile as _resolve_routing_profile
from _lib.calibrate import routing_reject_warning as _routing_reject_warning

# Calibrated-parameter STUB (shape of calibrated_v1, values are placeholders).
_CALIBRATED_STUB = {
    "a_cpu": 1.0,
    "b_cpu": 0.0,
    "a_gpu": 2.0,
    "b_gpu": 100.0,
    "h2d": 50.0,
    "d2h": 50.0,
    "note": "STUB: no measured profile; structure per spec 03 only",
}


def _fallback_err(what, fix="", doc=""):
    msg = str(what)
    if fix:
        msg += f" Fix: {fix}."
    if doc:
        msg += f" See {doc}"
    return ValueError(msg)


def _ops(capability):
    return set(capability["ops"])


def _chunkable(capability, op):
    return bool(capability.get("chunkable_hints", {}).get(op, False))


def compile_impl(jobs, capability, format_error=None):
    """compile(jobs) -> ExecutionGraph {nodes, inputs, outputs, metadata}."""
    err = format_error or _fallback_err
    nodes = []
    for j in jobs:
        for key in ("op", "inputs", "params", "out"):
            if key not in j:
                raise err(
                    f"job missing '{key}': jobs are {{op,inputs,params,out}}",
                    fix="build jobs via ir_* constructors",
                )
        if j["op"] not in _ops(capability):
            raise err(
                f"unknown op '{j['op']}': capability covers {sorted(_ops(capability))}",
                fix="add a Planner capability entry first for explicit new ops",
            )
        nodes.append({
            "kernel_id": j["op"],
            "inputs": list(j["inputs"]),
            "params": dict(j["params"]),
            "out": j["out"],
        })
    used = {i for n in nodes for i in n["inputs"]}
    outs = [n["out"] for n in nodes]
    return {
        "nodes": nodes,
        "inputs": [n["out"] for n in nodes if not n["inputs"]],
        "outputs": [o for o in outs if o not in used],
        "metadata": {"compiled": True},
    }


def _scalar_key(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return ("scalar", v)
    return None


def _params_fingerprint(params):
    """Structural CSE fingerprint: cheap, no huge-array serialization.

    Scalars -> exact values (CSE still fires on small identical nodes).
    Array-likes (list/tuple/ndarray/arrow) -> (kind, len, dtype, identity):
    identity-scoped, so two different live objects NEVER merge (no false
    CSE), while the same object shared across nodes still dedupes.
    json.dumps(params, default=str) on 10M-element columns cost ~5-18s per
    query (H2O 10M gate) and is never executed again after this change.
    """
    items = []
    for k in sorted(params):
        v = params[k]
        s = _scalar_key(v)
        if s is not None:
            items.append((k, s))
            continue
        if isinstance(v, dict):
            items.append((k, ("dict", _params_fingerprint(v))))
            continue
        if isinstance(v, (list, tuple)):
            if len(v) <= 64 and all(_scalar_key(x) is not None for x in v):
                items.append((k, ("small_seq", tuple(v))))
            else:
                items.append((k, ("big_seq", type(v).__name__, len(v), id(v))))
            continue
        shape, dtype = getattr(v, "shape", None), getattr(v, "dtype", None)
        if shape is not None or dtype is not None:
            items.append((k, ("array", str(shape), str(dtype), id(v))))
            continue
        try:
            items.append((k, ("json", json.dumps(v, sort_keys=True, default=str)[:256])))
        except TypeError:
            items.append((k, ("repr", type(v).__name__, id(v))))
    return tuple(items)


def optimize_impl(graph):
    """Real CSE+DCE pass (pure): dedupe identical nodes, drop dead nodes.

    Honest contract: optimized=True only if the graph changed; otherwise
    optimized=False and passes=[]. No pass-through-stub claims.
    """
    nodes = list(graph["nodes"])
    passes = []
    # CSE: identical (kernel_id, inputs, params) with different outs -> rewire.
    seen, rewired, deduped = {}, {}, []
    for n in nodes:
        key = (n["kernel_id"], tuple(n["inputs"]), _params_fingerprint(n["params"]))
        if key in seen:
            rewired[n["out"]] = seen[key]
            if "cse" not in passes:
                passes.append("cse")
        else:
            seen[key] = n["out"]
            deduped.append(n)
    if rewired:
        for n in deduped:
            n["inputs"] = [rewired.get(i, i) for i in n["inputs"]]
    outs = [rewired.get(o, o) for o in graph["outputs"]]
    for n in deduped:
        if n["out"] in rewired:
            n["out"] = rewired[n["out"]]
    # DCE: keep only nodes reaching outputs.
    live, changed = set(outs), True
    while changed:
        changed = False
        for n in deduped:
            if n["out"] in live:
                for i in n["inputs"]:
                    if i not in live:
                        live.add(i)
                        changed = True
    kept = [n for n in deduped if n["out"] in live]
    if len(kept) < len(deduped):
        passes.append("dce")
    by_out = {}
    for n in kept:
        by_out.setdefault(n["out"], n)
    kept = list(by_out.values())
    return {
        "nodes": kept,
        "inputs": [n["out"] for n in kept if not n["inputs"]],
        "outputs": outs,
        "metadata": {
            **graph.get("metadata", {}),
            "optimized": bool(passes),
            "passes": passes,
        },
    }


# Group-index strategy profile (single source of numbers; STRUCTURE is the
# cost model, VALUES are one measured profile -- see approved 10M matrix):
# dense direct wins while its backing arrays stay small; hash is the generic
# fallback only where dense is unsafe (never assumed a universal winner:
# naive open-addressing lost high-card uniform 9.0s vs 4.1s).
_GROUPINDEX_PROFILE = {
    "r_max": 64_000_000,      # hard cap on dense span R (backing arrays ~16*R)
    "k_rn": 4.0,              # dense only if R <= k_rn * N (range-vs-rows gate;
                              # rejects [0, 1e9]-style traps even when R < r_max)
    "hash_init": 1 << 24,     # hash start capacity at production scale
    "hash_max_load": 0.7,     # 2x rehash above this load
    "hash_growth": 2,
    "note": "groupindex profile (10M matrix); values live here, never inline",
}


def plan_groupby_impl(n, is_sorted, nunique_hint=None, *, span=None,
                      est_dense_bytes=None, budget_bytes=None, kmin=None,
                      est_code_bytes=None, naggs=1):
    """Grouping strategy choice (Planner owns the decision, CPU executes).

    Legacy observables only (n, is_sorted) -> 'sorted' | 'unique' | 'empty',
    exactly as before (no behaviour change for old callers).
    With range observables (span=R, est_dense_bytes, budget_bytes) the
    memory-aware cost gate applies; with kmin/est_code_bytes/naggs the dense
    family splits generically (all arithmetic on Python ints, no overflow):
      total_shift = 16*R+8*N + naggs*8*min(R,N)   (shift array + lut +
                    inverse + one worst-case state array per aggregate)
      total_code  = 16*M+... with M=kmax+1 (sums+counts M-aux + one shared
                    counts + one state array per aggregate, NO shift array,
                    NO lut, NO inverse)
      dense_by_code iff kmin>=0, M<=cap, total_code<=budget and
      total_code<=total_shift; else shift-based dense FALLBACK iff
      R<=cap and total_shift<=budget (reason cites both totals).
    Otherwise hash (generic, no giant allocs) with the rejection observable;
    unique stays the CPU-side last-resort fallback (hash alloc failure).
    Pure function of observables; probes (sortedness, min/max) run in CPU on
    vectorized paths, never here.
    """
    if n == 0:
        return {"strategy": "empty", "reason": "no rows"}
    if is_sorted:
        return {"strategy": "sorted", "reason": "sorted input: reduceat over runs"}
    if span is None or est_dense_bytes is None or budget_bytes is None:
        return {"strategy": "unique",
                "reason": "unsorted input: unique+inverse+bincount, sorted output"}
    p = _GROUPINDEX_PROFILE
    naggs = max(1, int(naggs))
    cap = min(p["r_max"], p["k_rn"] * n)
    # Per-aggregate worst case: one state array over groups (8*min(R,N)) plus
    # one accumulator-width copy of values (8*N, int64/float64 exactness).
    # Both dense variants pay it equally; it counts toward the budget gate.
    per_col = 8 * min(int(span), int(n)) + 8 * int(n)
    total_shift = int(est_dense_bytes) + naggs * per_col
    if kmin is not None and kmin >= 0 and est_code_bytes is not None:
        m_code = int(kmin) + int(span)
        total_code = int(est_code_bytes) + naggs * 8 * min(m_code, int(n))
        if (m_code <= cap and total_code <= budget_bytes
                and total_code <= total_shift):
            return {"strategy": "dense_by_code",
                    "reason": f"R={span},M={m_code}<=min({p['r_max']},{p['k_rn']}*N="
                              f"{cap:.0f}) and total={total_code / 1e6:.1f}MB<="
                              f"shift={total_shift / 1e6:.1f}MB<=budget="
                              f"{budget_bytes / 1e6:.1f}MB, naggs={naggs}: "
                              "direct by resident code, no shift/lut/inverse"}
    if span <= cap and total_shift <= budget_bytes:
        return {"strategy": "dense",
                "reason": f"R={span}<=min({p['r_max']},{p['k_rn']}*N={cap:.0f}) "
                          f"and total={total_shift / 1e6:.1f}MB<=budget="
                          f"{budget_bytes / 1e6:.1f}MB, naggs={naggs}: "
                          "shift-dense fallback (by-code gate failed)"}
    why = []
    if span > cap:
        why.append(f"R={span}>min({p['r_max']},{p['k_rn']}*N={cap:.0f})")
    if total_shift > budget_bytes:
        why.append(f"total={total_shift / 1e6:.1f}MB>budget={budget_bytes / 1e6:.1f}MB")
    return {"strategy": "hash",
            "reason": "dense rejected (" + "; ".join(why) + "): generic integer "
                      "hash, no giant allocs; unique is last-resort fallback"}


# Composite-pack strategy profile (single source; STRUCTURE is the cost
# model, VALUES are one measured profile -- see prep_q2q5 10M matrix):
# int32-direct radix wins while the composite fits int32 (one alloc, no
# casts, 4.06x vs int64 radix on 10M); legacy int64 radix is the generic
# fallback only where int32 is unsafe (never assumed slower: bitpack int64
# keeps stable cross-width labels where radix does not apply).
_PACK_I32_MAX = 2 ** 31 - 1


def plan_pack_impl(kmin1, kmax1, kmin2, kmax2, m2=None):
    """Composite-pack strategy choice (Planner owns the decision, CPU executes).

    Generic observables only (column minima/maxima + radix multiplier M2;
    no dataset names, no hardcoded cardinalities): mixed-radix composite
    c0*M2+c1 stays in int32 iff kmin1>=0, kmin2>=0 and
    kmax1*M2+kmax2 <= 2**31-1 (all arithmetic on Python ints, no overflow).
    m2=None -> auto M2=kmax2+1 (same definition as the radix auto path).
    int32_direct: one int32 alloc, no casts, bit-identical to int64 radix.
    Otherwise legacy int64 radix verbatim (stable labels, any range).
    Pure function of observables; probes (min/max) run in CPU on vectorized
    paths, never here.
    """
    kmin1, kmax1, kmin2, kmax2 = int(kmin1), int(kmax1), int(kmin2), int(kmax2)
    m2 = int(kmax2) + 1 if m2 is None else int(m2)
    if m2 < 1:
        raise _fallback_err(
            f"plan_pack needs M2>=1, got {m2}",
            fix="pass radix=None for auto or positive ints per input",
        )
    if kmin1 < 0 or kmin2 < 0:
        return {"strategy": "radix",
                "reason": f"kmin=({kmin1},{kmin2})<0: int32 radix needs "
                          "non-negative codes; legacy int64 radix fallback"}
    bound = kmax1 * m2 + kmax2
    if bound <= _PACK_I32_MAX:
        return {"strategy": "int32_direct",
                "reason": f"K1max*M2+K2max={kmax1}*{m2}+{kmax2}={bound}"
                          f"<2**31: one int32 alloc, no casts, "
                          "bit-identical to int64 radix"}
    return {"strategy": "radix",
            "reason": f"K1max*M2+K2max={kmax1}*{m2}+{kmax2}={bound}"
                      f">=2**31: int32 would wrap; legacy int64 radix fallback"}


# Filter compact strategy profile (single source; STRUCTURE is the cost
# model, VALUES are one measured profile): block-partials (GPU counts + host
# W-prefix + GPU
# scatter) wins once dispatches amortize; host-assisted (full-mask D2H +
# host prefix + GPU gather) only below the crossover.
_FILTER_HOST_MAX = 2048


def plan_filter_impl(n, selectivity=None):
    """Filter compact strategy choice (Planner owns the decision, GPU executes).

    Pure function of observables (n rows; selectivity hint unused for the
    choice -- block-partials traffic is selectivity-proportional either way,
    only the D2H tail varies). Returns strategy + measured reason.
    n<=_FILTER_HOST_MAX -> host-assisted (dispatch overhead dominates);
    else gpu_blocks (mask resident, only W=ceil(n/256) counts cross).
    """
    n = int(n)
    if n <= _FILTER_HOST_MAX:
        return {"strategy": "host",
                "reason": f"n={n}<=host_max={_FILTER_HOST_MAX}: one gather "
                          "dispatch beats counts+prefix+scatter "
                          "(bench_gpu_filter.json crossover)"}
    w = (n + 255) // 256
    return {"strategy": "gpu_blocks",
            "reason": f"n={n}: block-partials compact, mask resident, "
                      f"W={w} counts cross (bench_gpu_filter.json)"}


def select_backend_impl(graph, n, capability, profile=None):
    """min(cpu_total, gpu_total) + observable why; skeleton always cpu."""
    p = _CALIBRATED_STUB
    cpu = p["a_cpu"] * n + p["b_cpu"]
    gpu = p["a_gpu"] * n + p["b_gpu"] + p["h2d"] + p["d2h"]
    gpu_ok = capability.get("gpu", False) and all(
        n["kernel_id"] in _ops(capability) for n in graph["nodes"]
    )
    backend = "gpu" if (gpu_ok and gpu < cpu) else "cpu"
    return {
        "backend": backend,
        "reason": "skeleton: gpu capability absent for ops, cpu-oracle only"
        if backend == "cpu" else "stub cost model prefers gpu",
        "cost_estimate": {"cpu": cpu, "gpu": gpu},
        "profile": {
            "version": "stub",
            "source": "none",
            "age": "n/a",
            "matched": False,
            "warning": "no measured calibration profile; stub costs (spec 03)",
        },
    }


def select_backend_dual_impl(graph, n, cpu_capability, gpu_capability,
                             profile=None, hints=None):
    """Dual-driver select: gates first, measured cost second (SPEC-DELTA-03C/04R).

    Exactly one call per ExecutionGraph (Runtime.evaluate calls once; per-op
    routing is forbidden). Order: (1) hard eligibility gates (structure only:
    ops-subset, f64 guard, memory, dispatch); (2) profile
    resolution (explicit dict wins, else measured file, else stub with
    warning); (3) coverage gate (unmeasured op class -> cpu, observable);
    (4) min(cpu_total, gpu_host_total) over MEASURED fits. Selectivity only
    sizes downstream n_op/D2H tail via hints, never gates or routes.
    """
    hints = hints or {}
    sel = float(hints.get("selectivity", 0.5))
    state = hints.get("device_state", hints.get("state", "warm"))
    gate = _check_eligibility(graph, n, cpu_capability, gpu_capability)
    if not gate["eligible"]:
        return {
            "backend": "cpu",
            "reason": ("gpu ineligible ("
                       + ", ".join(gate["blockers"]) + "); cpu-oracle fallback"
                       + ("; " + "; ".join(gate["notes"]) if gate["notes"] else "")),
            "cost_estimate": {"cpu": None, "gpu": None},
            "profile": {"version": "stub", "source": "none", "age": "n/a",
                        "matched": False,
                        "warning": "no measured costs shown without a profile"},
            "gpu_eligible": False,
            "gpu_blockers": gate["blockers"],
            "coverage": {"ok": True, "blockers": []},
            "estimate": None,
            "context": _build_context(graph, n, hints),
        }
    if isinstance(profile, dict):
        prof, _decision = profile, None
    else:
        prof, _decision = _resolve_routing_profile(
            gpu_capability=gpu_capability)
    if prof is None:
        return _stub_select(graph, n, gpu_capability, gate, hints,
                            decision=_decision)
    cov = _check_coverage(graph, prof, hints)
    status = _profile_status(prof)
    ctx = _build_context(graph, n, hints, prof)
    if not cov["ok"]:
        return {
            "backend": "cpu",
            "reason": ("calibrated routing has no measured cost for "
                       + ", ".join(cov["blockers"]) + "; cpu observable "
                       "(explicit backend='gpu' still executes)"),
            "cost_estimate": {"cpu": None, "gpu": None},
            "profile": status,
            "gpu_eligible": True,
            "gpu_blockers": [],
            "coverage": cov,
            "estimate": None,
            "context": ctx,
        }
    est = _estimate_cost_v1(graph, n, prof, ctx)
    cpu, gpu = est["cpu_ms"], est["gpu_host_ms"]
    if gpu is None:
        backend = "cpu"
        reason = (f"measured {status['version']} ({est['state']}): "
                  f"gpu cost unknown ({', '.join(est['unknown_slots'])} "
                  "unmeasured); safe CPU choice "
                  f"(cpu={cpu:.3f}ms; selectivity={est['selectivity']})")
    else:
        backend = "gpu" if gpu < cpu else "cpu"
        reason = (f"measured {status['version']} ({est['state']}): "
                  f"cpu={cpu:.3f}ms vs gpu_host={gpu:.3f}ms "
                  f"(selectivity={est['selectivity']}); resident only removes round trips")
    return {
        "backend": backend,
        "reason": reason,
        "cost_estimate": {"cpu": cpu, "gpu": gpu},
        "profile": status,
        "gpu_eligible": True,
        "gpu_blockers": [],
        "coverage": cov,
        "estimate": est,
        "context": ctx,
    }


def _stub_select(graph, n, gpu_capability, gate, hints=None, decision=None):
    """Pre-calibration path (P9): safe CPU, no fabricated routing.

    Stub placeholders are documented structure only; without a measured
    profile GPU costs are unknown, so auto never prefers GPU here.
    Explicit backend='gpu' still executes (override path, Runtime owns it).

    `decision` is the routing-profile verdict when there was one to reject
    (see resolve_routing_profile): the stub must then name WHY it is on stub,
    because a profile that exists but describes another machine is a different
    situation from no profile at all, and only one of them is the user's to fix.
    """
    gops = _ops(gpu_capability)
    blockers = sorted({nd["kernel_id"] for nd in graph["nodes"]} - gops)
    eligible = not blockers
    if eligible:
        reason = ("no measured calibration profile (unknown costs): safe "
                  "CPU choice; calibrate to enable GPU routing")
    else:
        reason = ("gpu ineligible (ops not on GPU: "
                  + ", ".join(blockers) + "); cpu-oracle fallback")
    return {
        "backend": "cpu",
        "reason": reason,
        "cost_estimate": {"cpu": None, "gpu": None},
        "profile": {
            "version": "stub",
            "source": (decision or {}).get("origin", "none"),
            "age": "n/a",
            "matched": False,
            "warning": (_routing_reject_warning(decision) if decision else
                        "no measured calibration profile; stub costs "
                        "(spec 03)"),
        },
        "gpu_eligible": eligible,
        "gpu_blockers": blockers,
        "coverage": {"ok": True, "blockers": []},
        "estimate": None,
        "context": _build_context(graph, n, hints),
    }


def chunk_plan_impl(op, n, backend_limit=None, capability=None, itemsize=4,
                    graph=None, vram_budget=None):
    """VRAM-aware chunk plan: chunk_size from budget + graph memory footprint.

    backend_limit is the fallback (max_buffer_bytes // itemsize).
    vram_budget = (total_vram, available_budget) from gpu_vram_budget().
    graph: list of nodes for memory footprint estimation.
    Returns {op, n, num_chunks, chunk_size, reason}.
    """
    n = int(n)
    if backend_limit is None:
        if capability is None:
            raise _fallback_err(
                "chunk_plan needs backend_limit or driver capability",
                fix="pass backend_limit or wire capability()",
            )
        backend_limit = max(1, int(capability["max_buffer_bytes"] // itemsize))

    # VRAM-aware chunk size: compute from budget + graph footprint
    if vram_budget is not None and graph is not None:
        total_vram, budget = vram_budget
        # Estimate graph memory: source series nodes + 2x overhead for intermediates
        n_series = sum(1 for nd in graph
                       if nd.get("kernel_id", nd.get("op")) == "series")
        # Per-row bytes: each int32 source = 4B, each intermediate ~4B
        per_row = n_series * itemsize * 3  # 3x: source + intermediates + output
        chunk_rows = max(_MIN_CHUNK_ROWS, budget // max(1, per_row))
        chunk_rows = min(chunk_rows, n)  # don't exceed total rows
        num_chunks = max(1, (n + chunk_rows - 1) // chunk_rows)
        return {
            "op": op, "n": n, "num_chunks": num_chunks,
            "chunk_size": int(chunk_rows),
            "reason": (f"VRAM-aware: budget={budget / 1e6:.0f}MB, "
                       f"per_row={per_row}B, chunk={chunk_rows} rows, "
                       f"{num_chunks} chunks"),
        }

    # Fallback: use backend_limit (existing behavior)
    if n <= backend_limit:
        return {"op": op, "n": n, "num_chunks": 1,
                "chunk_size": n,
                "reason": "fits backend_limit"}
    chunkable = _chunkable(capability, op) if capability is not None else False
    if chunkable:
        cs = backend_limit
        nc = int(math.ceil(n / cs))
        return {
            "op": op, "n": n, "num_chunks": nc, "chunk_size": int(cs),
            "reason": "chunkable: split to fit backend_limit",
        }
    return {"op": op, "n": n, "num_chunks": 1, "chunk_size": n,
            "fallback": "cpu-fallback observable"}


_MIN_CHUNK_ROWS = 1 << 18  # 256K minimum chunk (amortize dispatch overhead)


def explain_impl(graph, execution_info=None):
    """EXPLAIN without exec: backend/graph/state/profile/estimated/gpu/edges."""
    info = dict(execution_info or {})
    if "estimate" not in info and "cost_estimate" in info:
        ce = info.get("cost_estimate") or {}
        info = {**info, "estimate": {
            "cpu_ms": ce.get("cpu"), "gpu_host_ms": ce.get("gpu"),
            "breakdown": [], "state": "n/a", "selectivity": "n/a"}}
    if "profile" not in info:
        info["profile"] = {"version": "n/a", "source": "n/a", "age": "n/a",
                           "matched": "n/a", "warning": ""}
    try:
        return _format_explain(graph, info)
    except (KeyError, TypeError, ValueError):
        lines = ["EXPLAIN (skeleton, no exec):"]
        for n in graph["nodes"]:
            lines.append(f"  {n['kernel_id']} out={n['out']} inputs={n['inputs']} params={n['params']}")
        lines.append(f"inputs={graph['inputs']} outputs={graph['outputs']}")
        return "\n".join(lines)
