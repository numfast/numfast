# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Calibration: measured-only cost model (SPEC-DELTA-03C/04R/06E).

Single source of measured numbers for routing. Rules (spec 03 normative):
- Every numeric cost parameter comes from nf.calibrate() measurements
  (seed 42, synthetic, stage breakdown cold vs warm). No physics literals
  (bandwidths, dispatch us, compile ms, fusion factors) anywhere here.
- Fitted (a, b) per op per backend ABSORB transfers+dispatch+compile:
  warm fits absorb steady-state dispatch; cold fits absorb compile miss.
  Spec-03 formula terms stay STRUCTURAL (sums + tail), numbers stay MEASURED.
- Resident placement is measured evidence (multi-node chain timings vs
  host composition), not a formula: prediction is a conservative upper
  bound for multi-node GPU graphs (resident can only remove round trips).
- Selectivity sizes downstream n_op and the D2H tail ONLY. It never gates
  eligibility and never routes: no N/selectivity thresholds in this file.
- Graphs with ops outside the measured matrix take the coverage gate
  (cpu, observable reason). Explicit backend='gpu' still executes them.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "calibrated_v1"
SEED = 42
MEASURED_OPS = ("compare", "mask", "filter", "gather", "reduce",
                "rng_fill_i32")
EXTENDED_OPS = ("groupby", "sort", "slice", "pack_keys")
ALL_OPS = MEASURED_OPS + EXTENDED_OPS
MATRIX_FULL_NS = (1000, 10000, 100000, 1000000)
MATRIX_QUICK_NS = (10000, 100000)
VALIDATION_NS = (500000,)
SELECTIVITIES = (0.1, 0.5, 0.9)
FIT_SELECTIVITY = 0.5
DEFAULT_SELECTIVITY = 0.5  # downstream-sizing assumption, NOT a routing rule
PROFILE_NAME = "calibration.toml"
DATASET_NAME = "calibration_dataset.json"
_WORKGROUP = 256
GROUPBY_CARDINALITIES = (100, 10_000, 100_000)
GROUPBY_NS = (10_000, 100_000, 1_000_000)
GROUPBY_VALIDATION = {"N": 500_000, "M": 50_000}
SORT_NS = (8192, 65536, 1048576, 4194304)  # all power-of-two for GPU sort
SLICE_NS = (100, 10_000, 100_000)
PACK_NS = (10_000, 100_000, 1_000_000)
TRANSFER_BYTES = (1 << 20, 10 << 20, 100 << 20, 1 << 30)  # 1MB/10MB/100MB/1GB (Pass 5 grid)
TRANSFER_H2D_FIT_MAX_BYTES = 100 << 20  # H2D superlinear at 1GB (measured):
# H2D fit covers <=100MB only, 1GB is holdout validation; D2H linear to 1GB.


def _fork_root():
    # The fork root is the nearest ancestor holding full.toml. It sits at
    # parents[4] in the checkout (src/Runtime/Planner/_lib/) but at parents[3]
    # in the wheel, where this file is vendored to numfast/_ext/Planner/_lib/.
    # A fixed index is right at one vendoring depth and wrong at the other, and
    # there the miss is silent: load_profile() returns None and the Planner
    # routes on stub costs. Walk the ancestors instead -- correct at any depth.
    here = Path(__file__).resolve()
    for p in (*here.parents, Path.cwd()):
        try:
            if (p / "full.toml").exists():
                return p
        except OSError:
            continue
    return here.parents[4]


# The GPU driver module this bench loads by file path is the same Extension in
# both vendoring layouts, at two different places under the fork root:
# src/Drivers/GPU/_lib/gpu.py in the checkout, and _ext/GPU/_lib/gpu.py in the
# wheel, where setup.py copies every [[extensions]] directory into
# numfast/_ext/. Both are spelled out and probed -- _fork_root()/"src"/...
# resolves only in the checkout, so a miss here used to reach exec_module as an
# opaque FileNotFoundError naming a path that was never valid in a wheel.
_GPU_MODULE_RELPATHS = (
    ("src", "Drivers", "GPU", "_lib", "gpu.py"),   # checkout
    ("_ext", "GPU", "_lib", "gpu.py"),              # wheel (setup.py)
)


def _gpu_module_path():
    """Absolute path of Drivers/GPU/_lib/gpu.py at either vendoring depth."""
    tried = []
    for root in (_fork_root(), Path.cwd()):
        for rel in _GPU_MODULE_RELPATHS:
            cand = root.joinpath(*rel)
            tried.append(cand)
            if cand.is_file():
                return cand
    raise FileNotFoundError(
        "calibrate: the GPU driver module gpu.py is not present at any known "
        "vendoring depth, so the H2D/D2H sweep cannot run. Probed: "
        + "; ".join(str(t) for t in tried)
        + ". Ship the GPU Extension (full.toml [[extensions]]) or build from a "
        "tree that has it.")


def profile_path(name=PROFILE_NAME):
    d = os.environ.get("NUMFAST_CALIBRATION_DIR")
    if d:
        return str(Path(d) / name)
    return str(_fork_root() / name)


def dataset_path(name=DATASET_NAME):
    d = os.environ.get("NUMFAST_CALIBRATION_DIR")
    if d:
        return str(Path(d) / name)
    return str(_fork_root() / name)


def fit_line(xs, ys):
    """Least-squares (a, b) for ms = a*n + b. Pure, no constants."""
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return 0.0, float(ys[0])
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0.0:
        return 0.0, my
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return a, my - a * mx


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


_PROFILE_MEMO = {}  # {path: (st_mtime_ns, st_size, parsed)} -- file-content memo
_PROFILE_MEMO_MAX = 8


def load_profile(path=None):
    """Read a measured profile. Returns dict or None (never raises).

    The bytes are read and TOML-parsed only when the file stamp changes:
    each call os.stat()s the path (~10us) and reuses the parsed tables
    while (st_mtime_ns, st_size) is unchanged. st_mtime_ns is 100 ns on
    NTFS, so any real rewrite invalidates even same-second/same-size
    edits; no consumer mutates the tables (read-only by grep). Per-call
    volatile keys (_path, _age_s) are overlaid on a fresh top-level dict,
    so the returned value equals an uncached read.
    """
    p = path or profile_path()
    try:
        st = os.stat(p)
    except OSError:
        return None
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _PROFILE_MEMO.get(p)
    if hit is not None and hit[0] == stamp[0] and hit[1] == stamp[1]:
        parsed = hit[2]                    # validated when it entered the memo
    else:
        try:
            parsed = _toml_loads(Path(p).read_bytes().decode("utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(parsed, dict) or parsed.get("model_version") != SCHEMA:
            return None
        cost = parsed.get("cost", {})
        if not isinstance(cost, dict):
            return None
        if len(_PROFILE_MEMO) >= _PROFILE_MEMO_MAX:
            _PROFILE_MEMO.clear()           # bounded; paths are few
        _PROFILE_MEMO[p] = (stamp[0], stamp[1], parsed)
    prof = dict(parsed)
    try:
        from datetime import datetime as _dt
        gen = _dt.fromisoformat(str(prof.get("generated", "")))
        age_s = (_dt.now(timezone.utc) - gen).total_seconds()
    except (ValueError, TypeError):
        age_s = -1.0
    prof["_path"] = p
    prof["_age_s"] = age_s
    return prof


def _toml_loads(text):
    """Minimal TOML reader for the calibration schema (tables + scalars)."""
    root = {}
    cur = root
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = root
            for part in line[1:-1].strip().split("."):
                cur = cur.setdefault(part.strip(), {})
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        cur[k.strip()] = _toml_value(v.strip())
    return root


def _toml_value(v):
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    if v.startswith("[") and v.endswith("]"):
        return [_toml_value(x.strip()) for x in v[1:-1].split(",") if x.strip()]
    for cast in (int, float):
        try:
            return cast(v)
        except (ValueError, TypeError):
            pass
    if v in ("true", "false"):
        return v == "true"
    return v


def _toml_dump(profile):
    """Minimal TOML writer for the calibration schema (measured values only)."""
    out = ["# NumFast calibration profile -- ALL numbers measured (seed 42).",
           "# Regenerate: alias['calibrate'](force=True). Never hand-edit.",
           f'model_version = "{profile["model_version"]}"',
           f'profile_version = {profile.get("profile_version", 1)}',
           f'generated = "{profile["generated"]}"',
           f'source = "{profile["source"]}"',
           f'quick = {"true" if profile.get("quick") else "false"}',
           ""]
    hw = profile.get("hardware", {})
    out.append("[hardware]")
    for k in ("cpu", "platform", "python", "backend", "gpu_device",
              "gpu_backend", "vram_mb", "vram_source"):
        v = hw.get(k, "unknown")
        out.append(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}")
    out.append("")
    out.append("[cost]")
    for k in sorted(profile.get("cost", {})):
        v = profile["cost"][k]
        if isinstance(v, list):
            out.append(f"{k} = {v!r}")
        else:
            out.append(f"{k} = {v!r}")
    out.append("")
    out.append("[measurements]")
    m = profile.get("measurements", {})
    for k in ("Ns", "quick", "validation_Ns", "reps", "warmup",
              "groupby_cardinalities", "groupby_Ns", "sort_Ns",
              "slice_Ns", "pack_Ns", "transfer_bytes"):
        if k in m:
            out.append(f"{k} = {m[k]!r}")
    gv = m.get("groupby_validation")
    if gv:
        out.append(f"groupby_validation_N = {gv['N']}")
        out.append(f"groupby_validation_M = {gv['M']}")
    out.append("")
    metrics = profile.get("metrics")
    if metrics:
        out.append("[metrics]")
        for k in sorted(metrics):
            out.append(f"{k} = {metrics[k]!r}")
        out.append("")
    return "\n".join(out) + "\n"


def profile_status(prof, current_device="unknown"):
    """EXPLAIN-grade profile descriptor (no numbers invented)."""
    if prof is None:
        return {"version": "stub", "source": "none", "age": "n/a",
                "matched": False,
                "warning": "no measured calibration profile; stub costs"}
    dev = prof.get("hardware", {}).get("gpu_device", "unknown")
    if current_device != "unknown" and dev != "unknown" \
            and current_device != dev:
        warn = (f"device changed ({dev} -> {current_device}); "
                "recalibrate, costs are stale")
        matched = False
    else:
        warn = ""
        matched = True
    age = prof.get("_age_s", -1.0)
    return {"version": prof.get("model_version", SCHEMA),
            "source": prof.get("source", "measured"),
            "age": f"{age:.0f}s" if age >= 0 else "n/a",
            "matched": matched, "warning": warn,
            "path": prof.get("_path", "")}


def _node_n_in(node, n_by_out, n, selectivity):
    ins = node.get("inputs", [])
    if not ins:
        return n
    vals = [n_by_out.get(i, n) for i in ins]
    return max(vals) if vals else n


def downstream_rows(graph, n, selectivity=DEFAULT_SELECTIVITY):
    """Per-node (n_in, n_out). Selectivity ONLY resizes filter output."""
    n_by_out, rows = {}, {}
    for node in graph.get("nodes", []):
        op = node.get("kernel_id", node.get("op", "?"))
        n_in = _node_n_in(node, n_by_out, n, selectivity)
        if op == "filter":
            n_out = int(round(n_in * selectivity))
        elif op == "reduce":
            n_out = 1
        else:
            n_out = n_in
        n_by_out[node.get("out", "")] = n_out
        rows[node.get("out", "")] = (op, n_in, n_out)
    return rows


def check_eligibility(graph, n, cpu_cap, gpu_cap):
    """Hard gates BEFORE any cost math. Structure only, no thresholds.

    Order: ops-subset -> dtype (f64-unscaled) -> memory -> dispatch.
    First failure wins, all observable in blockers/notes. Backend-internal
    rules (sort pow2, negative groupby keys) stay with the DRIVER at exec:
    select never duplicates them.
    """
    gops = set(gpu_cap.get("ops", []))
    kinds = [(nd.get("kernel_id", nd.get("op", "?")), nd) for nd in graph.get("nodes", [])]
    missing = sorted({k for k, _ in kinds} - gops)
    if missing:
        return {"eligible": False, "blockers": [f"op:{m}" for m in missing],
                "notes": ["ops not in gpu capability; cpu-oracle fallback"]}
    for k, nd in kinds:
        if k == "series":
            p = nd.get("params", {})
            if p.get("dtype") == "float64" and "scale" not in p:
                return {"eligible": False, "blockers": ["dtype:float64-unscaled"],
                        "notes": ["MapF64 guard: unscaled f64 needs CPU/float32"]}
    n = int(n)
    n_series = sum(1 for k, _ in kinds if k == "series")
    n_state = sum(1 for k, _ in kinds
                  if k not in ("series", "reduce"))
    peak = n * 4 * (n_series + n_state)
    max_bytes = int(gpu_cap.get("max_buffer_bytes", 2 ** 31 - 1))
    notes = []
    if peak > max_bytes:
        hints = gpu_cap.get("chunkable_hints", {})
        if all(bool(hints.get(k, False)) for k, _ in kinds if k != "series"):
            notes.append(f"peak~{peak}B>limit: all chunkable, chunk_plan owns it")
        else:
            return {"eligible": False,
                    "blockers": ["memory:peak-exceeds-backend-limit"],
                    "notes": [f"peak~{peak}B>max_buffer_bytes={max_bytes}; "
                              "non-chunkable op present"]}
    mx = gpu_cap.get("max_dispatch", {}).get("x", 65535)
    if (n + _WORKGROUP - 1) // _WORKGROUP > int(mx):
        hints = gpu_cap.get("chunkable_hints", {})
        if all(bool(hints.get(k, False)) for k, _ in kinds if k != "series"):
            notes.append("dispatch exceeds backend x: chunkable, chunk_plan owns it")
        else:
            return {"eligible": False,
                    "blockers": ["dispatch:exceeds-backend-x"],
                    "notes": ["workgroups exceed backend max_dispatch.x; "
                              "non-chunkable op present"]}
    return {"eligible": True, "blockers": [], "notes": notes}


def check_coverage(graph, profile, hints=None):
    """Calibrated-cost coverage: every non-series op must be measured.

    GroupBy is cardinality-conservative (Auto Planner v1, P9): the fit used
    is nearest-measured, so a cardinality estimate must exist (node param
    or hints cardinality/groupby_cardinality); without it the gate fails
    to the observable CPU fallback. An estimate outside the measured span
    is extrapolation, not coverage -> CPU fallback as well. Explicit
    backend='gpu' still executes (coverage gates auto only).
    """
    hints = dict(hints or {})
    cost = profile.get("cost", {}) if profile else {}
    missing = set()
    hint_card = hints.get("cardinality", hints.get("groupby_cardinality"))
    for nd in graph.get("nodes", []):
        op = nd.get("kernel_id", nd.get("op", "?"))
        if op == "series":
            continue
        if op == "groupby":
            # groupby needs at least one cardinality measured; groupby_multi
            # has no dedicated fits -> stays uncovered (conservative CPU).
            cardinals = cost.get("groupby_cardinalities", [])
            if not cardinals or not any(
                    f"groupby_M{M}_gpu_a" in cost for M in cardinals):
                missing.add(op)
                continue
            est = nd.get("params", {}).get("cardinality", hint_card)
            if est is None:
                missing.add(f"{op}(cardinality-unknown)")
                continue
            try:
                est_i = int(est)
            except (TypeError, ValueError):
                missing.add(f"{op}(cardinality-unknown)")
                continue
            lo, hi = min(cardinals), max(cardinals)
            if not (lo <= est_i <= hi):
                missing.add(f"{op}(cardinality-out-of-span)")
        else:
            if f"{op}_gpu_a" not in cost or f"{op}_cpu_a" not in cost:
                missing.add(op)
    return {"ok": not missing, "blockers": sorted(f"op:{m}" for m in missing)}


def estimate_graph(graph, n, profile, state="warm", selectivity=DEFAULT_SELECTIVITY):
    """Graph-level cost from MEASURED fits only. One call per graph."""
    cost = profile.get("cost", {})
    rows = downstream_rows(graph, int(n), selectivity)
    breakdown, cpu_total, gpu_total, unmeasured = [], 0.0, 0.0, []
    compile_ops = set()
    for nd in graph.get("nodes", []):
        op = nd.get("kernel_id", nd.get("op", "?"))
        out = nd.get("out", "")
        _, n_in, _ = rows.get(out, (op, int(n), int(n)))
        if op == "series":
            breakdown.append({"node": out, "op": op, "n_op": n_in,
                              "cpu_ms": 0.0, "gpu_ms": 0.0,
                              "note": "ingest view; bytes ride H2D inside op fits"})
            continue
        # groupby: cardinality-aware lookup
        if op == "groupby":
            card_key = _closest_groupby_cardinality(cost, out, nd, graph, n)
            ca = cost.get(f"groupby_M{card_key}_cpu_a")
            cb = cost.get(f"groupby_M{card_key}_cpu_b")
            ga = cost.get(f"groupby_M{card_key}_gpu_a")
            gb = cost.get(f"groupby_M{card_key}_gpu_b")
            compile_key = f"groupby_M{card_key}_gpu_compile_ms"
        else:
            ca, cb = cost.get(f"{op}_cpu_a"), cost.get(f"{op}_cpu_b")
            ga, gb = cost.get(f"{op}_gpu_a"), cost.get(f"{op}_gpu_b")
            compile_key = f"{op}_gpu_compile_ms"
        if None in (ca, cb, ga, gb):
            unmeasured.append(op)
            breakdown.append({"node": out, "op": op, "n_op": n_in,
                              "cpu_ms": 0.0, "gpu_ms": 0.0,
                              "note": "unmeasured op class: excluded, coverage gate owns it"})
            continue
        c_ms = float(ca) * n_in + float(cb)
        g_ms = float(ga) * n_in + float(gb)
        cpu_total += c_ms
        gpu_total += g_ms
        compile_ops.add(op)
        if op == "groupby":
            est_m = _groupby_est_m(nd, out, graph)
            note = (f"nearest-measured M={card_key}"
                    + (f" for est={est_m}" if est_m is not None else
                       " (no cardinality hint: conservative upper bound; "
                       "held-out M=50000 validation in dataset, not a fit)"))
        else:
            note = ""
        breakdown.append({"node": out, "op": op, "n_op": n_in,
                          "cpu_ms": c_ms, "gpu_ms": g_ms, "note": note})
    compile_ms = 0.0
    if state == "cold":
        for op in sorted(compile_ops):
            # groupby compile keyed by cardinality
            if op == "groupby":
                card = _closest_groupby_cardinality(
                    cost, None, None, graph, n)
                compile_ms += float(cost.get(f"groupby_M{card}_gpu_compile_ms", 0.0) or 0.0)
            else:
                compile_ms += float(cost.get(f"{op}_gpu_compile_ms", 0.0) or 0.0)
        gpu_total += compile_ms
    return {"cpu_ms": cpu_total, "gpu_host_ms": gpu_total,
            "compile_ms": compile_ms,
            "breakdown": breakdown, "unmeasured": sorted(set(unmeasured)),
            "state": state, "selectivity": selectivity,
            "note": ("resident multi-node GPU can only remove round trips "
                     "already inside host fits: gpu_host_ms is a conservative "
                     "upper bound; measured chains in dataset quantify it")}


def _groupby_est_m(nd, out, graph, hints=None):
    """Cardinality estimate for a groupby node: params, hints, or None."""
    hints = dict(hints or {})
    est_m = None
    if nd is not None:
        est_m = nd.get("params", {}).get("cardinality")
    if est_m is None:
        est_m = hints.get("cardinality", hints.get("groupby_cardinality"))
    if est_m is None and out is not None:
        # search graph for cardinality hints
        for node in graph.get("nodes", []):
            if node.get("out") == out:
                est_m = node.get("params", {}).get("cardinality")
                break
    if est_m is None:
        return None
    try:
        return int(est_m)
    except (TypeError, ValueError):
        return None


def _closest_groupby_cardinality(cost, out, nd, graph, n, hints=None):
    """Find closest measured cardinality for groupby estimation."""
    available = cost.get("groupby_cardinalities", list(GROUPBY_CARDINALITIES))
    est_m = _groupby_est_m(nd, out, graph, hints)
    if est_m is None:
        # default: use first available cardinality
        return available[0] if available else 100
    return min(available, key=lambda m: abs(m - est_m))


def format_explain(graph, info=None):
    """Full EXPLAIN: backend/graph/state/profile/estimated/gpu/breakdown/edges."""
    info = info or {}
    prof = info.get("profile", {})
    est = info.get("estimate") or {}
    lines = [f"EXPLAIN backend={info.get('actual', 'n/a')} "
             f"requested={info.get('requested', 'n/a')}"]
    ops = [f"{nd.get('kernel_id', nd.get('op', '?'))}:{nd.get('out', '')}"
           for nd in graph.get("nodes", [])]
    lines.append(f"GRAPH nodes={len(ops)} outputs={graph.get('outputs', [])} "
                 f"ops=[{', '.join(ops)}]")
    lines.append(f"STATE placement={info.get('placement', 'n/a')} "
                 f"state={est.get('state', 'n/a')} "
                 f"selectivity={est.get('selectivity', 'n/a')} "
                 f"dispatches={info.get('dispatches', 'n/a')} "
                 f"chunks={info.get('num_chunks', 'n/a')}")
    lines.append(f"PROFILE version={prof.get('version', 'n/a')} "
                 f"source={prof.get('source', 'n/a')} age={prof.get('age', 'n/a')} "
                 f"matched={prof.get('matched', 'n/a')} "
                 f"warning={prof.get('warning', '') or 'none'}")
    lines.append(f"ESTIMATED cpu={est.get('cpu_ms', 'n/a')}ms "
                 f"gpu_host={est.get('gpu_host_ms', 'n/a')}ms "
                 f"chosen={info.get('reason', 'n/a')}")
    lines.append(f"GPU eligible={info.get('gpu_eligible', 'n/a')} "
                 f"blockers={info.get('gpu_blockers', 'n/a')} "
                 f"coverage={info.get('coverage', 'n/a')}")
    for b in est.get("breakdown", []):
        lines.append(f"  node {b['node']} op={b['op']} n_op={b['n_op']} "
                     f"cpu={b['cpu_ms']:.3f}ms gpu={b['gpu_ms']:.3f}ms"
                     + (f" [{b['note']}]" if b.get("note") else ""))
    for nd in graph.get("nodes", []):
        lines.append(f"EDGE {nd.get('out', '')} <- {nd.get('inputs', [])} "
                     f"[{nd.get('kernel_id', nd.get('op', '?'))}]")
    return "\n".join(lines)


# ---- measurement (runs through PUBLIC aliases only; no driver imports) ----

def _synthetic(seed_n):
    import numpy as np
    rng = np.random.default_rng(SEED)
    a = rng.integers(0, 100, seed_n, dtype=np.int32)
    b = rng.integers(0, 2, seed_n).astype(bool)
    c = rng.integers(0, 2, seed_n).astype(bool)
    v = rng.integers(0, 100, seed_n, dtype=np.int32)
    ix = np.arange(seed_n, dtype=np.int32)
    return a, b, c, v, ix


def _op_jobs(alias, op, n, sel, cardinality=None):
    import numpy as _np
    a, b, c, v, ix = _synthetic(n)
    if op == "compare":
        return ([alias["ir_series"]("x", a),
                 alias["ir_compare"]("m", "x", 50, ">")], "m")
    if op == "mask":
        return ([alias["ir_series"]("x", b, "bool"),
                 alias["ir_series"]("y", c, "bool"),
                 alias["ir_mask"]("m", "x", "y", op="and")], "m")
    if op == "filter":
        thr = {0.1: 90, 0.5: 50, 0.9: 10}[min(SELECTIVITIES, key=lambda s: abs(s - sel))]
        return ([alias["ir_series"]("x", v),
                 alias["ir_compare"]("m", "x", thr, ">"),
                 alias["ir_filter"]("f", "x", "m")], "f")
    if op == "gather":
        return ([alias["ir_series"]("x", v),
                 alias["ir_series"]("i", ix),
                 alias["ir_gather"]("g", "x", "i")], "g")
    if op == "reduce":
        return ([alias["ir_series"]("x", v),
                 alias["ir_reduce"]("s", "x", "sum")], "s")
    if op == "rng_fill_i32":
        # Same harness: seed 42, production Ns (MATRIX_FULL_NS + VALIDATION_NS
        # like all MEASURED_OPS, not the candidate 10K-10M grid). Linear op
        # (candidate R^2>0.999), so the 1K-1M fit extrapolates to Pass 4A Ns;
        # single harness/reps/warmup, measurements.Ns stays consistent.
        # Counter-domain fill: CPU==GPU bit-exact, integrity exact.
        return ([alias["ir_rng_fill_i32"]("r", int(n), SEED, 0, 0, 0, 100)],
                "r")
    if op == "groupby":
        M = cardinality if cardinality is not None else 100
        rng = _np.random.default_rng(SEED)
        keys = rng.integers(0, M, n, dtype=_np.int32)
        vals = rng.integers(0, 100, n, dtype=_np.int32)
        return ([alias["ir_series"]("v", vals),
                 alias["ir_series"]("k", keys),
                 alias["ir_groupby"]("g", "v", "k", "sum")], "g")
    if op == "sort":
        return ([alias["ir_series"]("s", v),
                 alias["ir_sort"]("p", "s")], "p")
    if op == "slice":
        return ([alias["ir_series"]("s", v),
                 alias["ir_slice"]("ps", "s", limit=1000)], "ps")
    if op == "pack_keys":
        rng = _np.random.default_rng(SEED)
        k1 = rng.integers(0, 50, n, dtype=_np.int32)
        k2 = rng.integers(0, 100, n, dtype=_np.int32)
        return ([alias["ir_series"]("c1", k1),
                 alias["ir_series"]("c2", k2),
                 alias["ir_pack_keys"]("pk", "c1", "c2")], "pk")
    raise ValueError(f"calibrate: op '{op}' not in measured matrix")


def _chain_jobs(alias, name, n):
    a, b, c, v, ix = _synthetic(n)
    if name == "compare_filter":
        return ([alias["ir_series"]("x", v),
                 alias["ir_compare"]("m", "x", 50, ">"),
                 alias["ir_filter"]("f", "x", "m")], "f")
    if name == "compare_mask_filter":
        return ([alias["ir_series"]("x", v),
                 alias["ir_compare"]("m1", "x", 10, ">"),
                 alias["ir_compare"]("m2", "x", 90, "<"),
                 alias["ir_mask"]("m", "m1", "m2", op="and"),
                 alias["ir_filter"]("f", "x", "m")], "f")
    raise ValueError(f"calibrate: chain '{name}' unknown")


def _as_np(x):
    import numpy as np
    if isinstance(x, np.generic):
        return x.item()
    return np.asarray(x)


def _equal(cpu_res, gpu_res):
    import numpy as np
    c, g = _as_np(cpu_res), _as_np(gpu_res)
    if isinstance(c, (int, float)) or np.ndim(c) == 0:
        return float(c) == float(g)
    c, g = np.asarray(c).ravel(), np.asarray(g).ravel()
    return c.shape == g.shape and bool((c == g).all())


def _equal_groupby(cpu_res, gpu_res):
    if isinstance(cpu_res, dict) and isinstance(gpu_res, dict):
        if len(cpu_res) != len(gpu_res):
            return False
        return sum(int(v) for v in cpu_res.values()) == \
               sum(int(v) for v in gpu_res.values())
    return _equal(cpu_res, gpu_res)


def _r_squared(xs, ys):
    """R^2 for least-squares line. Pure, no constants."""
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    n = len(xs)
    if n < 2:
        return 1.0
    mx = sum(xs) / n
    my = sum(ys) / n
    ss_tot = sum((y - my) ** 2 for y in ys)
    if ss_tot == 0.0:
        return 1.0
    a, b = fit_line(xs, ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    return 1.0 - ss_res / ss_tot


def _mape(preds, meass):
    """Mean absolute percentage error. Pure, no constants."""
    errs = []
    for p, m in zip(preds, meass):
        if m != 0.0:
            errs.append(abs(p - m) / abs(m))
    return (sum(errs) / len(errs) * 100.0) if errs else 0.0


def _time_fn(fn):
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1000.0, out


def measure_matrix(alias, Ns, reps, warmup, sel_fit=FIT_SELECTIVITY):
    """Seed-42 synthetic matrix, cold vs warm, host + resident chains.

    Integrity first: GPU result must equal CPU exactly (int paths);
    mismatch raises (loss = STOP), never silently timed.
    """
    stages = []
    data = {"ops": {}, "chains": {}, "selectivity_sweep": {}}

    def run(backend, jobs):
        graph = alias["compile"](jobs)
        if backend == "cpu":
            return alias["cpu_execute"](graph["nodes"])
        return alias["gpu_execute"](graph["nodes"])

    # Device-init warmup (discarded, reported): first GPU call pays
    # adapter/device setup, never an op cost.
    jobs, out = _op_jobs(alias, "compare", 256, sel_fit)
    t_init, _ = _time_fn(lambda: run("gpu", jobs))
    stages.append(f"device_init=1x compare@256 gpu {t_init:.2f}ms (discarded)")

    for op in MEASURED_OPS:
        data["ops"][op] = {}
        for n in Ns:
            sels = SELECTIVITIES if op == "filter" else (sel_fit,)
            for sel in sels:
                jobs, out = _op_jobs(alias, op, n, sel)
                _, cpu_bufs = _time_fn(lambda: run("cpu", jobs))
                t0 = time.perf_counter()
                gpu_bufs = run("gpu", jobs)
                cold_ms = (time.perf_counter() - t0) * 1000.0
                if not _equal(cpu_bufs[out], gpu_bufs[out]):
                    raise RuntimeError(
                        f"calibrate integrity STOP: {op}@n={n} sel={sel}: "
                        "CPU != GPU (loss=100% of check)")
                key = f"{op}@n={n}" + (f" sel={sel}" if op == "filter" else "")
                if op == "filter" and abs(sel - sel_fit) > 1e-9:
                    data["selectivity_sweep"][f"n={n} sel={sel}"] = {
                        "cold_ms": cold_ms}
                ts = []
                for _ in range(max(0, warmup - 1)):
                    run("gpu", jobs)
                for _ in range(reps):
                    t, _ = _time_fn(lambda: run("gpu", jobs))
                    ts.append(t)
                ts.sort()
                warm_ms = ts[len(ts) // 2]
                c_ts = []
                for _ in range(reps):
                    t, _ = _time_fn(lambda: run("cpu", jobs))
                    c_ts.append(t)
                c_ts.sort()
                rec = {"cold_ms": cold_ms, "warm_ms": warm_ms,
                       "cpu_ms": c_ts[len(c_ts) // 2],
                       "cpu_runs_ms": c_ts, "gpu_runs_ms": ts}
                if op == "filter":
                    import numpy as _np
                    m = _np.asarray(gpu_bufs[out]).size
                    rec["measured_sel"] = (m / n) if n else 0.0
                data["ops"][op].setdefault(str(n), {})[str(sel)] = rec
                stages.append(f"{key} cpu={rec['cpu_ms']:.3f}ms "
                              f"gpu_cold={cold_ms:.3f}ms gpu_warm={warm_ms:.3f}ms")
    for name in ("compare_filter", "compare_mask_filter"):
        data["chains"][name] = {}
        for n in Ns:
            jobs, out = _chain_jobs(alias, name, n)
            _, cpu_bufs = _time_fn(lambda: run("cpu", jobs))
            t0 = time.perf_counter()
            gpu_bufs = run("gpu", jobs)
            cold_ms = (time.perf_counter() - t0) * 1000.0
            if not _equal(cpu_bufs[out], gpu_bufs[out]):
                raise RuntimeError(
                    f"calibrate integrity STOP: chain {name}@n={n}: CPU != GPU")
            ts = []
            for _ in range(max(0, warmup - 1)):
                run("gpu", jobs)
            for _ in range(reps):
                t, _ = _time_fn(lambda: run("gpu", jobs))
                ts.append(t)
            ts.sort()
            data["chains"][name][str(n)] = {
                "cold_ms": cold_ms, "warm_ms": ts[len(ts) // 2],
                "gpu_runs_ms": ts}
            stages.append(f"chain {name}@n={n} gpu_cold={cold_ms:.3f}ms "
                          f"gpu_warm={ts[len(ts)//2]:.3f}ms")

    # ---- extended ops: groupby (M x N matrix), sort, slice, pack_keys ----
    _measure_extended_ops(alias, data, stages, run, reps, warmup, sel_fit)

    # ---- transfer sweep (Pass 6): H2D/D2H byte grid, same seed/harness ----
    _measure_transfer(data, stages, reps, warmup)

    return data, stages


def _measure_transfer(data, stages, reps, warmup):
    """H2D/D2H byte sweep over TRANSFER_BYTES (Pass 5 grid, seed 42).

    Same harness conventions as the op matrix: synthetic seed-42 int32,
    warmup discarded, median of reps, integrity STOP, stage breakdown.
    Pure-transfer primitives (upload/download), so the fit ms = a*bytes + b
    needs no kernel differencing. The GPU driver module file is loaded
    directly (bench-only importlib pattern from Pass 5): no driver file is
    modified, no new API, frozen ABIs intact.
    """
    import gc
    import importlib.util as _ilu
    import numpy as _np
    gpu_py = _gpu_module_path()
    spec = _ilu.spec_from_file_location("nfcal_transfer_gpu", str(gpu_py))
    G = _ilu.module_from_spec(spec)
    spec.loader.exec_module(G)
    # Transfer-device warmup (discarded, reported): first touch pays setup,
    # never a transfer cost (same convention as device_init in measure_matrix).
    w0 = _np.random.default_rng(SEED).integers(0, 100, 256, dtype=_np.int32)
    t0 = time.perf_counter()
    G.r_download(G.r_upload(w0), _np.int32)
    stages.append(f"transfer device warmup 1x 1KB roundtrip "
                  f"{(time.perf_counter() - t0) * 1000.0:.2f}ms (discarded)")
    del w0
    data["transfer"] = {}
    for direction in ("h2d", "d2h"):
        data["transfer"][direction] = {}
        for nb in TRANSFER_BYTES:
            arr = _np.ascontiguousarray(
                _np.random.default_rng(SEED).integers(
                    0, 100, nb // 4, dtype=_np.int32))
            if direction == "h2d":
                for _ in range(warmup):
                    G.r_upload(arr)
                ts = []
                for _ in range(reps):
                    t, _ = _time_fn(lambda: G.r_upload(arr))
                    ts.append(t)
            else:
                buf = G.r_upload(arr)
                if not bool((G.r_download(buf, _np.int32) == arr).all()):
                    raise RuntimeError(
                        f"calibrate integrity STOP: d2h@{nb}B: "
                        "upload != download (loss=100% of check)")
                for _ in range(warmup):
                    G.r_download(buf, _np.int32)
                ts = []
                for _ in range(reps):
                    t, _ = _time_fn(lambda: G.r_download(buf, _np.int32))
                    ts.append(t)
                del buf
            ts.sort()
            med = ts[len(ts) // 2]
            data["transfer"][direction][str(nb)] = {
                "bytes": nb, "median_ms": med, "runs_ms": ts,
                "state": "CPU->GPU" if direction == "h2d" else "GPU->CPU"}
            mb = nb / 2 ** 20
            stages.append(f"transfer {direction.upper()} "
                          f"{mb:.0f}MB median={med:.2f}ms "
                          f"{nb / med / 1e6:.2f}GB/s")
            del arr
            gc.collect()


def _measure_extended_ops(alias, data, stages, run, reps, warmup, sel_fit):
    """Measure extended ops: groupby (cardinality x N), sort, slice, pack_keys."""
    # groupby: cardinality x N matrix
    data["ops"]["groupby"] = {}
    for M in GROUPBY_CARDINALITIES:
        for n in GROUPBY_NS:
            jobs, out = _op_jobs(alias, "groupby", n, sel_fit, cardinality=M)
            _, cpu_bufs = _time_fn(lambda: run("cpu", jobs))
            t0 = time.perf_counter()
            gpu_bufs = run("gpu", jobs)
            cold_ms = (time.perf_counter() - t0) * 1000.0
            if not _equal_groupby(cpu_bufs[out], gpu_bufs[out]):
                raise RuntimeError(
                    f"calibrate integrity STOP: groupby@n={n} M={M}: "
                    "CPU != GPU (loss=100% of check)")
            ts = []
            for _ in range(max(0, warmup - 1)):
                run("gpu", jobs)
            for _ in range(reps):
                t, _ = _time_fn(lambda: run("gpu", jobs))
                ts.append(t)
            ts.sort()
            warm_ms = ts[len(ts) // 2]
            c_ts = []
            for _ in range(reps):
                t, _ = _time_fn(lambda: run("cpu", jobs))
                c_ts.append(t)
            c_ts.sort()
            rec = {"cold_ms": cold_ms, "warm_ms": warm_ms,
                   "cpu_ms": c_ts[len(c_ts) // 2],
                   "cpu_runs_ms": c_ts, "gpu_runs_ms": ts}
            data["ops"]["groupby"].setdefault(str(n), {})[str(M)] = rec
            stages.append(f"groupby@n={n} M={M} cpu={rec['cpu_ms']:.3f}ms "
                          f"gpu_cold={cold_ms:.3f}ms gpu_warm={warm_ms:.3f}ms")
    # groupby validation point
    vn, vm = GROUPBY_VALIDATION["N"], GROUPBY_VALIDATION["M"]
    jobs, out = _op_jobs(alias, "groupby", vn, sel_fit, cardinality=vm)
    _, cpu_bufs = _time_fn(lambda: run("cpu", jobs))
    t0 = time.perf_counter()
    gpu_bufs = run("gpu", jobs)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    if not _equal_groupby(cpu_bufs[out], gpu_bufs[out]):
        raise RuntimeError(
            f"calibrate integrity STOP: groupby@n={vn} M={vm}: CPU != GPU")
    ts = []
    for _ in range(max(0, warmup - 1)):
        run("gpu", jobs)
    for _ in range(reps):
        t, _ = _time_fn(lambda: run("gpu", jobs))
        ts.append(t)
    ts.sort()
    c_ts = []
    for _ in range(reps):
        t, _ = _time_fn(lambda: run("cpu", jobs))
        c_ts.append(t)
    c_ts.sort()
    data["ops"]["groupby"].setdefault(str(vn), {})[str(vm)] = {
        "cold_ms": cold_ms, "warm_ms": ts[len(ts) // 2],
        "cpu_ms": c_ts[len(c_ts) // 2],
        "cpu_runs_ms": c_ts, "gpu_runs_ms": ts}
    stages.append(f"groupby@n={vn} M={vm} [validation] cpu={c_ts[len(c_ts)//2]:.3f}ms "
                  f"gpu_cold={cold_ms:.3f}ms gpu_warm={ts[len(ts)//2]:.3f}ms")

    # sort: power-of-two Ns
    data["ops"]["sort"] = {}
    for n in SORT_NS:
        jobs, out = _op_jobs(alias, "sort", n, sel_fit)
        _, cpu_bufs = _time_fn(lambda: run("cpu", jobs))
        t0 = time.perf_counter()
        gpu_bufs = run("gpu", jobs)
        cold_ms = (time.perf_counter() - t0) * 1000.0
        if not _equal(cpu_bufs[out], gpu_bufs[out]):
            raise RuntimeError(
                f"calibrate integrity STOP: sort@n={n}: CPU != GPU")
        ts = []
        for _ in range(max(0, warmup - 1)):
            run("gpu", jobs)
        for _ in range(reps):
            t, _ = _time_fn(lambda: run("gpu", jobs))
            ts.append(t)
        ts.sort()
        warm_ms = ts[len(ts) // 2]
        c_ts = []
        for _ in range(reps):
            t, _ = _time_fn(lambda: run("cpu", jobs))
            c_ts.append(t)
        c_ts.sort()
        rec = {"cold_ms": cold_ms, "warm_ms": warm_ms,
               "cpu_ms": c_ts[len(c_ts) // 2],
               "cpu_runs_ms": c_ts, "gpu_runs_ms": ts}
        data["ops"]["sort"][str(n)] = {"0.5": rec}
        stages.append(f"sort@n={n} cpu={rec['cpu_ms']:.3f}ms "
                      f"gpu_cold={cold_ms:.3f}ms gpu_warm={warm_ms:.3f}ms")

    # slice: Ns
    data["ops"]["slice"] = {}
    for n in SLICE_NS:
        jobs, out = _op_jobs(alias, "slice", n, sel_fit)
        _, cpu_bufs = _time_fn(lambda: run("cpu", jobs))
        t0 = time.perf_counter()
        gpu_bufs = run("gpu", jobs)
        cold_ms = (time.perf_counter() - t0) * 1000.0
        if not _equal(cpu_bufs[out], gpu_bufs[out]):
            raise RuntimeError(
                f"calibrate integrity STOP: slice@n={n}: CPU != GPU")
        ts = []
        for _ in range(max(0, warmup - 1)):
            run("gpu", jobs)
        for _ in range(reps):
            t, _ = _time_fn(lambda: run("gpu", jobs))
            ts.append(t)
        ts.sort()
        warm_ms = ts[len(ts) // 2]
        c_ts = []
        for _ in range(reps):
            t, _ = _time_fn(lambda: run("cpu", jobs))
            c_ts.append(t)
        c_ts.sort()
        rec = {"cold_ms": cold_ms, "warm_ms": warm_ms,
               "cpu_ms": c_ts[len(c_ts) // 2],
               "cpu_runs_ms": c_ts, "gpu_runs_ms": ts}
        data["ops"]["slice"][str(n)] = {"0.5": rec}
        stages.append(f"slice@n={n} cpu={rec['cpu_ms']:.3f}ms "
                      f"gpu_cold={cold_ms:.3f}ms gpu_warm={warm_ms:.3f}ms")

    # pack_keys: Ns (skip integrity check - CPU/GPU use different encodings)
    data["ops"]["pack_keys"] = {}
    for n in PACK_NS:
        jobs, out = _op_jobs(alias, "pack_keys", n, sel_fit)
        _, cpu_bufs = _time_fn(lambda: run("cpu", jobs))
        t0 = time.perf_counter()
        gpu_bufs = run("gpu", jobs)
        cold_ms = (time.perf_counter() - t0) * 1000.0
        # pack_keys: CPU=int64 radix, GPU=int32 direct - different encodings
        # are expected. Integrity verified downstream via groupby totals.
        ts = []
        for _ in range(max(0, warmup - 1)):
            run("gpu", jobs)
        for _ in range(reps):
            t, _ = _time_fn(lambda: run("gpu", jobs))
            ts.append(t)
        ts.sort()
        warm_ms = ts[len(ts) // 2]
        c_ts = []
        for _ in range(reps):
            t, _ = _time_fn(lambda: run("cpu", jobs))
            c_ts.append(t)
        c_ts.sort()
        rec = {"cold_ms": cold_ms, "warm_ms": warm_ms,
               "cpu_ms": c_ts[len(c_ts) // 2],
               "cpu_runs_ms": c_ts, "gpu_runs_ms": ts}
        data["ops"]["pack_keys"][str(n)] = {"0.5": rec}
        stages.append(f"pack_keys@n={n} cpu={rec['cpu_ms']:.3f}ms "
                      f"gpu_cold={cold_ms:.3f}ms gpu_warm={warm_ms:.3f}ms")


def fit_profile(data, Ns, quick, reps, warmup, hardware):
    """Fits from measured points only. Validation Ns excluded from fits."""
    fit_ns = [n for n in Ns if n not in VALIDATION_NS]
    cost = {}
    metrics = {}
    for op in MEASURED_OPS:
        have = [n for n in fit_ns
                if str(n) in data["ops"][op]
                and _fit_sel(op) in data["ops"][op][str(n)]]
        for backend, field in (("cpu", "cpu_ms"), ("gpu", "warm_ms")):
            ys = [data["ops"][op][str(n)][_fit_sel(op)][field] for n in have]
            a, b = fit_line(have, ys)
            cost[f"{op}_{backend}_a"] = a
            cost[f"{op}_{backend}_b"] = b
        gaps = [data["ops"][op][str(n)][_fit_sel(op)]["cold_ms"]
                - data["ops"][op][str(n)][_fit_sel(op)]["warm_ms"]
                for n in have]
        gaps.sort()
        cost[f"{op}_gpu_compile_ms"] = max(0.0, gaps[len(gaps) // 2]) \
            if gaps else 0.0
        cost[f"{op}_cpu_compile_ms"] = 0.0
        # R² per op (warm fits over fit Ns)
        for backend, field in (("cpu", "cpu_ms"), ("gpu", "warm_ms")):
            ys_fit = [data["ops"][op][str(n)][_fit_sel(op)][field] for n in have]
            r2 = _r_squared(have, ys_fit)
            metrics[f"{op}_{backend}_r2"] = r2

    # ---- extended ops: groupby per-cardinality fits ----
    cost["groupby_cardinalities"] = list(GROUPBY_CARDINALITIES)
    for M in GROUPBY_CARDINALITIES:
        have = [n for n in fit_ns
                if str(n) in data["ops"].get("groupby", {})
                and str(M) in data["ops"]["groupby"].get(str(n), {})]
        for backend, field in (("cpu", "cpu_ms"), ("gpu", "warm_ms")):
            ys = [data["ops"]["groupby"][str(n)][str(M)][field] for n in have]
            a, b = fit_line(have, ys)
            cost[f"groupby_M{M}_{backend}_a"] = a
            cost[f"groupby_M{M}_{backend}_b"] = b
            r2 = _r_squared(have, ys)
            metrics[f"groupby_M{M}_{backend}_r2"] = r2
        gaps = [data["ops"]["groupby"][str(n)][str(M)]["cold_ms"]
                - data["ops"]["groupby"][str(n)][str(M)]["warm_ms"]
                for n in have]
        gaps.sort()
        cost[f"groupby_M{M}_gpu_compile_ms"] = max(0.0, gaps[len(gaps) // 2]) \
            if gaps else 0.0
        cost[f"groupby_M{M}_cpu_compile_ms"] = 0.0

    # ---- extended ops: sort, slice, pack_keys standard fits ----
    sort_validation_n = 1048576  # held-out power-of-two for sort validation
    for op, op_ns, val_n in (("sort", SORT_NS, sort_validation_n),
                              ("slice", SLICE_NS, 5000),
                              ("pack_keys", PACK_NS, 500000)):
        op_fit_ns = [n for n in op_ns if n != val_n]
        have = [n for n in op_fit_ns
                if str(n) in data["ops"].get(op, {})
                and _fit_sel(op) in data["ops"][op].get(str(n), {})]
        for backend, field in (("cpu", "cpu_ms"), ("gpu", "warm_ms")):
            ys = [data["ops"][op][str(n)][_fit_sel(op)][field] for n in have]
            a, b = fit_line(have, ys)
            cost[f"{op}_{backend}_a"] = a
            cost[f"{op}_{backend}_b"] = b
            r2 = _r_squared(have, ys)
            metrics[f"{op}_{backend}_r2"] = r2
        gaps = [data["ops"][op][str(n)][_fit_sel(op)]["cold_ms"]
                - data["ops"][op][str(n)][_fit_sel(op)]["warm_ms"]
                for n in have]
        gaps.sort()
        cost[f"{op}_gpu_compile_ms"] = max(0.0, gaps[len(gaps) // 2]) \
            if gaps else 0.0
        cost[f"{op}_cpu_compile_ms"] = 0.0

    # ---- transfer fits (Pass 6): ms = a*bytes + b, measured points only ----
    for direction in ("h2d", "d2h"):
        pts = data.get("transfer", {}).get(direction, {})
        if direction == "h2d":
            # H2D superlinear at 1GB (measured, see 1GB holdout below):
            # fit covers <=100MB only; 1GB extrapolation is unmeasured.
            fit_bytes = [b for b in TRANSFER_BYTES
                         if b <= TRANSFER_H2D_FIT_MAX_BYTES]
            cov = TRANSFER_H2D_FIT_MAX_BYTES
        else:
            fit_bytes = [b for b in TRANSFER_BYTES]
            cov = max(TRANSFER_BYTES)
        have = [b for b in fit_bytes
                if str(b) in pts and "median_ms" in pts[str(b)]]
        ys = [pts[str(b)]["median_ms"] for b in have]
        a, b_ = fit_line(have, ys)
        cost[f"transfer_{direction}_a_ms_per_byte"] = a
        cost[f"transfer_{direction}_b_ms"] = b_
        cost[f"transfer_{direction}_fit_bytes"] = [int(x) for x in have]
        cost[f"transfer_{direction}_coverage_bytes"] = int(cov)
        metrics[f"transfer_{direction}_r2"] = _r_squared(have, ys)

    # ---- validation ----
    validation = {}
    # original ops validation
    for op in MEASURED_OPS:
        for n in VALIDATION_NS:
            rec = (data["ops"][op].get(str(n), {}).get(_fit_sel(op)))
            if rec is None:
                continue
            warm_pred = {b: cost[f"{op}_{b}_a"] * n + cost[f"{op}_{b}_b"]
                         for b in ("cpu", "gpu")}
            preds = {"cpu/warm": (warm_pred["cpu"], rec["cpu_ms"]),
                     "gpu/warm": (warm_pred["gpu"], rec["warm_ms"]),
                     "cpu/cold": (warm_pred["cpu"], rec["cpu_ms"]),
                     "gpu/cold": (warm_pred["gpu"]
                                  + cost[f"{op}_gpu_compile_ms"],
                                  rec["cold_ms"])}
            for key, (pred, meas) in preds.items():
                validation[f"{op}/{key}@n={n}"] = {
                    "predicted_ms": pred, "measured_ms": meas,
                    "abs_err_ms": pred - meas,
                    "rel_err": (pred - meas) / meas if meas else 0.0}
    # groupby validation: held-out (N=500K, M=50K)
    gvn = str(GROUPBY_VALIDATION["N"])
    gvm = str(GROUPBY_VALIDATION["M"])
    grec = data["ops"].get("groupby", {}).get(gvn, {}).get(gvm)
    if grec is not None:
        for M in GROUPBY_CARDINALITIES:
            ca = cost.get(f"groupby_M{M}_cpu_a")
            cb = cost.get(f"groupby_M{M}_cpu_b")
            ga = cost.get(f"groupby_M{M}_gpu_a")
            gb = cost.get(f"groupby_M{M}_gpu_b")
            if None not in (ca, cb, ga, gb):
                # find closest M
                closest = min(GROUPBY_CARDINALITIES, key=lambda m: abs(m - int(gvm)))
                if closest == M:
                    n_val = int(gvn)
                    pred_cpu = ca * n_val + cb
                    pred_gpu = ga * n_val + gb
                    validation[f"groupby/cpu/warm@n={gvn} M={gvm} (fit_M={M})"] = {
                        "predicted_ms": pred_cpu, "measured_ms": grec["cpu_ms"],
                        "abs_err_ms": pred_cpu - grec["cpu_ms"],
                        "rel_err": (pred_cpu - grec["cpu_ms"]) / grec["cpu_ms"]
                        if grec["cpu_ms"] else 0.0}
                    validation[f"groupby/gpu/warm@n={gvn} M={gvm} (fit_M={M})"] = {
                        "predicted_ms": pred_gpu, "measured_ms": grec["warm_ms"],
                        "abs_err_ms": pred_gpu - grec["warm_ms"],
                        "rel_err": (pred_gpu - grec["warm_ms"]) / grec["warm_ms"]
                        if grec["warm_ms"] else 0.0}
    # sort/slice/pack_keys validation at held-out Ns
    sort_val_n = 1048576  # held-out power-of-two
    slice_val_n = 5000
    pack_val_n = 500000
    for op, val_n in (("sort", sort_val_n), ("slice", slice_val_n),
                      ("pack_keys", pack_val_n)):
        rec = data["ops"].get(op, {}).get(str(val_n), {}).get(_fit_sel(op))
        if rec is None:
            continue
        warm_pred = {b: cost[f"{op}_{b}_a"] * val_n + cost[f"{op}_{b}_b"]
                     for b in ("cpu", "gpu")}
        for b in ("cpu", "gpu"):
            pred = warm_pred[b]
            meas = rec["cpu_ms"] if b == "cpu" else rec["warm_ms"]
            validation[f"{op}/{b}/warm@n={val_n}"] = {
                "predicted_ms": pred, "measured_ms": meas,
                "abs_err_ms": pred - meas,
                "rel_err": (pred - meas) / meas if meas else 0.0}
    # transfer H2D 1GB holdout (fit covers <=100MB only: superlinear point)
    hpts = data.get("transfer", {}).get("h2d", {})
    hold = str(max(TRANSFER_BYTES))
    if hold in hpts and "transfer_h2d_a_ms_per_byte" in cost:
        nbytes, meas = int(hold), hpts[hold]["median_ms"]
        pred = cost["transfer_h2d_a_ms_per_byte"] * nbytes \
            + cost["transfer_h2d_b_ms"]
        validation["transfer_h2d@1GB_holdout"] = {
            "predicted_ms": pred, "measured_ms": meas,
            "abs_err_ms": pred - meas,
            "rel_err": (pred - meas) / meas if meas else 0.0,
            "note": "H2D superlinear at 1GB (measured): fit covers "
                    "<=100MB (transfer_h2d_coverage_bytes); "
                    "1GB extrapolation is unmeasured"}
    for name, series in data["chains"].items():
        for n in VALIDATION_NS:
            rec = series.get(str(n))
            if rec is None:
                continue
            validation[f"chain:{name}@n={n}"] = {
                "measured_warm_ms": rec["warm_ms"],
                "note": "resident evidence: no composition formula claims it"}
    # MAPE summary per op
    mape_summary = {}
    for op in ALL_OPS:
        for backend in ("cpu", "gpu"):
            preds_list, meass_list = [], []
            for k, v in validation.items():
                if k.startswith(f"{op}/{backend}/warm@") and "rel_err" in v:
                    preds_list.append(v["predicted_ms"])
                    meass_list.append(v["measured_ms"])
            if preds_list:
                mape_summary[f"{op}_{backend}_mape"] = _mape(preds_list, meass_list)
    profile = {"model_version": SCHEMA, "profile_version": 2,
               "generated": _now_iso(), "source": "measured:seed42",
               "quick": bool(quick), "hardware": hardware, "cost": cost,
               "metrics": {**metrics, **mape_summary},
               "measurements": {"Ns": list(Ns), "quick": bool(quick),
                                "validation_Ns": list(VALIDATION_NS),
                                "reps": reps, "warmup": warmup,
                                "selectivities": list(SELECTIVITIES),
                                "fit_selectivity": FIT_SELECTIVITY,
                                "seed": SEED,
                                "groupby_cardinalities": list(GROUPBY_CARDINALITIES),
                                "groupby_Ns": list(GROUPBY_NS),
                                "groupby_validation": GROUPBY_VALIDATION,
                                 "sort_Ns": list(SORT_NS),
                                 "slice_Ns": list(SLICE_NS),
                                 "pack_Ns": list(PACK_NS),
                                 "transfer_bytes": list(TRANSFER_BYTES)}}
    dataset = {"profile": {k: v for k, v in profile.items()},
               "raw": data, "validation": validation}
    return profile, dataset


def _fit_sel(op):
    if op == "filter":
        return str(FIT_SELECTIVITY)
    return str(FIT_SELECTIVITY)


# ---- Auto Planner v1 (P1-P4, P8-P9): additive context, structured cold/
# transfer slots, residence, calibration hooks. No numbers invented here:
# every measured value comes from the profile; anything unmeasured is an
# explicit unknown slot and routes to the safe CPU choice, never to a
# fabricated GPU win. No N->backend thresholds in this block.

# Residence vocabulary (whole-graph placement; per-op routing is forbidden
# by construction, so ping-pong cannot be expressed).
_RESIDENCE_SIDES = ("host", "resident")
_RESIDENCE_ALIAS = {"cpu": "host", "gpu": "resident"}


def _side(v, default="host"):
    s = str(v).lower() if v is not None else default
    return _RESIDENCE_ALIAS.get(s, s) if s in _RESIDENCE_ALIAS or s in _RESIDENCE_SIDES else default


def build_context(graph, n, hints=None, profile=None):
    """Additive planner context (P1). Pure; callers keep old signatures.

    Returns {op_multiset, n, n_ops, dtypes, cardinalities, nearest,
    selectivity, residence_from, device_state, profile}. selectivity only
    sizes downstream rows (downstream_rows), never gates or routes.
    """
    hints = dict(hints or {})
    nodes = list(graph.get("nodes", []))
    counts = {}
    for nd in nodes:
        op = nd.get("kernel_id", nd.get("op", "?"))
        counts[op] = counts.get(op, 0) + 1
    dtypes = sorted({str(nd.get("params", {}).get("dtype", "int32"))
                     for nd in nodes if "series" in
                     (nd.get("kernel_id", nd.get("op", "")),)})
    cards = {}
    for nd in nodes:
        if nd.get("kernel_id", nd.get("op", "?")) in ("groupby", "groupby_multi"):
            est = nd.get("params", {}).get("cardinality",
                                           hints.get("cardinality",
                                                     hints.get("groupby_cardinality")))
            if est is not None:
                try:
                    cards[nd.get("out", "")] = int(est)
                except (TypeError, ValueError):
                    pass
    cost = profile.get("cost", {}) if isinstance(profile, dict) else {}
    measured_ms = [m for m in cost.get("groupby_cardinalities", [])
                   if isinstance(m, int)]
    nearest = {out: (min(measured_ms, key=lambda m: abs(m - est))
                     if measured_ms else None)
               for out, est in cards.items()}
    sel = hints.get("selectivity", DEFAULT_SELECTIVITY)
    try:
        sel = float(sel)
    except (TypeError, ValueError):
        sel = DEFAULT_SELECTIVITY
    state = hints.get("device_state", hints.get("state", "warm"))
    state = str(state).lower() if state in ("cold", "warm") else "warm"
    return {"op_multiset": sorted(counts.items()), "n": int(n),
            "n_ops": sum(c for op, c in counts.items() if op != "series"),
            "dtypes": dtypes, "cardinalities": cards, "nearest": nearest,
            "selectivity": sel,
            "residence_from": _side(hints.get("residence_from",
                                              hints.get("residence", "host"))),
            "device_state": state,
            "profile": profile_status(profile) if isinstance(profile, dict)
            else profile_status(None)}


def transfer_slots(residence_from, residence_to, n, profile=None,
                   absorbed=False):
    """Structured H2D/D2H slots (P3-P4). No fabricated per-byte numbers.

    Same side -> zero crossing (structural, needs no measurement).
    Cross side with absorbed fits -> absorbed-in-fit (warm (a,b) absorb
    steady-state dispatch+transfer, DELTA-10): +0.0 additional.
    Cross side without measurement evidence -> unknown/unmeasured slot
    (ms None): the safe choice is CPU, decided by the caller.
    """
    frm, to = _side(residence_from), _side(residence_to)
    slots = {}
    h2d_cross = frm == "host" and to == "resident"
    d2h_cross = frm == "resident" and to == "host"
    cost = profile.get("cost", {}) if isinstance(profile, dict) else {}
    for name, cross in (("h2d", h2d_cross), ("d2h", d2h_cross)):
        if not cross:
            slots[name] = {"status": "zero-by-residence", "ms": 0.0,
                           "note": f"{frm}->{to}: no bytes cross"}
            continue
        if absorbed:
            slots[name] = {"status": "absorbed-in-fit", "ms": 0.0,
                           "note": f"{frm}->{to}: warm fits absorb "
                                   "steady-state transfer (DELTA-10)"}
            continue
        key = f"transfer_{name}_ms"
        if key in cost and cost[key] is not None:
            slots[name] = {"status": "measured", "ms": float(cost[key]),
                           "note": f"{frm}->{to}: measured transfer slot"}
        else:
            slots[name] = {"status": "unmeasured", "ms": None,
                           "note": f"{frm}->{to}: no measured transfer "
                                   "matrix; safe CPU choice downstream"}
    slots["residence"] = {"from": frm, "to": to, "n": int(n)}
    return slots


def cold_slots(state, compile_ops, cost):
    """Structured cold slots (P3): device_init_one_time + compile_miss.

    device_init is reported-and-discarded at calibration time, never an op
    cost: without a dedicated measurement the slot stays unmeasured.
    compile_miss sums measured per-op cold-warm gaps only; ops without a
    measured gap are listed unmeasured, never zero-filled silently.
    """
    cost = cost if isinstance(cost, dict) else {}
    dev_key = "device_init_one_time_ms"
    if dev_key in cost and cost[dev_key] is not None:
        dev = {"status": "measured", "ms": float(cost[dev_key]), "note": ""}
    else:
        dev = {"status": "unmeasured", "ms": None,
               "note": "first-GPU-call setup is discarded at calibration; "
                       "no dedicated measurement: unknown"}
    if state != "cold":
        return {"device_init_one_time": {"status": "none", "ms": 0.0,
                                         "note": "warm: setup already paid"},
                "compile_miss": {"status": "none", "ms": 0.0, "ops": [],
                                 "note": "warm: pipelines already compiled"}}
    total, measured_ops, unmeasured_ops = 0.0, [], []
    for op in sorted(set(compile_ops)):
        key = None
        if op == "groupby":
            cands = [k for k in cost if k.startswith("groupby_M")
                     and k.endswith("_gpu_compile_ms")]
            key = sorted(cands)[0] if cands else "groupby_gpu_compile_ms"
        else:
            key = f"{op}_gpu_compile_ms"
        if key in cost and cost[key] is not None:
            total += float(cost[key])
            measured_ops.append(op)
        else:
            unmeasured_ops.append(op)
    return {"device_init_one_time": dev,
            "compile_miss": {"status": "measured" if not unmeasured_ops
                             else "partially-measured",
                             "ms": total, "ops": measured_ops,
                             "unmeasured_ops": unmeasured_ops,
                             "note": "median cold-warm gap per op pipeline"}}


def estimate_cost_v1(graph, n, profile, context=None):
    """Cost-model v1 (P2): CPU Σ(a·n_in+b); GPU host Σ + cold + transfer.

    Sums come from measured fits only (estimate_graph). Transfer/cold ride
    structured slots; any unmeasured slot forces estimated gpu to None
    (unknown) so the caller keeps the safe CPU choice. CPU is always a
    candidate after filtering (cpu_ms None only if a CPU fit is missing,
    which never happens for measured ops).
    """
    ctx = context or build_context(graph, n, profile=profile)
    est = estimate_graph(graph, int(n), profile,
                         state=ctx["device_state"],
                         selectivity=ctx["selectivity"])
    cost = profile.get("cost", {}) if isinstance(profile, dict) else {}
    compile_ops = [b["op"] for b in est.get("breakdown", [])
                   if b["op"] != "series" and "unmeasured" not in b.get("note", "")]
    cold = cold_slots(ctx["device_state"], compile_ops, cost)
    absorbed = not est.get("unmeasured")
    slots = transfer_slots(ctx["residence_from"], "resident", int(n), profile,
                           absorbed=absorbed)
    gpu_unknown = [k for k, v in {**cold, **{kk: vv for kk, vv in slots.items()
                                             if kk in ("h2d", "d2h")}}.items()
                   if isinstance(v, dict) and v.get("ms") is None]
    if ctx["device_state"] == "cold" and \
            cold["device_init_one_time"].get("ms") is None:
        gpu_unknown.append("device_init_one_time")
    gpu_ms = None if gpu_unknown else est["gpu_host_ms"]
    for name in ("h2d", "d2h"):
        if slots[name].get("ms"):
            gpu_ms = (gpu_ms or 0.0) + float(slots[name]["ms"])
    return {"cpu_ms": est["cpu_ms"], "gpu_host_ms": gpu_ms,
            "slots": {"cold": cold, "transfer": slots},
            "unknown_slots": sorted(set(gpu_unknown)),
            "breakdown": est.get("breakdown", []),
            "unmeasured": est.get("unmeasured", []),
            "state": est.get("state"), "selectivity": est.get("selectivity"),
            "context": ctx,
            "note": est.get("note", "") + "; v1: cold+transfer structural, "
                    "unknown slots force safe CPU"}


def propagate_residence(graph, residence_from="host", backend="cpu"):
    """Residence propagation (P4): whole-graph placement, no ping-pong.

    Per-op routing is forbidden (one backend per graph), so the placement
    is a single edge: host->host (0), host->resident (H2D), resident->host
    (D2H), resident->resident (0). Returns {placement, crossings}.
    """
    frm = _side(residence_from)
    to = "resident" if backend == "gpu" else "host"
    if frm == to:
        crossings = []
    elif to == "resident":
        crossings = ["h2d"]
    else:
        crossings = ["d2h"]
    return {"placement": to,
            "from": frm, "to": to, "crossings": crossings,
            "ping_pong": False,
            "note": "single backend per graph: at most one crossing"}


# ---- Calibration hooks (P8): STRUCTURE only, measurements NOT generated ----
CALIBRATION_HOOKS = (
    {"hook": "rng-i32", "ops": ["rng_fill_i32"],
     "kind": "measured-matrix",
     "detail": "counter-domain RNG fill; needs cold-vs-warm host+resident"},
    {"hook": "map", "ops": ["map"],
     "kind": "not-applicable",
     "detail": "no GPU capability (CPU-oracle): nothing to measure"},
    {"hook": "filter", "ops": ["filter"],
     "kind": "measured-matrix",
     "detail": "selectivity sweep sizes downstream n_op + D2H tail only"},
    {"hook": "groupby-card", "ops": ["groupby", "groupby_multi"],
     "kind": "measured-matrix-per-cardinality",
     "detail": "per-M fits; held-out (N=500K, M=50K) validation red flag"},
    {"hook": "sort-nonpow2", "ops": ["sort"],
     "kind": "not-applicable",
     "detail": "non-pow2/multi-key/NaN sorts are CPU-only explicit"},
    {"hook": "transfer", "ops": [],
     "kind": "unmeasured-slot",
     "detail": "H2D/D2H per-byte fits (Pass 6: H2D fit <=100MB + 1GB "
               "holdout, D2H fit <=1GB)"},
    {"hook": "device-init", "ops": [],
     "kind": "unmeasured-slot",
     "detail": "first-GPU-call setup discarded at calibration: unknown"},
)


def hook_status(profile=None):
    """Per-hook measured/unmeasured report from profile cost keys only."""
    cost = profile.get("cost", {}) if isinstance(profile, dict) else {}
    out = []
    for h in CALIBRATION_HOOKS:
        if h["kind"] == "not-applicable":
            out.append({**h, "measured": None, "status": "not-applicable"})
            continue
        if h["kind"] == "unmeasured-slot":
            if h["hook"] == "transfer":
                # Pass 6: per-byte linear fits (ms = a*bytes + b); H2D fit
                # covers <=100MB (1GB holdout in dataset), D2H covers 1GB.
                has = ("transfer_h2d_a_ms_per_byte" in cost
                       and "transfer_d2h_a_ms_per_byte" in cost)
            else:
                key = {"transfer": "transfer_h2d_ms",
                       "device-init": "device_init_one_time_ms"}[h["hook"]]
                has = key in cost and cost[key] is not None
            out.append({**h, "measured": bool(has),
                        "status": "measured" if has else "unmeasured"})
            continue
        if h["hook"] == "groupby-card":
            ms = cost.get("groupby_cardinalities", [])
            ok = bool(ms) and all(f"groupby_M{m}_gpu_a" in cost for m in ms)
            out.append({**h, "measured": bool(ok),
                        "status": "measured" if ok else "unmeasured",
                        "cardinalities": list(ms)})
            continue
        ok = all(f"{op}_gpu_a" in cost and f"{op}_cpu_a" in cost
                 for op in h["ops"])
        out.append({**h, "measured": bool(ok),
                    "status": "measured" if ok else "unmeasured"})
    return out


def _hardware(alias):
    import platform
    hw = {"cpu": platform.processor() or "unknown",
          "platform": platform.platform(), "python": platform.python_version(),
          "backend": "webgpu", "gpu_device": "unknown",
          "gpu_backend": "unknown", "vram_mb": "unknown",
          "vram_source": "unknown"}
    try:
        cap = alias.get("gpu_capability", lambda: {})()
        note = str(cap.get("note", ""))
        for token in ("RTX 2060", "Vulkan", "DX12"):
            if token in note:
                if "RTX" in token:
                    hw["gpu_device"] = "NVIDIA GeForce RTX 2060"
                else:
                    hw["gpu_backend"] = token
    except (AttributeError, TypeError, KeyError):
        pass
    if hw["gpu_device"] == "unknown":
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader"], capture_output=True, text=True,
                timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                name, mem = [x.strip() for x in
                             out.stdout.strip().split(",")[:2]]
                hw["gpu_device"] = name
                hw["vram_mb"] = int("".join(c for c in mem if c.isdigit()) or 0)
                hw["vram_source"] = "nvidia-smi"
        except (OSError, ValueError):
            pass
    return hw


def calibrate_impl(alias, quick=False, force=False, path=None):
    """nf.calibrate(quick, force). Measure -> fit -> write artifacts."""
    p = path or profile_path()
    if not force and Path(p).exists() and load_profile(p) is not None:
        prof = load_profile(p)
        return {"status": "reused", "path": p,
                "profile": profile_status(prof),
                "note": "profile exists; force=True remeasures"}
    Ns = tuple(MATRIX_QUICK_NS) if quick else tuple(MATRIX_FULL_NS)
    all_ns = tuple(Ns) + tuple(n for n in VALIDATION_NS if n not in Ns)
    reps, warmup = (3, 2) if quick else (5, 3)
    data, stages = measure_matrix(alias, all_ns, reps, warmup)
    profile, dataset = fit_profile(data, all_ns, quick, reps, warmup,
                                   _hardware(alias))
    Path(p).write_text(_toml_dump(profile), encoding="utf-8")
    dp = dataset_path()
    Path(dp).write_text(json.dumps(dataset, indent=1), encoding="utf-8")
    return {"status": "measured", "path": p, "dataset": dp,
            "profile": profile_status(profile),
            "stages": stages, "validation": dataset["validation"],
            "note": "all numbers measured this run (seed 42); "
                    "validation Ns excluded from fits"}
