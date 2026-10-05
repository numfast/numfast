# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.calibrate import calibrate_impl as _calibrate_impl
from _lib.calibrate import hook_status as _hook_status
from _lib.calibrate import load_profile as _load_profile
from _lib.calibrate import profile_path as _profile_path
from _lib.calibrate import profile_status as _profile_status
from _lib.calibrate import resolve_routing_profile as _resolve_routing_profile
from _lib.planner import chunk_plan_impl as _chunk_plan_impl
from _lib.planner import compile_impl as _compile_impl
from _lib.planner import explain_impl as explain
from _lib.planner import optimize_impl as optimize
from _lib.planner import plan_filter_impl as _plan_filter_impl
from _lib.planner import plan_groupby_impl as _plan_groupby_impl
from _lib.planner import plan_pack_impl as _plan_pack_impl
from _lib.planner import select_backend_dual_impl as _select_backend_dual_impl

_box = {}


def _gpu_cap():
    return _box["kernel"].alias["gpu_capability"]()


def _cap():
    return _box["kernel"].alias["cpu_capability"]()


def _err():
    return _box["kernel"].alias["format_error"]


def compile(jobs):
    return _compile_impl(jobs, _cap(), _err())


def select_backend(graph, n, profile=None, hints=None):
    return _select_backend_dual_impl(graph, n, _cap(), _gpu_cap(), profile,
                                     hints)


def chunk_plan(op, n, backend_limit=None, backend="cpu", graph=None):
    cap = _gpu_cap() if backend == "gpu" else _cap()
    vram = None
    if backend == "gpu":
        try:
            vram = _box["kernel"].alias["gpu_vram_budget"]()
        except (KeyError, TypeError, AttributeError):
            vram = None
    return _chunk_plan_impl(op, n, backend_limit, cap, graph=graph,
                            vram_budget=vram)


def plan_groupby(n, is_sorted, nunique_hint=None, *, span=None,
                 est_dense_bytes=None, budget_bytes=None, kmin=None,
                 est_code_bytes=None, naggs=1):
    if budget_bytes is None and span is not None:
        try:
            budget_bytes = _cap().get("max_buffer_bytes")
        except (AttributeError, KeyError, TypeError):
            budget_bytes = None
    return _plan_groupby_impl(n, is_sorted, nunique_hint, span=span,
                              est_dense_bytes=est_dense_bytes,
                              budget_bytes=budget_bytes, kmin=kmin,
                              est_code_bytes=est_code_bytes, naggs=naggs)


def plan_pack(kmin1, kmax1, kmin2, kmax2, m2=None):
    return _plan_pack_impl(kmin1, kmax1, kmin2, kmax2, m2)


def plan_filter(n, selectivity=None):
    return _plan_filter_impl(n, selectivity)


def calibrate(quick=False, force=False):
    return _calibrate_impl(dict(_box["kernel"].alias), quick=quick,
                           force=force)


def calibrate_info():
    """What the Planner would route on here, and why.

    `profile` describes the FILE at `path` (what it measured, how complete).
    `routing` is the decision the planner acts on: whether that file may drive
    routing on this machine, and which [hardware] fields stood in the way when
    it may not. They differ on purpose -- a shipped profile can be complete and
    still not yours.
    """
    prof = _load_profile()
    _applied, routing = _resolve_routing_profile(_gpu_cap())
    return {"path": _profile_path(),
            "profile": _profile_status(prof),
            "routing": routing,
            "hooks": _hook_status(prof)}


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("Planner", {})["version"] = "0.2.0"


PUBLIC = {
    "compile": compile,
    "optimize": optimize,
    "select_backend": select_backend,
    "chunk_plan": chunk_plan,
    "explain": explain,
    "plan_groupby": plan_groupby,
    "plan_pack": plan_pack,
    "plan_filter": plan_filter,
    "calibrate": calibrate,
    "calibrate_info": calibrate_info,
}
