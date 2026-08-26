# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""WalkForwardFullFused -- extension facade.

Reuses Compute._lib.walkforward_full_fused (descriptor / wgsl / cpu) --
no duplicated kernel logic. Adds:
- WalkForwardFullFused(): high-level dispatch (compile -> execute ->
  resolve_output), GPU driver preferred, CPU fallback.
- last_profile(): stage breakdown (ms) of the last dispatch.

Output layout (both backends, by construction):
    out u32[2*ntuples*folds]; index i=(t*folds+fold):
    out[2i] = f32-bits of masked return sum, out[2i+1] = count.
"""
import time

import numpy as np

from Compute._lib.walkforward_full_fused.cpu import cpu, evaluate, mean_ret
from Compute._lib.walkforward_full_fused.descriptor import (
    SEED_OUT,
    describe as _describe_base,
    describe_seed,
)
from Compute._lib.walkforward_full_fused.wgsl import WGSL, SEED_WGSL, build_tuples

_OUT_NAME = "wf_full_fused_out"
_KERNEL = {"k": None}
_RT_CACHE = {"rt": None, "backend": None}
_LAST_PROFILE = {}

# Mirror of descriptor.register() capabilities (static config).
_CAPABILITIES = {
    "input_dtype": "float",
    "output_dtype": "uint",
    "zero_init": True,
    "multi_input": True,
    "multi_output": False,
    "chunkable": False,
    "streaming": False,
    "workspace": False,
}

_SEED_CAPABILITIES = {
    "input_dtype": "float",
    "output_dtype": "float",
    "zero_init": False,
    "multi_input": False,
    "multi_output": False,
    "chunkable": False,
    "streaming": False,
    "workspace": False,
}


def describe(params: dict):
    """Base describe + output_size_fn (u32[2*ntuples*folds]).

    Base plan has no output_size_fn -> generic builder would allocate
    input-sized output. This kernel's output is fixed-size.
    """
    plan = _describe_base(params)
    size = 2 * len(build_tuples()) * int(params.get("folds", 4))
    plan.output_size_fn = lambda input_sizes: [size]
    return plan


def register(runtime):
    """Register WalkForwardFullFused (+Seed) into a Runtime kernel_table."""
    from Compute._lib.walkforward_full_fused.descriptor import _cpu_seed
    runtime.register_kernel(
        "WalkForwardFullFused",
        describe=describe,
        cpu=cpu,
        wgsl=WGSL,
        abi_version=1,
        capabilities=_CAPABILITIES,
    )
    runtime.register_kernel(
        "WalkForwardFullFusedSeed",
        describe=describe_seed,
        cpu=_cpu_seed,
        wgsl=SEED_WGSL,
        abi_version=1,
        capabilities=_SEED_CAPABILITIES,
    )


def bind_kernel(kernel):
    """Capture app kernel (setup-time) -- source of Runtime aliases.

    Aliases come from the builder-loaded Runtime extension, whose
    singleton owns the authoritative kernel_table.
    """
    _KERNEL["k"] = kernel


def _standalone_runtime():
    """Direct-import path (no app build): fresh Runtime + own kernel."""
    from Runtime._lib.runtime import Runtime as _RT

    backend = "cpu"
    rt = None
    try:
        from Runtime._lib.Drivers.WebGPU import WebGpuDriver

        rt = _RT(driver=WebGpuDriver())
        backend = "gpu"
    except Exception:  # noqa: BLE001 - no adapter -> CPU oracle fallback
        from Runtime._lib.Drivers.CPU._lib.cpu_driver import CpuDriver

        rt = _RT(driver=CpuDriver())
    register(rt)
    return rt, backend


def _get_runtime():
    """Runtime sharing authoritative app kernel_table; GPU preferred."""
    if _RT_CACHE["rt"] is not None:
        return _RT_CACHE["rt"], _RT_CACHE["backend"]

    k = _KERNEL["k"]
    if k is None or "Runtime" not in k.alias:
        rt, backend = _standalone_runtime()
        _RT_CACHE.update(rt=rt, backend=backend)
        _LAST_PROFILE["backend"] = backend
        return rt, backend

    backend = "cpu"
    rt = None
    try:
        rt = k.alias["Runtime"](driver=k.alias["GpuDriver"]())
        backend = "gpu"
    except Exception:  # noqa: BLE001 - no adapter -> CPU oracle fallback
        rt = k.alias["Runtime"](driver=k.alias["CpuDriver"]())
    _RT_CACHE.update(rt=rt, backend=backend)
    _LAST_PROFILE["backend"] = backend
    return rt, backend


def last_profile():
    """Stage breakdown (ms) of the last WalkForwardFullFused call."""
    return dict(_LAST_PROFILE)


def WalkForwardFullFused(close, folds=4, fold_sz=None, chunk=4096):
    """Two-dispatch walk-forward evaluation over close prices.

    opt="checkpointing-B": job chain
        WalkForwardFullFusedSeed (grid (R,folds), serial state snapshots)
          -> WalkForwardFullFused (grid (dx,folds), loads checkpoint,
             zero replay).
    Runtime ExecutionScheduler fuses the 2-packet DAG: close uploads once,
    the snapshot buffer stays GPU-resident between packets, only out[160]
    u32 is read back => 1 upload / 2 dispatch / 1 readback per dataset.
    Math/semantics unchanged (snapshot == old replay state, bit-exact f32).

    Args:
        close: array-like, f32-castable close prices.
        folds: number of contiguous folds, [1, 65535].
        fold_sz: rows per fold; default n // folds.
        chunk: rows per workgroup chunk within a fold, >= 1.

    Returns:
        (sums f64[ntuples*folds], counts i64[ntuples*folds]);
        tuple t, fold k -> index t*folds+k. Mean per cell:
        mean_ret(sums, counts, folds=folds).
    """
    data = np.ascontiguousarray(np.asarray(close, dtype=np.float32))
    n = int(data.size)
    folds = int(folds)
    chunk = int(chunk)
    if n < 1:
        raise ValueError(f"WalkForwardFullFused n must be >= 1; got {n}")
    if folds < 1 or folds > 65535:
        raise ValueError(f"WalkForwardFullFused folds must be in [1,65535]; got {folds}")
    if chunk < 1:
        raise ValueError(f"WalkForwardFullFused chunk must be >= 1; got {chunk}")
    if fold_sz is None:
        fold_sz = max(1, n // folds)

    # Port name must NOT be a QuoteTable logical field ('close' would be
    # expanded to OHLC physicals + AstEval by compiler._resolve_quote_table).
    params = {"n": n, "fold_sz": int(fold_sz), "folds": folds, "chunk": chunk}
    jobs = [
        {
            "op": "WalkForwardFullFusedSeed",
            "inputs": ["wf_close"],
            "params": params,
            "out": SEED_OUT,
        },
        {
            "op": "WalkForwardFullFused",
            "inputs": ["wf_close", SEED_OUT],
            "params": params,
            "out": _OUT_NAME,
        },
    ]
    rt, backend = _get_runtime()

    t0 = time.perf_counter()
    tasks = rt.compile(jobs)
    t1 = time.perf_counter()
    rt.execute(tasks, {"wf_close": data})
    t2 = time.perf_counter()
    raw = np.asarray(rt.driver.resolve_output(_OUT_NAME))
    t3 = time.perf_counter()

    ntuples = len(build_tuples())
    # resolve_output returns u32 values stored as f64 (builder allocates
    # float64 raw; bytes_to_blockview casts u32 -> f64 losslessly).
    raw_u32 = np.asarray(raw, dtype=np.float64).astype(np.uint32)
    sums = np.ascontiguousarray(raw_u32[0::2]).view(np.float32).astype(np.float64)
    cnts = raw_u32[1::2].astype(np.int64)

    _LAST_PROFILE.update({
        "backend": _LAST_PROFILE.get("backend", "?"),
        "n": n,
        "folds": folds,
        "fold_sz": int(fold_sz),
        "chunk": chunk,
        "compile_ms": round((t1 - t0) * 1000.0, 3),
        "execute_ms": round((t2 - t1) * 1000.0, 3),
        "readback_ms": round((t3 - t2) * 1000.0, 3),
        "total_ms": round((t3 - t0) * 1000.0, 3),
    })
    return sums, cnts


__all__ = [
    "WGSL",
    "SEED_WGSL",
    "WalkForwardFullFused",
    "bind_kernel",
    "build_tuples",
    "cpu",
    "describe",
    "describe_seed",
    "evaluate",
    "last_profile",
    "mean_ret",
    "register",
]
