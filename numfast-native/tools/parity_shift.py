# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Shift full-port parity: NumPy reference vs Rust-native(ctypes) vs
WASM(via Node wasm_shift.mjs) vs GPU(WGSL shift_take), bit-exact.

Contract (frozen IR semantics, data plane): out[i] = 0 for i < periods
else src[i-periods]; periods == 0 copies; periods >= n zero-fills;
empty -> empty; NaN/Inf ride bit-exact (memcpy, never arithmetic).
Validity sidecar travels host-side (same carry rule as CPU _shift_ref):
checked here through the IR cpu_execute path vs the oracle.

GPU covers int32/float32 only (existing contract: float64 raises
ValueError) — asserted as explicit-error, never silent fallback.
IR/API/semantics untouched; no tolerance anywhere (tobytes equality
for floats = bitwise, NaN payload included).

Seed 42 everywhere. Prints table + stage breakdown, writes
results/parity_shift.json. Exit 0 = PASS.
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
import wasm_artifact

# The CURRENT build, validated: 86 exports. This used to point at
# tools/numfast_native.wasm, a TRACKED 56-function build 29 symbols
# behind, and reported green while validating the wrong artefact.
# wasm_artifact.resolve() aborts loudly rather than running against a
# missing or stale .wasm -- no fallback, no skip.
WASM = wasm_artifact.resolve()
print(wasm_artifact.banner(WASM))
MJS = os.path.join(ROOT, "tools", "wasm_shift.mjs")
NODE = shutil.which("node") or r"C:\Program Files\nodejs\node.exe"
TMP = os.path.join(RES, ".shift-tmp")
os.makedirs(TMP, exist_ok=True)
DEV = os.path.normpath(os.path.join(ROOT, "..", ".."))  # dev-env root
APP = os.path.join(DEV, "numfast")  # numfast package root (builder APP_DIR)

import importlib.util

sys.path.insert(0, os.path.join(DEV, "app-builder"))  # builder pkg
from builder import MAIN  # noqa: E402  (IR path for validity-carry checks)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NAT = _load("nf_native_cpu",
            os.path.join(APP, "src", "Drivers", "CPU", "_lib", "native_cpu.py"))
GPU = _load("nf_gpu",
            os.path.join(APP, "src", "Drivers", "GPU", "_lib", "gpu.py"))

KINDS = {"i32": np.int32, "f32": np.float32, "f64": np.float64}


def oracle_data(vals, periods):
    """Independent oracle, data plane only (fill 0 / 0.0 head)."""
    vals = np.ascontiguousarray(vals)
    n = vals.size
    out = np.zeros(n, dtype=vals.dtype)
    if periods < n:
        out[periods:] = vals[:n - periods]
    return out


def oracle_validity(n, periods, valid):
    """Host-side carry rule (same as CPU _shift_ref): head False."""
    if n == 0 and valid is None:
        return None  # empty all-valid input keeps no sidecar
    ov = np.zeros(n, dtype=bool)
    if periods == 0 and valid is None:
        return None  # all-valid identity keeps no sidecar
    if periods < n and n:
        if valid is None:
            ov[periods:] = True
        else:
            ov[periods:] = np.ascontiguousarray(np.asarray(valid, dtype=bool))[:n - periods]
    return ov


def bit_exact(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def wasm_shift(vals, kind, periods):
    dt = KINDS[kind]
    vals = np.ascontiguousarray(vals, dtype=dt)
    n = vals.size
    src = os.path.join(TMP, f"src.{kind}")
    vals.tofile(src)
    t = time.perf_counter()
    out = subprocess.run(
        [NODE, MJS, WASM, src, kind, str(n), str(periods), "parity", TMP],
        capture_output=True, text=True, check=True,
    )
    t_ms = (time.perf_counter() - t) * 1e3
    raw = open(os.path.join(TMP, f"shift_out.{kind}"), "rb").read()
    got = np.frombuffer(raw, dtype=dt).copy()
    assert got.size == n, (got.size, n)
    return got, t_ms


def gpu_shift(vals, kind, periods):
    dt = {"i32": "int32", "f32": "float32", "f64": "float64"}[kind]
    t = time.perf_counter()
    got = GPU.shift_take(np.ascontiguousarray(vals), periods, dt)
    return np.ascontiguousarray(got), (time.perf_counter() - t) * 1e3


def main():
    rng = np.random.default_rng(42)
    fails = []
    rows = []

    def check(label, cond, detail=""):
        rows.append((label, cond, detail))
        print(("PASS " if cond else "FAIL ") + label + ((" | " + detail) if detail else ""))
        if not cond:
            fails.append(label)

    # --- vectors: edges + seeded bulk (validity gaps, NaN/Inf lanes) ---
    cases = []
    cases.append(("empty/i32/p1", np.zeros(0, np.int32), "i32", 1, None))
    cases.append(("empty/f64/p0", np.zeros(0, np.float64), "f64", 0, None))
    cases.append(("n1/i32/p0", np.array([7], np.int32), "i32", 0, None))
    cases.append(("n1/i32/p1", np.array([7], np.int32), "i32", 1, None))
    cases.append(("n1/i32/p5", np.array([7], np.int32), "i32", 5, None))
    n = 1000
    vi = rng.integers(-2**30, 2**30, size=n).astype(np.int32)
    vf = rng.normal(0, 1000, size=n).astype(np.float32)
    vf[::97] = np.nan
    vf[::211] = np.inf
    vf[::313] = -np.inf
    vd = rng.normal(0, 1000, size=n).astype(np.float64)
    vd[::101] = np.nan
    gaps = (rng.integers(0, 2, size=n) == 1)
    for kind, vals in (("i32", vi), ("f32", vf), ("f64", vd)):
        nn = vals.size
        for p in (0, 1, 7, nn - 1, nn, nn + 3):
            cases.append((f"bulk/{kind}/p{p}", vals, kind, p, None))
        cases.append((f"bulk/{kind}/p3/gaps", vals, kind, 3, gaps))

    t_ref = t_nat = 0.0
    a = MAIN["build"](APP).alias
    for label, vals, kind, p, valid in cases:
        ref = oracle_data(vals, p)
        # CPU/reference is the oracle itself; CPU/native via ctypes.
        t = time.perf_counter()
        nat = NAT.shift_scatter(vals, p)
        t_nat += (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        _ = oracle_data(vals, p)
        t_ref += (time.perf_counter() - t) * 1e3
        check(f"{label} native==ref",
              bit_exact(nat, ref), f"dtype={kind} n={vals.size}")
        w, _ = wasm_shift(vals, kind, p)
        check(f"{label} wasm==ref", bit_exact(w, ref), f"dtype={kind} n={vals.size}")
        if kind in ("i32", "f32"):
            g, _ = gpu_shift(vals, kind, p)
            check(f"{label} gpu==ref", bit_exact(g, ref), f"dtype={kind} n={vals.size}")
        else:
            try:
                gpu_shift(vals, kind, p)
                check(f"{label} gpu-f64-explicit-error", False, "no error raised")
            except ValueError:
                check(f"{label} gpu-f64-explicit-error", True, "ValueError as contracted")
        # Validity carry through the real IR CPU path (builder).
        kw = {} if valid is None else {"validity": [int(v) for v in valid]}
        jobs = [a["ir_series"]("s", np.ascontiguousarray(vals),
                               {"i32": "int32", "f32": "float32",
                                "f64": "float64"}[kind], **kw),
                a["ir_shift"]("h", "s", p)]
        bufs = a["cpu_execute"](a["optimize"](a["compile"](jobs))["nodes"])
        got, gotv = np.ascontiguousarray(bufs["h"]), bufs.get("h#validity")
        expv = oracle_validity(vals.size, p, valid)
        vok = (expv is None and gotv is None) or (
            expv is not None and gotv is not None
            and [bool(v) for v in gotv] == [bool(v) for v in expv])
        check(f"{label} ir-data==ref", bit_exact(got, ref), f"n={vals.size}")
        check(f"{label} ir-validity-carry", vok,
              "no-sidecar" if expv is None else f"head{p}-False")

    print(f"\nstages ms: oracle_total={t_ref:.3f} native_total={t_nat:.3f} "
          f"cases={len(cases)}")
    json.dump({"seed": 42, "cases": len(cases),
               "oracle_ms": t_ref, "native_ms": t_nat,
               "pass": not fails, "fails": fails},
              open(os.path.join(RES, "parity_shift.json"), "w"), indent=1)
    import shutil as _s
    _s.rmtree(TMP, ignore_errors=True)
    print("PASS" if not fails else f"FAIL {fails}")
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
