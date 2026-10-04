# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Cumsum full-port parity: NumPy oracle vs CPU(IR) vs Rust-native(ctypes)
vs WASM(via Node wasm_cumsum.mjs) vs GPU(standalone WGSL scan_inclusive).

Contract (frozen IR semantics c52316e, data plane): inclusive prefix
out[i] = sum(x[0..i]); int32 wraps mod 2**32 (never trap); floats
accumulate in their own dtype (existing numerical contract from
specs/conformance-profile.toml vs an f64 oracle); NaN is a
value with IEEE forward propagation (never a validity signal);
invalid rows contribute 0, output validity is a per-row copy of the
input (downstream resumes); empty -> empty same dtype.

Validity sidecar travels host-side (0-fill before data kernels, carry
after): checked here through the IR cpu_execute path vs the oracle.

GPU covers int32 only (existing standalone scan has exact u32 lanes;
float lanes do not exist) — f32/f64 asserted as explicit gaps, never
silent fallback. IR/API/semantics untouched; int bit-exact (tobytes),
float tolerance from the profile (NaN positions must match exactly).

Seed 42 everywhere. Prints table + stage breakdown, writes
results/parity_cumsum.json. Exit 0 = PASS.
"""
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
import tomllib

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
MJS = os.path.join(ROOT, "tools", "wasm_cumsum.mjs")
NODE = shutil.which("node") or r"C:\Program Files\nodejs\node.exe"
TMP = os.path.join(RES, ".cumsum-tmp")
os.makedirs(TMP, exist_ok=True)
DEV = os.path.normpath(os.path.join(ROOT, "..", ".."))  # dev-env root
APP = os.path.join(DEV, "numfast")  # numfast package root (builder APP_DIR)
PROFILE = os.path.join(APP, "specs-rebuilt", "conformance-profile.toml")

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

with open(PROFILE, "rb") as f:
    _PROF = tomllib.load(f)


def _tol(dt):
    key = "f32" if np.dtype(dt) == np.dtype(np.float32) else "f64"
    sec = _PROF["tolerance"][key]
    return sec["atol"], sec["rtol"]


def oracle_cumsum(vals, validity=None):
    """Independent oracle: (out_vals, out_valid_or_None)."""
    vals = np.ascontiguousarray(np.asarray(vals))
    n = vals.size
    dt = vals.dtype
    if validity is None:
        fill, ov = vals, None
    else:
        m = np.ascontiguousarray(np.asarray(validity, dtype=bool))
        fill = np.where(m, vals, dt.type(0))
        ov = m.copy()
    if dt == np.dtype(np.int32):
        acc = np.cumsum(fill.astype(np.int64, copy=False), dtype=np.int64)
        w = acc % np.int64(2 ** 32)
        out = np.where(w >= np.int64(2 ** 31),
                       w - np.int64(2 ** 32), w).astype(np.int32)
    else:
        out = np.cumsum(fill.astype(dt, copy=False), dtype=dt)
    return np.ascontiguousarray(out), (None if ov is None
                                       else np.ascontiguousarray(ov))


def bit_exact(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def float_match(got, ref):
    """Numerical contract: NaN positions exact, else max_abs <= max(atol, rtol*|ref|)."""
    got = np.ascontiguousarray(got)
    ref = np.ascontiguousarray(ref)
    if got.dtype != ref.dtype or got.shape != ref.shape:
        return False
    atol, rtol = _tol(ref.dtype)
    gn, rn = np.isnan(got), np.isnan(ref)
    if not np.array_equal(gn, rn):
        return False
    m = ~rn
    if not np.all(np.isfinite(got[m]) | np.isnan(got[m])):
        pass
    diff = np.abs(got[m].astype(np.float64) - ref[m].astype(np.float64))
    allow = np.maximum(atol, rtol * np.abs(ref[m].astype(np.float64)))
    # inf-inf of same sign: diff is nan -> treat as match iff signs equal
    both_inf = np.isinf(got[m]) & np.isinf(ref[m]) & (np.sign(got[m]) == np.sign(ref[m]))
    ok = (diff <= allow) | both_inf | np.isnan(diff) & both_inf
    return bool(np.all(ok))


def wasm_cumsum(vals, kind):
    dt = KINDS[kind]
    vals = np.ascontiguousarray(vals, dtype=dt)
    n = vals.size
    src = os.path.join(TMP, f"src.{kind}")
    vals.tofile(src)
    t = time.perf_counter()
    out = subprocess.run(
        [NODE, MJS, WASM, src, kind, str(n), "parity", TMP],
        capture_output=True, text=True, check=True,
    )
    t_ms = (time.perf_counter() - t) * 1e3
    raw = open(os.path.join(TMP, f"cumsum_out.{kind}"), "rb").read()
    got = np.frombuffer(raw, dtype=dt).copy()
    assert got.size == n, (got.size, n)
    return got, t_ms


def gpu_cumsum_i32(vals):
    """Standalone int32 path: host 0-fill is done by caller; scan here."""
    t = time.perf_counter()
    got = GPU.scan_inclusive(np.ascontiguousarray(vals), "int32")
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
    cases.append(("empty/i32", np.zeros(0, np.int32), "i32", None))
    cases.append(("empty/f64", np.zeros(0, np.float64), "f64", None))
    cases.append(("n1/i32", np.array([7], np.int32), "i32", None))
    cases.append(("n1/f32-nan", np.array([np.nan], np.float32), "f32", None))
    cases.append(("int-minmax", np.array([2**31 - 1, 1, -2**31, -1, 2**31 - 1],
                                         np.int32), "i32", None))
    cases.append(("int-wrap-bulk", np.full(16, 2**30, np.int32), "i32", None))
    cases.append(("all-invalid/i32", np.array([1, 2, 3, 4], np.int32), "i32",
                 np.array([False] * 4)))
    cases.append(("all-invalid/f64", np.array([1.5, np.nan, 3.0], np.float64),
                  "f64", np.array([False] * 3)))
    cases.append(("alternating/i32", np.arange(1, 9, dtype=np.int32), "i32",
                 np.array([True, False] * 4)))
    cases.append(("alternating/f32", np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0],
                                              np.float32), "f32",
                 np.array([True, False, True, True, False, True])))
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
        cases.append((f"bulk/{kind}", vals, kind, None))
        cases.append((f"bulk/{kind}/gaps", vals, kind, gaps))

    # --- GPU probe: device present? (honest gap if not) ---
    try:
        _g, _t = gpu_cumsum_i32(np.array([1, 2, 3], np.int32))
        gpu_live = True
    except Exception as e:  # noqa: BLE001
        gpu_live = False
        print(f"NOTE gpu-standalone unavailable: {type(e).__name__}: {e}")

    t_ref = t_nat = 0.0
    a = MAIN["build"](APP).alias
    for label, vals, kind, valid in cases:
        ref, refv = oracle_cumsum(vals, valid)
        dt = KINDS[kind]
        fill = (np.ascontiguousarray(vals, dtype=dt) if valid is None
                else np.where(np.ascontiguousarray(np.asarray(valid, dtype=bool)),
                              np.ascontiguousarray(vals, dtype=dt),
                              np.dtype(dt).type(0)))
        # native (data plane on 0-filled input)
        t = time.perf_counter()
        nat = NAT.cumsum_scatter(fill)
        t_nat += (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        _ = oracle_cumsum(vals, valid)
        t_ref += (time.perf_counter() - t) * 1e3
        exact = bit_exact if kind == "i32" else float_match
        check(f"{label} native==oracle", exact(nat, ref),
              f"dtype={kind} n={vals.size}")
        # wasm (data plane on 0-filled input)
        w, _ = wasm_cumsum(fill, kind)
        check(f"{label} wasm==oracle", exact(w, ref), f"dtype={kind} n={vals.size}")
        # gpu: int32 standalone scan vs oracle; floats explicit gap
        if kind == "i32":
            if gpu_live:
                try:
                    g, _ = gpu_cumsum_i32(fill)
                    check(f"{label} gpu==oracle", bit_exact(g, ref),
                          f"dtype={kind} n={vals.size}")
                except Exception as e:  # noqa: BLE001
                    check(f"{label} gpu==oracle", False,
                          f"{type(e).__name__}: {e}")
            else:
                check(f"{label} gpu-gap", True, "no-device honest gap")
        else:
            try:
                GPU.scan_inclusive(fill, "float32" if kind == "f32" else "float64")
                check(f"{label} gpu-float-explicit-gap", False, "no error raised")
            except (ValueError, KeyError):
                check(f"{label} gpu-float-explicit-gap", True,
                      "ValueError as contracted (no float scan lanes)")
        # Validity carry through the real IR CPU path (builder).
        kw = {} if valid is None else {"validity": [int(v) for v in valid]}
        jobs = [a["ir_series"]("s", np.ascontiguousarray(vals, dtype=dt),
                               {"i32": "int32", "f32": "float32",
                                "f64": "float64"}[kind], **kw),
                a["ir_cumsum"]("h", "s")]
        bufs = a["cpu_execute"](a["optimize"](a["compile"](jobs))["nodes"])
        got, gotv = np.ascontiguousarray(bufs["h"]), bufs.get("h#validity")
        vok = (refv is None and gotv is None) or (
            refv is not None and gotv is not None
            and [bool(v) for v in gotv] == [bool(v) for v in refv])
        check(f"{label} ir-data==oracle", exact(got, ref), f"n={vals.size}")
        check(f"{label} ir-validity-carry", vok,
              "no-sidecar" if refv is None else "per-row-copy")

    print(f"\nstages ms: oracle_total={t_ref:.3f} native_total={t_nat:.3f} "
          f"cases={len(cases)} gpu_live={gpu_live}")
    json.dump({"seed": 42, "cases": len(cases),
               "oracle_ms": t_ref, "native_ms": t_nat,
               "gpu_live": gpu_live,
               "pass": not fails, "fails": fails},
              open(os.path.join(RES, "parity_cumsum.json"), "w"), indent=1)
    shutil.rmtree(TMP, ignore_errors=True)
    print("PASS" if not fails else f"FAIL {fails}")
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
