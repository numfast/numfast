# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Map full-port parity: NumPy reference vs Rust-native(ctypes) vs
WASM(via Node wasm_map.mjs) vs GPU(WGSL map_elem), plus IR validity-carry.

Contract (frozen public semantic = CPU driver `op == "map"` branch):
add/sub/mul (int wrap, out==in dtype), div/pow (f64 round-trip; int
rint+astype with the x86 cvttsd2si rule, f32 widens to f64), floor_div/mod
(int Python-floor with sign correction, //0->0; float computed wide).
Validity = AND of input sidecars (DELTA-3), values at invalid rows still
computed. pow array-exponent rejected (scalar-exp only, spec 01).

Rules: integer lanes bit-exact (tobytes); float non-pow exact
(array_equal equal_nan, max|diff| reported, must be 0); float pow under
the conformance-profile tolerance (f32 1e-5, f64 1e-12); out dtypes must
match the reference (widening is contract: f32 div->f64, int+float
floor/mod->f64). GPU covers i32 add/sub/mul/floor_div/mod + f32
add/sub/mul only -- everything else asserts the explicit ValueError gap.

Seed 42 everywhere. Prints table + stage breakdown, writes
results/parity_map.json. Exit 0 = PASS.
"""
import importlib.util
import json
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
WASM = os.path.join(ROOT, "tools", "numfast_native.wasm")
MJS = os.path.join(ROOT, "tools", "wasm_map.mjs")
NODE = shutil.which("node") or r"C:\Program Files\nodejs\node.exe"
TMP = os.path.join(RES, ".map-tmp")
os.makedirs(TMP, exist_ok=True)
DEV = os.path.normpath(os.path.join(ROOT, "..", ".."))  # dev-env root
APP = os.path.join(DEV, "numfast")  # numfast package root (builder APP_DIR)

with open(os.path.join(APP, "specs-rebuilt", "conformance-profile.toml"), "rb") as f:
    PROF = tomllib.load(f)
TOL = {"f32": (PROF["tolerance"]["f32"]["atol"], PROF["tolerance"]["f32"]["rtol"]),
       "f64": (PROF["tolerance"]["f64"]["atol"], PROF["tolerance"]["f64"]["rtol"])}

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
OPCODE = {"add": 0, "sub": 1, "mul": 2, "div": 3, "pow": 4,
          "floor_div": 5, "mod": 6}
DTNAME = {"i32": "int32", "f32": "float32", "f64": "float64"}

# GPU-covered (fn, kind): i32 add/sub/mul/floor_div/mod + f32 add/sub/mul.
GPU_OK = {(f, "i32") for f in ("add", "sub", "mul", "floor_div", "mod")}
GPU_OK |= {(f, "f32") for f in ("add", "sub", "mul")}


def bit_exact(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def float_exact(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if not bool(np.array_equal(a, b, equal_nan=True)):
        return False
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return bool(np.all((d == 0.0) | np.isnan(d)))


def float_close(a, b, key):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    atol, rtol = TOL[key]
    if a.size == 0:
        return True
    ok = np.abs(a.astype(np.float64) - b.astype(np.float64)) <= (
        atol + rtol * np.abs(b.astype(np.float64)))
    nan_ok = np.isnan(a) == np.isnan(b)
    # inf-inf subtracts to nan (ok=False there): same-sign infs are exact.
    same_inf = (np.isinf(a) & np.isinf(b)
                & (np.sign(a) == np.sign(b)))
    return bool(np.all(((ok | same_inf) & nan_ok)
                       | (np.isnan(a) & np.isnan(b))))


def wasm_map(a_vals, b_vals, kind, op, scalar_kind, scalar_val):
    dt = KINDS[kind]
    a_vals = np.ascontiguousarray(a_vals, dtype=dt)
    n = a_vals.size
    a_path = os.path.join(TMP, f"a.{kind}")
    a_vals.tofile(a_path)
    if scalar_kind == "arr":
        b_vals = np.ascontiguousarray(b_vals, dtype=dt)
        b_path = os.path.join(TMP, f"b.{kind}")
        b_vals.tofile(b_path)
        sk, sv = "arr", "0"
    else:
        b_path = "-"
        sk, sv = scalar_kind, repr(float(scalar_val) if scalar_kind == "f64"
                                   else int(scalar_val))
    t = time.perf_counter()
    out = subprocess.run(
        [NODE, MJS, WASM, a_path, b_path, kind, str(op), str(n),
         sk, sv, "parity", TMP],
        capture_output=True, text=True, check=True,
    )
    t_ms = (time.perf_counter() - t) * 1e3
    out_kind = kind
    if (kind == "f32" and op in (3, 4)) or (
            kind == "i32" and scalar_kind == "f64" and op in (5, 6)):
        out_kind = "f64"
    raw = open(os.path.join(TMP, f"map_out.{out_kind}"), "rb").read()
    got = np.frombuffer(raw, dtype=KINDS[out_kind]).copy()
    assert got.size == n, (got.size, n)
    return got, t_ms


def gpu_map(a_vals, b_vals, fn, kind, scalar_kind):
    dt = DTNAME[kind]
    t = time.perf_counter()
    if scalar_kind == "arr":
        got = GPU.map_elem(np.ascontiguousarray(a_vals),
                           np.ascontiguousarray(b_vals), fn, dt)
    else:
        got = GPU.map_elem(np.ascontiguousarray(a_vals), b_vals, fn, dt)
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

    a = MAIN["build"](APP).alias

    # --- vectors ---
    cases = []
    for kind in ("i32", "f32", "f64"):
        dt = KINDS[kind]
        cases.append((f"empty/{kind}", np.zeros(0, dt), kind, "arr",
                      np.zeros(0, dt), None))
        cases.append((f"empty/{kind}/scal", np.zeros(0, dt), kind,
                      "f64" if kind != "i32" else "i32",
                      2 if kind == "i32" else 2.5, None))
    cases.append(("n1/i32", np.array([7], np.int32), "i32", "arr",
                  np.array([3], np.int32), None))
    cases.append(("n1/f32", np.array([7.5], np.float32), "f32", "f64",
                  2.0, None))
    ie = np.array([2**31 - 1, -2**31, -1, 0, 1, 2**30], dtype=np.int32)
    idiv = np.array([-1, 0, 1, 2, -2, 3], dtype=np.int32)
    cases.append(("int-edges/div-arr", ie, "i32", "arr", idiv, None))
    cases.append(("int-edges/floordiv-arr", ie, "i32", "arr", idiv, None))
    cases.append(("int-edges/mod-arr", ie, "i32", "arr", idiv, None))
    cases.append(("int-edges/mul-arr", ie, "i32", "arr", idiv, None))
    for s, sk in ((0, "i32"), (1, "i32"), (-1, "i32"), (2, "i32"), (2.5, "f64")):
        cases.append((f"int-edges/scal-{s}", ie, "i32", sk, s, None))

    n = 100_000
    vi = rng.integers(-10_000, 10_000, size=n).astype(np.int32)
    vd = rng.integers(-500, 500, size=n).astype(np.int32)
    vd0 = vd.copy()
    vd0[::997] = 0  # zero divisors stay: //0->0 is contract
    vf = rng.normal(0, 1000, size=n).astype(np.float32)
    vf[::97] = np.nan
    vf[::211] = np.inf
    vf[::313] = -np.inf
    vg = rng.normal(0, 100, size=n).astype(np.float32)
    vg[::101] = np.nan
    vd_ = rng.normal(0, 1000, size=n)
    vd_[::101] = np.nan
    vd_[::211] = np.inf
    ve_ = rng.normal(0, 10, size=n)
    gaps = (rng.integers(0, 2, size=n) == 1)
    bulk = {"i32": (vi, vd, vd0), "f32": (vf, vg, vg), "f64": (vd_, ve_, ve_)}
    FN = ["add", "sub", "mul", "div", "pow", "floor_div", "mod"]
    for kind in ("i32", "f32", "f64"):
        vv, bb, bb0 = bulk[kind]
        for fn in FN:
            if fn == "pow":
                continue  # scalar-exp only; covered below
            cases.append((f"bulk/{kind}/{fn}/arr", vv, kind, "arr", bb, None))
            cases.append((f"bulk/{kind}/{fn}/zero-div", vv, kind, "arr", bb0, None))
            cases.append((f"bulk/{kind}/{fn}/ints", vv, kind, "i32", 2, None))
            cases.append((f"bulk/{kind}/{fn}/neg", vv, kind, "i32", -3, None))
            fs = 2.5 if kind == "i32" else 0.5
            cases.append((f"bulk/{kind}/{fn}/floats", vv, kind, "f64", fs, None))
            cases.append((f"bulk/{kind}/{fn}/gaps", vv, kind, "i32", 2, gaps))
        for s, sk in ((2, "i32"), (2.5, "f64") if kind == "i32" else (0.5, "f64"),):
            cases.append((f"bulk/{kind}/pow/scal-{s}", vv, kind, sk, s, None))
    # 1M smoke (add + floor_div + div, all kinds, array form)
    n1m = 1_000_000
    for kind in ("i32", "f32", "f64"):
        dt = KINDS[kind]
        if kind == "i32":
            x = rng.integers(-10_000, 10_000, size=n1m).astype(dt)
            y = rng.integers(-500, 500, size=n1m).astype(dt)
            y[y == 0] = 7
        else:
            x = rng.normal(0, 1000, size=n1m).astype(dt)
            y = rng.normal(0, 100, size=n1m).astype(dt)
        for fn in ("add", "div", "floor_div"):
            cases.append((f"1M/{kind}/{fn}", x, kind, "arr", y, None))

    t_ref = t_nat = 0.0
    for label, vals, kind, sk, other, valid in cases:
        fn = label.split("/")[2] if label.startswith(("bulk/", "1M/")) else {
            "empty": "add", "n1": "add", "int-edges": None}[label.split("/")[0]]
        if fn is None:
            fn = label.split("/")[1].split("-")[0]
            fn = {"div": "div", "floordiv": "floor_div", "mod": "mod",
                  "mul": "mul", "scal": "add"}[fn]
        op = OPCODE[fn]
        oth = other if sk == "arr" else other
        t = time.perf_counter()
        ref = NAT._fb_map(vals, fn, oth if sk == "arr" else other)
        t_ref += (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        nat = NAT.map_scatter(vals, fn, oth if sk == "arr" else other)
        t_nat += (time.perf_counter() - t) * 1e3
        is_int = np.asarray(ref).dtype == np.dtype(np.int32)
        is_pow_float = fn == "pow" and np.asarray(ref).dtype.kind == "f"
        if is_int:
            check(f"{label} native==ref", bit_exact(nat, ref),
                  f"dtype={np.asarray(ref).dtype} n={np.asarray(vals).size}")
        elif is_pow_float:
            key = "f32" if np.asarray(ref).dtype == np.dtype(np.float32) else "f64"
            check(f"{label} native~=ref", float_close(nat, ref, key),
                  f"tol-{key} n={np.asarray(vals).size}")
        else:
            check(f"{label} native==ref", float_exact(nat, ref),
                  f"dtype={np.asarray(ref).dtype} n={np.asarray(vals).size}")
        # WASM (skip int-lane out-of-range int scalars: reference-owned)
        skip_wasm = (kind == "i32" and sk == "i32"
                     and not (-2 ** 31 <= int(other) <= 2 ** 31 - 1))
        if not skip_wasm and not (fn == "pow" and sk == "arr"):
            if kind == "i32" and fn in ("floor_div", "mod") and sk == "f64":
                sk_w, sv_w = "f64", float(other)  # device widens like native
            elif sk == "arr":
                sk_w, sv_w = "arr", 0
            elif kind == "i32" and sk == "i32":
                sk_w, sv_w = "i32", int(other)
            else:
                sk_w, sv_w = "f64", float(other)
            w, _ = wasm_map(vals, other if sk == "arr" else None,
                            kind, op, sk_w, sv_w)
            if is_int:
                check(f"{label} wasm==ref", bit_exact(w, ref), f"n={np.asarray(vals).size}")
            elif is_pow_float:
                key = "f32" if np.asarray(ref).dtype == np.dtype(np.float32) else "f64"
                check(f"{label} wasm~=ref", float_close(w, ref, key), f"tol-{key}")
            else:
                check(f"{label} wasm==ref", float_exact(w, ref), f"n={np.asarray(vals).size}")
        # GPU: covered -> parity; gaps -> explicit ValueError.
        # i32 lane + float scalar is ALWAYS a GPU gap (float scalars ride
        # float64 on CPU, not expressible in WGSL); int scalars covered.
        gpu_covered = (fn, kind) in GPU_OK and not (
            kind == "i32" and sk == "f64")
        if gpu_covered:
            g, _ = gpu_map(vals, other if sk == "arr" else other, fn, kind, sk)
            if is_int:
                check(f"{label} gpu==ref", bit_exact(g, ref), f"n={np.asarray(vals).size}")
            else:
                check(f"{label} gpu==ref", float_exact(g, ref), f"n={np.asarray(vals).size}")
        else:
            try:
                gpu_map(vals, other if sk == "arr" else other, fn, kind, sk)
                check(f"{label} gpu-gap-explicit", False, "no error raised")
            except ValueError:
                check(f"{label} gpu-gap-explicit", True, "ValueError as contracted")
        # IR validity-carry through the real CPU path (builder).
        kw = {} if valid is None else {"validity": [int(v) for v in valid]}
        jobs = [a["ir_series"]("s", np.ascontiguousarray(vals), DTNAME[kind], **kw)]
        if sk == "arr":
            jobs.append(a["ir_series"]("o", np.ascontiguousarray(other), DTNAME[kind]))
            jobs.append(a["ir_map"]("m", "s", fn, "o"))
            exp_valid = None if valid is None else valid
        else:
            jobs.append(a["ir_map"]("m", "s", fn, other))
            exp_valid = None if valid is None else valid
        bufs = a["cpu_execute"](a["optimize"](a["compile"](jobs))["nodes"])
        got, gotv = np.ascontiguousarray(bufs["m"]), bufs.get("m#validity")
        if is_pow_float:
            key = "f32" if got.dtype == np.dtype(np.float32) else "f64"
            check(f"{label} ir-data~=ref", float_close(got, ref, key), f"tol-{key}")
        elif is_int:
            check(f"{label} ir-data==ref", bit_exact(got, ref), f"n={np.asarray(vals).size}")
        else:
            check(f"{label} ir-data==ref", float_exact(got, ref), f"n={np.asarray(vals).size}")
        if exp_valid is None:
            check(f"{label} ir-validity", gotv is None, "no-sidecar")
        else:
            vok = (gotv is not None and
                   [bool(v) for v in gotv] == [bool(v) for v in exp_valid])
            check(f"{label} ir-validity", vok, "AND-carry")

    # pow array-exponent rejected on every path (scalar-exp only, spec 01).
    for path, thunk in (
        ("ref", lambda: NAT._fb_map(np.array([1, 2], np.int32), "pow",
                                    np.array([2, 3], np.int32))),
        ("native", lambda: NAT.map_scatter(np.array([1, 2], np.int32), "pow",
                                           np.array([2, 3], np.int32))),
    ):
        try:
            thunk()
            check(f"pow-array-exp-{path}-rejected", False, "no error")
        except ValueError as e:
            check(f"pow-array-exp-{path}-rejected", "scalar-exp" in str(e), "scalar-exp")
    jobs = [a["ir_series"]("x", np.array([1, 2], np.int32), "int32"),
            a["ir_series"]("e", np.array([2, 3], np.int32), "int32"),
            a["ir_map"]("m", "x", "pow", "e")]
    try:
        a["cpu_execute"](a["optimize"](a["compile"](jobs))["nodes"])
        check("pow-array-exp-ir-rejected", False, "no error")
    except ValueError as e:
        check("pow-array-exp-ir-rejected", "scalar-exp" in str(e), "scalar-exp")
    try:
        GPU.map_elem(np.array([1, 2], np.int32), np.array([2, 3], np.int32),
                     "pow", "int32")
        check("pow-array-exp-gpu-rejected", False, "no error")
    except ValueError:
        check("pow-array-exp-gpu-rejected", True, "ValueError")
    # unknown fn rejected at IR construction (no backend involved).
    try:
        a["ir_map"]("m", "s", "hypot", 1)
        check("unknown-fn-rejected", False, "no error")
    except ValueError as e:
        check("unknown-fn-rejected", "unknown map fn" in str(e), "unknown map fn")
    # Runtime-level explicit gap: backend='gpu' on an i32 div graph raises
    # (no silent fallback), covered add graph executes on GPU.
    try:
        gadd = a["optimize"](a["compile"](
            [a["ir_series"]("s", np.array([1, 2, 3], np.int32), "int32"),
             a["ir_map"]("m", "s", "add", 1)]))
        res = a["evaluate"](gadd, "gpu", 3)
        check("runtime-gpu-add", res["execution_info"]["actual"] == "gpu"
              and [int(v) for v in res["result"]] == [2, 3, 4], "actual=gpu")
    except Exception as e:
        check("runtime-gpu-add", False, str(e)[:120])
    try:
        gdiv = a["optimize"](a["compile"](
            [a["ir_series"]("s", np.array([1, 2, 3], np.int32), "int32"),
             a["ir_map"]("m", "s", "div", 2)]))
        a["evaluate"](gdiv, "gpu", 3)
        check("runtime-gpu-div-explicit", False, "no error (silent route?)")
    except (ValueError, RuntimeError) as e:
        check("runtime-gpu-div-explicit", True, str(e)[:100])

    print(f"\nstages ms: oracle_total={t_ref:.3f} native_total={t_nat:.3f} "
          f"cases={len(cases)}")
    json.dump({"seed": 42, "cases": len(cases),
               "checks": len(rows),
               "oracle_ms": t_ref, "native_ms": t_nat,
               "pass": not fails, "fails": fails},
              open(os.path.join(RES, "parity_map.json"), "w"), indent=1)
    shutil.rmtree(TMP, ignore_errors=True)
    print("PASS" if not fails else f"FAIL {fails}")
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
