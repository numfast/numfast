# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Unique full-port parity: NumPy oracle vs CPU(IR) vs Rust-native(ctypes)
vs WASM(via Node wasm_unique.mjs) vs GPU(explicit gap, never silent).

Contract (frozen unique.rs + cpu.py unique_inverse branch): sorted-order
uniq ascending + inv = sorted-position codes (NOT first-appearance);
uniq[inv] == keys on valid rows; invalid rows -> inv -1, excluded from
uniq; empty/all-invalid -> ng=0; chunkable=false (CPU+GPU hints); GPU
has no unique_inverse lane -> explicit ValueError (spec 14). IR/API/
Planner/TEXT/Join duplicate-check untouched; no new primitives.

Seed 42 everywhere. Prints table, writes results/parity_unique.json.
Exit 0 = PASS.
"""
import importlib.util
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
MJS = os.path.join(ROOT, "tools", "wasm_unique.mjs")
NODE = shutil.which("node") or r"C:\Program Files\nodejs\node.exe"
TMP = os.path.join(RES, ".unique-tmp")
os.makedirs(TMP, exist_ok=True)
DEV = os.path.normpath(os.path.join(ROOT, "..", ".."))  # dev-env root
APP = os.path.join(DEV, "numfast")  # numfast package root (builder APP_DIR)

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

KINDS = {"i32": np.int32, "i64": np.int64}
SEED = 42


def oracle_unique(vals, validity=None):
    """Independent oracle: (uniq, inv_full, ng).

    uniq sorted ascending over valid rows (keys dtype); inv_full int64
    with -1 on invalid rows; ng = distinct valid count.
    """
    vals = np.ascontiguousarray(np.asarray(vals))
    n = vals.size
    if validity is None:
        u, iv = np.unique(vals, return_inverse=True)
        return (np.ascontiguousarray(u),
                np.ascontiguousarray(iv.astype(np.int64)), int(u.size))
    m = np.ascontiguousarray(np.asarray(validity, dtype=bool))
    assert m.size == n
    u, iv = np.unique(vals[m], return_inverse=True)
    inv = np.full(n, -1, dtype=np.int64)
    inv[m] = iv.astype(np.int64)
    return np.ascontiguousarray(u), np.ascontiguousarray(inv), int(u.size)


def wasm_unique(vals, kind):
    dt = KINDS[kind]
    vals = np.ascontiguousarray(vals, dtype=dt)
    n = vals.size
    src = os.path.join(TMP, f"src.{kind}")
    vals.tofile(src)
    out = subprocess.run(
        [NODE, MJS, WASM, src, kind, str(n), "parity", TMP],
        capture_output=True, text=True, check=True,
    )
    meta = json.loads(out.stdout.strip())
    ng = int(meta["ng"])
    assert ng >= 0, meta
    raw_u = open(os.path.join(TMP, f"unique_out.{kind}"), "rb").read()
    raw_i = open(os.path.join(TMP, "unique_inv.i32"), "rb").read()
    got_u = np.frombuffer(raw_u, dtype=dt).copy()
    got_i = np.frombuffer(raw_i, dtype=np.int32).copy().astype(np.int64)
    assert got_u.size == ng and got_i.size == n, (got_u.size, got_i.size, ng, n)
    return got_u, got_i, ng


def main():
    rng = np.random.default_rng(SEED)
    fails = []
    rows = []

    def check(label, cond, detail=""):
        rows.append((label, cond, detail))
        print(("PASS " if cond else "FAIL ") + label + ((" | " + detail) if detail else ""))
        if not cond:
            fails.append(label)

    I32MIN, I32MAX = -(2 ** 31), 2 ** 31 - 1
    I64MIN, I64MAX = -(2 ** 63), 2 ** 63 - 1
    cases = [
        ("empty/i32", np.zeros(0, np.int32), "i32", None),
        ("empty/i64", np.zeros(0, np.int64), "i64", None),
        ("n1/i32", np.array([7], np.int32), "i32", None),
        ("n1/i64", np.array([-5], np.int64), "i64", None),
        ("alleq/i32", np.full(1000, 7, np.int32), "i32", None),
        ("duplicates/i32", np.array([5, -3, 5, 0, -3, 7, 0, 0], np.int32), "i32", None),
        ("duplicates/i64", np.array([9, -1, 9, 3, -1, 3], np.int64), "i64", None),
        ("sorted/i32", np.arange(-50, 50, dtype=np.int32), "i32", None),
        ("reverse/i64", np.arange(99, -1, -1, dtype=np.int64), "i64", None),
        # first-appearance order != sorted order: codes must be sorted-position
        ("first-ne-sorted/i32", np.array([3, 1, 2, 1, 3, 2, 0], np.int32), "i32", None),
        ("first-ne-sorted/i64", np.array([30, 10, 20, 10, 30], np.int64), "i64", None),
        ("minmax/i32", np.array([I32MIN, I32MAX, 0, -1, 1, I32MIN, I32MAX], np.int32), "i32", None),
        ("minmax/i64", np.array([I64MIN, I64MAX, 0, -1, 1, I64MIN], np.int64), "i64", None),
        ("all-invalid/i32", np.array([1, 2, 3, 4], np.int32), "i32",
         np.array([False] * 4)),
        ("all-invalid/i64", np.array([1, 2, 3], np.int64), "i64",
         np.array([False] * 3)),
        ("alternating/i32", np.arange(1, 9, dtype=np.int32), "i32",
         np.array([True, False] * 4)),
        ("alternating/i64", np.array([4, 1, 4, 1, 2, 2, 9, 9], np.int64), "i64",
         np.array([True, False] * 4)),
    ]
    # seeded bulk: random + fewuniq + validity gaps, both dtypes
    for kind, dt in (("i32", np.int32), ("i64", np.int64)):
        lo = I32MIN if dt == np.int32 else I64MIN
        hi = I32MAX if dt == np.int32 else I64MAX
        bulk = rng.integers(lo, hi, size=10_000).astype(dt)
        bulk[0], bulk[1] = dt(lo), dt(hi)
        cases.append((f"bulk-random/{kind}", bulk, kind, None))
        few = rng.integers(-50, 50, size=10_000).astype(dt)
        gaps = rng.integers(0, 2, size=10_000) == 1
        cases.append((f"bulk-fewuniq-gaps/{kind}", few, kind, gaps))
    # 100K + 1M spot parity (data once per cell, seed 42 stream continues)
    for n, tag in ((100_000, "100K"), (1_000_000, "1M")):
        for kind, dt in (("i32", np.int32), ("i64", np.int64)):
            v = rng.integers(-500_000, 500_000, size=n).astype(dt)
            cases.append((f"spot-{tag}/{kind}", v, kind, None))

    check("native available", NAT.unique_available(), NAT.why())
    t_ref = t_nat = 0.0
    a = MAIN["build"](APP).alias
    for label, vals, kind, valid in cases:
        dt = KINDS[kind]
        vals = np.ascontiguousarray(vals, dtype=dt)
        n = vals.size
        t = time.perf_counter()
        ref_u, ref_inv, ref_ng = oracle_unique(vals, valid)
        t_ref += (time.perf_counter() - t) * 1e3
        inv_ok = bool(np.array_equal(ref_u[ref_inv[ref_inv >= 0]], vals[ref_inv >= 0])) if ref_ng else True

        # --- IR CPU path (owns validity: inv=-1, ng over valid) ---
        kw = {} if valid is None else {"validity": [int(v) for v in valid]}
        jobs = [a["ir_series"]("s", vals,
                               {"i32": "int32", "i64": "int64"}[kind], **kw),
                a["ir_unique_inverse"]("u", "s")]
        bufs = a["cpu_execute"](a["optimize"](a["compile"](jobs))["nodes"])
        got_u = np.ascontiguousarray(bufs["u"])
        got_inv = np.ascontiguousarray(bufs["u#inv"]).astype(np.int64)
        got_ng = int(bufs["u#ng"])
        check(f"{label} ir-uniq==oracle",
              got_u.dtype == ref_u.dtype and np.array_equal(got_u, ref_u),
              f"ng={got_ng}/{ref_ng} n={n}")
        check(f"{label} ir-inv==oracle", np.array_equal(got_inv, ref_inv),
              f"ng={got_ng} n={n}")
        check(f"{label} ir-ng", got_ng == ref_ng == int(got_u.size),
              f"ng={got_ng}")
        rec = (bool(np.array_equal(got_u[got_inv[got_inv >= 0]],
                                   vals[got_inv >= 0]))
               if got_ng else (got_inv.size == n and bool((got_inv == -1).all() if valid is not None and not np.asarray(valid).any() else True)))
        check(f"{label} ir-invariant", rec and inv_ok, f"n={n}")

        # --- native ctypes (data plane; validity subset fed directly) ---
        if valid is None:
            feed, eu, eiv, eng = vals, ref_u, ref_inv, ref_ng
        else:
            m = np.ascontiguousarray(np.asarray(valid, dtype=bool))
            feed = np.ascontiguousarray(vals[m])
            eu, eiv, eng = oracle_unique(feed, None)
        t = time.perf_counter()
        if feed.size == 0:
            nu, ni, be = NAT.unique_inverse(feed)
            nok = (be == "numpy" and nu.size == 0 and ni.size == 0)
        else:
            nu, ni, be = NAT.unique_inverse(feed)
            nok = (be == "native" and nu.dtype == eu.dtype
                   and np.array_equal(nu, eu) and np.array_equal(ni, eiv)
                   and ni.dtype == np.dtype(np.int64)
                   and bool(np.array_equal(nu[ni], feed)))
        t_nat += (time.perf_counter() - t) * 1e3
        check(f"{label} native==oracle", nok,
              f"be={be if feed.size else 'numpy(empty)'} ng={eng} n={n}")

        # --- WASM (data plane on the same feed; empty -> ng=0) ---
        wu, wi, wng = wasm_unique(feed, kind)
        wok = (wu.dtype == eu.dtype and np.array_equal(wu, eu)
               and np.array_equal(wi, eiv)
               and (bool(np.array_equal(wu[wi], feed)) if wng else wi.size == feed.size))
        check(f"{label} wasm==oracle", wok, f"ng={wng}/{eng} n={n}")

    # --- GPU: unique_inverse is CPU-only explicit (never silent) ---
    try:
        jobs = [a["ir_series"]("s", np.array([1, 2, 1], np.int32), "int32"),
                a["ir_unique_inverse"]("u", "s")]
        GPU.gpu_execute_impl(a["optimize"](a["compile"](jobs))["nodes"])
        check("gpu-unique-explicit-gap", False, "no error raised")
    except ValueError as e:
        check("gpu-unique-explicit-gap", "unique_inverse" in str(e),
              f"ValueError: {str(e)[:90]}")
    # --- capability/chunkable gates ---
    cpu_cap = a["cpu_capability"]()
    check("cpu-chunkable-false",
          cpu_cap["chunkable_hints"].get("unique_inverse") is False
          and "unique_inverse" in cpu_cap["ops"], "spec04/14")
    try:
        gcap = GPU.gpu_capability_impl()
        check("gpu-no-unique-op", "unique_inverse" not in gcap["ops"], "spec14")
    except Exception as e:  # noqa: BLE001 -- no-device host still must not claim it
        check("gpu-no-unique-op", True, f"honest gap: {type(e).__name__}")

    print(f"\nstages ms: oracle_total={t_ref:.3f} native_total={t_nat:.3f} "
          f"cases={len(cases)}")
    json.dump({"seed": SEED, "cases": len(cases),
               "oracle_ms": t_ref, "native_ms": t_nat,
               "pass": not fails, "fails": fails},
              open(os.path.join(RES, "parity_unique.json"), "w"), indent=1)
    shutil.rmtree(TMP, ignore_errors=True)
    print("PASS" if not fails else f"FAIL {fails}")
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
