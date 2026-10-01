# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Cumsum benchmark: NumPy oracle vs Rust-native(ctypes) vs WASM(Node)
vs GPU(WGSL scan_inclusive, int32 only) at 100K/1M, seed 42.

Stages ms + throughput + memory. WASM via wasm_cumsum.mjs bench mode
(5 warm + 10 measured, median of the in-WASM kernel call only).
GPU int32 only (f32/f64 are explicit gaps, asserted as explicit-error).
GPU split: total end-to-end (incl. H2D/D2H) vs loopback transfer
(upload+download of the same byte size, no compute) vs
kernel = total - transfer. Cold = first call per backend (device
init + pipeline compile for GPU), warm = median after 5 (numpy/native)
or driver-internal warmup. Int bit-exact asserted (tobytes); float
within the conformance profile (imported check, no hardcoded thresholds).
Writes results/bench_cumsum.json.
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
MJS = os.path.join(ROOT, "tools", "wasm_cumsum.mjs")
NODE = shutil.which("node") or r"C:\Program Files\nodejs\node.exe"
TMP = os.path.join(RES, ".cumsum-bench")
os.makedirs(TMP, exist_ok=True)
DEV = os.path.normpath(os.path.join(ROOT, "..", ".."))
APP = os.path.join(DEV, "numfast")
PROFILE = os.path.join(APP, "specs-rebuilt", "conformance-profile.toml")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NAT = _load("nf_native_cpu_b",
            os.path.join(APP, "src", "Drivers", "CPU", "_lib", "native_cpu.py"))
GPU = _load("nf_gpu_b",
            os.path.join(APP, "src", "Drivers", "GPU", "_lib", "gpu.py"))

with open(PROFILE, "rb") as f:
    _PROF = tomllib.load(f)

KINDS = {"i32": (np.int32, 4), "f32": (np.float32, 4), "f64": (np.float64, 8)}


def _tol(dt):
    key = "f32" if np.dtype(dt) == np.dtype(np.float32) else "f64"
    sec = _PROF["tolerance"][key]
    return sec["atol"], sec["rtol"]


def oracle(vals):
    vals = np.ascontiguousarray(vals)
    if vals.dtype == np.dtype(np.int32):
        acc = np.cumsum(vals.astype(np.int64, copy=False), dtype=np.int64)
        w = acc % np.int64(2 ** 32)
        return np.ascontiguousarray(np.where(
            w >= np.int64(2 ** 31), w - np.int64(2 ** 32), w).astype(np.int32))
    return np.ascontiguousarray(np.cumsum(vals, dtype=vals.dtype))


def float_ok(got, ref):
    got = np.ascontiguousarray(got)
    ref = np.ascontiguousarray(ref)
    atol, rtol = _tol(ref.dtype)
    gn, rn = np.isnan(got), np.isnan(ref)
    if not np.array_equal(gn, rn):
        return False
    m = ~rn
    diff = np.abs(got[m].astype(np.float64) - ref[m].astype(np.float64))
    allow = np.maximum(atol, rtol * np.abs(ref[m].astype(np.float64)))
    both_inf = np.isinf(got[m]) & np.isinf(ref[m]) & (np.sign(got[m]) == np.sign(ref[m]))
    return bool(np.all((diff <= allow) | both_inf))


def med(ts):
    ts = sorted(ts)
    return ts[len(ts) >> 1]


def bench_numpy(vals, runs=10):
    t0 = time.perf_counter()
    _ = oracle(vals)  # cold (first call, page faults included)
    cold = (time.perf_counter() - t0) * 1e3
    for _ in range(5):
        _ = oracle(vals)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        out = oracle(vals)
        ts.append((time.perf_counter() - t) * 1e3)
    return cold, med(ts), out


def bench_native(vals, runs=10):
    t0 = time.perf_counter()
    _ = NAT.cumsum_scatter(vals)  # cold (DLL probe + first call)
    cold = (time.perf_counter() - t0) * 1e3
    for _ in range(5):
        NAT.cumsum_scatter(vals)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        out = NAT.cumsum_scatter(vals)
        ts.append((time.perf_counter() - t) * 1e3)
    return cold, med(ts), out


def bench_wasm(vals, kind):
    src = os.path.join(TMP, f"bench.{kind}")
    np.ascontiguousarray(vals).tofile(src)
    # cold: fresh node process, first WASM call (compile included)
    t0 = time.perf_counter()
    r0 = subprocess.run(
        [NODE, MJS, WASM, src, kind, str(vals.size), "parity", TMP],
        capture_output=True, text=True, check=True)
    cold_file_ms = (time.perf_counter() - t0) * 1e3
    cold_kernel_ms = json.loads(r0.stdout.strip())["ms"]
    out = subprocess.run(
        [NODE, MJS, WASM, src, kind, str(vals.size), "bench"],
        capture_output=True, text=True, check=True)
    kernel_med = json.loads(out.stdout.strip())["median_ms"]
    return cold_file_ms, cold_kernel_ms, kernel_med


def bench_gpu_transfer(nbytes, runs=10):
    """Loopback H2D+D2H of nbytes (no compute): the transfer baseline."""
    probe = np.zeros(nbytes, dtype=np.uint8)
    t0 = time.perf_counter()
    b = GPU.r_upload(probe)
    _ = GPU.r_download(b, np.uint8, nbytes)
    cold = (time.perf_counter() - t0) * 1e3
    for _ in range(2):
        b = GPU.r_upload(probe)
        _ = GPU.r_download(b, np.uint8, nbytes)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        b = GPU.r_upload(probe)
        _ = GPU.r_download(b, np.uint8, nbytes)
        ts.append((time.perf_counter() - t) * 1e3)
    return cold, med(ts)


def bench_gpu(vals, runs=10):
    t0 = time.perf_counter()
    out = GPU.scan_inclusive(np.ascontiguousarray(vals), "int32")  # cold
    cold = (time.perf_counter() - t0) * 1e3
    for _ in range(2):
        GPU.scan_inclusive(np.ascontiguousarray(vals), "int32")
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        out = GPU.scan_inclusive(np.ascontiguousarray(vals), "int32")
        ts.append((time.perf_counter() - t) * 1e3)
    return cold, med(ts), np.ascontiguousarray(out)


def main():
    rng = np.random.default_rng(42)
    out = {"seed": 42, "cases": []}
    for n in (100_000, 1_000_000):
        for kind, (dt, B) in KINDS.items():
            if dt == np.int32:
                vals = np.ascontiguousarray(
                    rng.integers(-2**30, 2**30, size=n, dtype=np.int64).astype(dt))
            else:
                vals = np.ascontiguousarray(rng.normal(0, 1000, size=n).astype(dt))
            mem_mb = n * B * 2 / 2**20
            np_cold, np_med, ref = bench_numpy(vals)
            nat_cold, nat_med, nat = bench_native(vals)
            exact = (nat.tobytes() == ref.tobytes()) if kind == "i32" \
                else float_ok(nat, ref)
            assert exact, (n, kind, "native")
            w_cold_file, w_cold_kern, wasm_med = bench_wasm(vals, kind)
            row = {"n": n, "kind": kind, "mem_mb": round(mem_mb, 2),
                   "numpy_cold_ms": round(np_cold, 3), "numpy_ms": round(np_med, 3),
                   "native_cold_ms": round(nat_cold, 3), "native_ms": round(nat_med, 3),
                   "wasm_cold_process_ms": round(w_cold_file, 3),
                   "wasm_cold_kernel_ms": round(w_cold_kern, 3),
                   "wasm_ms": round(wasm_med, 3),
                   "gb_s": {k: round(n * B / 2**30 / (m / 1e3), 2)
                            for k, m in (("numpy", np_med), ("native", nat_med),
                                         ("wasm", wasm_med))}}
            if kind == "i32":
                t_cold, t_med = bench_gpu_transfer(n * B)
                g_cold, g_med, g = bench_gpu(vals)
                assert g.tobytes() == ref.tobytes(), (n, kind, "gpu")
                kern = max(g_med - t_med, 0.0)
                row["gpu_cold_ms"] = round(g_cold, 3)
                row["gpu_total_ms"] = round(g_med, 3)
                row["gpu_transfer_ms"] = round(t_med, 3)
                row["gpu_transfer_cold_ms"] = round(t_cold, 3)
                row["gpu_kernel_ms"] = round(kern, 3)
                row["gb_s"]["gpu_total"] = round(n * B / 2**30 / (g_med / 1e3), 2)
                row["gb_s"]["gpu_kernel"] = round(n * B / 2**30 / (kern / 1e3), 2) \
                    if kern > 0 else None
                gpu_s = (f"gpu_total={g_med:.3f}ms (kernel~{kern:.3f} transfer~{t_med:.3f} "
                         f"cold={g_cold:.3f})")
            else:
                try:
                    GPU.scan_inclusive(vals, "float32" if kind == "f32" else "float64")
                    raise SystemExit(f"gpu-float-{kind}: no error raised (gap broken)")
                except (ValueError, KeyError):
                    row["gpu_total_ms"] = "explicit-gap(no-float-scan-lanes)"
                gpu_s = "gpu=explicit-gap"
            out["cases"].append(row)
            print(f"n={n} {kind}: numpy={np_med:.3f}ms(cold {np_cold:.3f}) "
                  f"native={nat_med:.3f}ms(cold {nat_cold:.3f}) "
                  f"wasm={wasm_med:.3f}ms(cold-kern {w_cold_kern:.3f}/proc {w_cold_file:.3f}) "
                  f"{gpu_s} mem={mem_mb:.1f}MB "
                  f"{'EXACT' if kind == 'i32' else 'PROFILE-TOL'}")
    json.dump(out, open(os.path.join(RES, "bench_cumsum.json"), "w"), indent=1)
    shutil.rmtree(TMP, ignore_errors=True)
    print("BENCH-PASS")


if __name__ == "__main__":
    main()
