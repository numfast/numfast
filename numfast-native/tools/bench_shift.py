# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Shift benchmark: NumPy reference vs Rust-native(ctypes) vs WASM(Node)
vs GPU(WGSL) at 100K/1M, seed 42. Stages ms + throughput + memory.
WASM via wasm_shift.mjs bench mode (5 warm + 10 measured, median).
GPU covers int32/float32 only (existing contract). Writes
results/bench_shift.json. Bit-exactness asserted (tobytes), no tolerance.
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
WASM = os.path.join(ROOT, "tools", "numfast_native.wasm")
MJS = os.path.join(ROOT, "tools", "wasm_shift.mjs")
NODE = shutil.which("node") or r"C:\Program Files\nodejs\node.exe"
TMP = os.path.join(RES, ".shift-bench")
os.makedirs(TMP, exist_ok=True)
DEV = os.path.normpath(os.path.join(ROOT, "..", ".."))
APP = os.path.join(DEV, "numfast")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NAT = _load("nf_native_cpu_b",
            os.path.join(APP, "src", "Drivers", "CPU", "_lib", "native_cpu.py"))
GPU = _load("nf_gpu_b",
            os.path.join(APP, "src", "Drivers", "GPU", "_lib", "gpu.py"))

KINDS = {"i32": (np.int32, 4), "f32": (np.float32, 4), "f64": (np.float64, 8)}


def bench_numpy(vals, periods, runs=10):
    for _ in range(5):  # warm
        out = np.zeros(vals.size, dtype=vals.dtype)
        out[periods:] = vals[:vals.size - periods]
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        out = np.zeros(vals.size, dtype=vals.dtype)
        out[periods:] = vals[:vals.size - periods]
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[runs >> 1], out


def bench_native(vals, periods, runs=10):
    for _ in range(5):
        NAT.shift_scatter(vals, periods)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        out = NAT.shift_scatter(vals, periods)
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[runs >> 1], out


def bench_wasm(vals, kind, periods):
    src = os.path.join(TMP, f"bench.{kind}")
    np.ascontiguousarray(vals).tofile(src)
    out = subprocess.run(
        [NODE, MJS, WASM, src, kind, str(vals.size), str(periods), "bench"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip())["median_ms"]


def bench_gpu(vals, kind, periods, runs=10):
    dt = {"i32": "int32", "f32": "float32"}[kind]
    for _ in range(2):
        GPU.shift_take(vals, periods, dt)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        out = GPU.shift_take(vals, periods, dt)
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[runs >> 1], np.ascontiguousarray(out)


def main():
    rng = np.random.default_rng(42)
    out = {"seed": 42, "cases": []}
    for n in (100_000, 1_000_000):
        for kind, (dt, B) in KINDS.items():
            if dt == np.int32:
                vals = np.ascontiguousarray(rng.integers(-2**30, 2**30, size=n, dtype=np.int64).astype(dt))
            else:
                vals = np.ascontiguousarray(rng.normal(0, 1000, size=n).astype(dt))
            for periods in (1, 7):
                mem_mb = n * B * 2 / 2**20
                t = time.perf_counter()
                ref_med, ref = bench_numpy(vals, periods)
                ref_stage = (time.perf_counter() - t) * 1e3
                nat_med, nat = bench_native(vals, periods)
                assert nat.tobytes() == ref.tobytes(), (n, kind, periods, "native")
                wasm_med = bench_wasm(vals, kind, periods)
                row = {"n": n, "kind": kind, "periods": periods,
                       "mem_mb": round(mem_mb, 2),
                       "numpy_ms": round(ref_med, 3), "native_ms": round(nat_med, 3),
                       "wasm_ms": round(wasm_med, 3),
                       "gb_s": {k: round(n * B / 2**30 / (m / 1e3), 2)
                                for k, m in (("numpy", ref_med), ("native", nat_med),
                                             ("wasm", wasm_med))}}
                if kind in ("i32", "f32"):
                    gpu_med, g = bench_gpu(vals, kind, periods)
                    assert g.tobytes() == ref.tobytes(), (n, kind, periods, "gpu")
                    row["gpu_ms"] = round(gpu_med, 3)
                    row["gb_s"]["gpu"] = round(n * B / 2**30 / (gpu_med / 1e3), 2)
                else:
                    row["gpu_ms"] = "explicit-error(f64-contract)"
                out["cases"].append(row)
                print(f"n={n} {kind} p={periods}: numpy={ref_med:.3f}ms "
                      f"native={nat_med:.3f}ms wasm={wasm_med:.3f}ms "
                      f"gpu={row['gpu_ms']} mem={mem_mb:.1f}MB "
                      f"(stage_total={ref_stage:.1f}ms) EXACT")
    json.dump(out, open(os.path.join(RES, "bench_shift.json"), "w"), indent=1)
    shutil.rmtree(TMP, ignore_errors=True)
    print("BENCH-PASS")


if __name__ == "__main__":
    main()
