# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Map benchmark: NumPy reference vs Rust-native(ctypes) vs WASM(Node)
vs GPU(WGSL) at 100K/1M, seed 42. Stages ms + throughput + memory.
WASM via wasm_map.mjs bench mode (5 warm + 10 measured, median).
GPU: cold (first call incl. device/pipeline init, measured once per
dtype) reported separately from warm medians (2 warm + 10 measured);
e2e includes H2D+D2H+kernel (transfer bytes reported, no kernel-only
claim). GPU-covered ops only (i32 add/floor_div, f32 add); gaps are
explicit errors, never timed as fallback. Writes results/bench_map.json.
Bit-exactness asserted (tobytes for int, array_equal equal_nan for
float non-pow), no tolerance.
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
MJS = os.path.join(ROOT, "tools", "wasm_map.mjs")
NODE = shutil.which("node") or r"C:\Program Files\nodejs\node.exe"
TMP = os.path.join(RES, ".map-bench")
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
OPCODE = {"add": 0, "div": 3, "floor_div": 5}
GPU_OK = {("i32", "add"), ("i32", "floor_div"), ("f32", "add")}
DTNAME = {"i32": "int32", "f32": "float32", "f64": "float64"}


def med(ts):
    ts = sorted(ts)
    return ts[len(ts) >> 1]


def bench_numpy(a, fn, b, runs=10):
    for _ in range(5):
        NAT._fb_map(a, fn, b)
    ts, out = [], None
    for _ in range(runs):
        t = time.perf_counter()
        out = NAT._fb_map(a, fn, b)
        ts.append((time.perf_counter() - t) * 1e3)
    return med(ts), out


def bench_native(a, fn, b, runs=10):
    for _ in range(5):
        NAT.map_scatter(a, fn, b)
    ts, out = [], None
    for _ in range(runs):
        t = time.perf_counter()
        out = NAT.map_scatter(a, fn, b)
        ts.append((time.perf_counter() - t) * 1e3)
    return med(ts), out


def bench_wasm(a, b, kind, op, scalar_kind, scalar_val):
    dt = KINDS[kind][0]
    a = np.ascontiguousarray(a, dtype=dt)
    a_path = os.path.join(TMP, f"bench_a.{kind}")
    a.tofile(a_path)
    if scalar_kind == "arr":
        b = np.ascontiguousarray(b, dtype=dt)
        b_path = os.path.join(TMP, f"bench_b.{kind}")
        b.tofile(b_path)
        sk, sv = "arr", "0"
    else:
        b_path = "-"
        sk, sv = scalar_kind, repr(float(scalar_val))
    out = subprocess.run(
        [NODE, MJS, WASM, a_path, b_path, kind, str(op), str(a.size),
         sk, sv, "bench"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip())["median_ms"]


def bench_gpu(a, fn, b, kind, runs=10):
    dt = DTNAME[kind]
    for _ in range(2):
        GPU.map_elem(a, b, fn, dt)
    ts, out = [], None
    for _ in range(runs):
        t = time.perf_counter()
        out = GPU.map_elem(a, b, fn, dt)
        ts.append((time.perf_counter() - t) * 1e3)
    return med(ts), np.ascontiguousarray(out)


def main():
    rng = np.random.default_rng(42)
    out = {"seed": 42, "cases": []}
    # GPU cold: first call per dtype (device + pipeline init), honest once.
    cold = {}
    for kind in ("i32", "f32"):
        dt, _ = KINDS[kind]
        x = np.zeros(1024, dtype=dt)
        t = time.perf_counter()
        GPU.map_elem(x, x, "add", DTNAME[kind])
        cold[kind] = (time.perf_counter() - t) * 1e3
        print(f"gpu-cold/{kind}: {cold[kind]:.1f}ms (device+pipeline init, once)")
    for n in (100_000, 1_000_000):
        for kind, (dt, B) in KINDS.items():
            if kind == "i32":
                x = np.ascontiguousarray(
                    rng.integers(-10_000, 10_000, size=n, dtype=np.int64).astype(dt))
                y = np.ascontiguousarray(
                    rng.integers(-500, 500, size=n, dtype=np.int64).astype(dt))
                y[y == 0] = 7
            else:
                x = np.ascontiguousarray(rng.normal(0, 1000, size=n).astype(dt))
                y = np.ascontiguousarray(rng.normal(0, 100, size=n).astype(dt))
            for fn in ("add", "div", "floor_div"):
                for form, b, sk, sv in (("arr", y, "arr", 0), ("scal", 7, "i32", 7)):
                    op = OPCODE[fn]
                    nbytes = n * B * (3 if form == "arr" else 2)
                    mem_mb = nbytes / 2**20
                    t = time.perf_counter()
                    ref_med, ref = bench_numpy(x, fn, b)
                    ref_stage = (time.perf_counter() - t) * 1e3
                    nat_med, nat = bench_native(x, fn, b)
                    assert nat.dtype == ref.dtype and nat.tobytes() == ref.tobytes(), \
                        (n, kind, fn, form, "native")
                    if kind == "i32" and form == "scal":
                        sk_w = "i32"
                    elif form == "arr":
                        sk_w = "arr"
                    else:
                        sk_w = "f64"
                    wasm_med = bench_wasm(x, b, kind, op, sk_w, sv)
                    row = {"n": n, "kind": kind, "fn": fn, "form": form,
                           "mem_mb": round(mem_mb, 2),
                           "xfer_mb": round(mem_mb, 2),
                           "numpy_ms": round(ref_med, 3),
                           "native_ms": round(nat_med, 3),
                           "wasm_ms": round(wasm_med, 3),
                           "gb_s": {k: round(nbytes / 2**30 / (m / 1e3), 2)
                                    for k, m in (("numpy", ref_med),
                                                 ("native", nat_med),
                                                 ("wasm", wasm_med))}}
                    if (kind, fn) in GPU_OK:
                        gpu_med, gg = bench_gpu(x, fn, b, kind)
                        assert gg.dtype == ref.dtype and gg.tobytes() == ref.tobytes(), \
                            (n, kind, fn, form, "gpu")
                        row["gpu_warm_ms"] = round(gpu_med, 3)
                        row["gpu_cold_ms"] = round(cold[kind], 1)
                        row["gb_s"]["gpu_warm"] = round(
                            nbytes / 2**30 / (gpu_med / 1e3), 2)
                    else:
                        row["gpu_warm_ms"] = "explicit-gap(cpu-owns)"
                        row["gpu_cold_ms"] = "explicit-gap(cpu-owns)"
                    out["cases"].append(row)
                    print(f"n={n} {kind} {fn}/{form}: numpy={ref_med:.3f}ms "
                          f"native={nat_med:.3f}ms wasm={wasm_med:.3f}ms "
                          f"gpu={row['gpu_warm_ms']} xfer={mem_mb:.1f}MB "
                          f"(stage_total={ref_stage:.1f}ms) EXACT")
    json.dump(out, open(os.path.join(RES, "bench_map.json"), "w"), indent=1)
    shutil.rmtree(TMP, ignore_errors=True)
    print("BENCH-PASS")


if __name__ == "__main__":
    main()
