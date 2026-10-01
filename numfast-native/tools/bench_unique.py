# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Unique benchmark: NumPy oracle vs Rust-native(ctypes) vs WASM(Node)
at 100K/1M, seed 42. GPU is an explicit gap (asserted, not timed).

Stages ms + throughput + memory. WASM via wasm_unique.mjs bench mode
(5 warm + 10 measured, median of the in-WASM kernel call only).
Native via native_cpu.unique_inverse (1 cold + best-of-3 warm, backend
must read "native"). Bit-exact asserted (uniq values + int64 codes +
uniq[inv]==keys). Writes results/bench_unique.json.
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
MJS = os.path.join(ROOT, "tools", "wasm_unique.mjs")
NODE = shutil.which("node") or r"C:\Program Files\nodejs\node.exe"
TMP = os.path.join(RES, ".unique-bench")
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

KINDS = {"i32": (np.int32, 4), "i64": (np.int64, 8)}
SEED = 42


def wasm_bench(vals, kind):
    dt, _ = KINDS[kind]
    vals = np.ascontiguousarray(vals, dtype=dt)
    src = os.path.join(TMP, f"bench.{kind}")
    vals.tofile(src)
    out = subprocess.run(
        [NODE, MJS, WASM, src, kind, str(vals.size), "bench"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip())


def main():
    assert NAT.unique_available(), NAT.why()
    rng = np.random.default_rng(SEED)
    rows = {}
    for n in (100_000, 1_000_000):
        for kind, (dt, wb) in KINDS.items():
            t0 = time.perf_counter()
            k = rng.integers(-500_000, 500_000, size=n).astype(dt)
            k = np.ascontiguousarray(k)
            gen_ms = (time.perf_counter() - t0) * 1e3
            # caller bytes: keys + uniq + inv4 + perm4 + tmp0 + tmp1 + tmpp4
            mem_mb = n * (wb + wb + 4 + 4 + wb + wb + 4) / 1048576.0

            t = time.perf_counter()
            eu, ei = np.unique(k, return_inverse=True)
            cold_base = (time.perf_counter() - t) * 1e3
            gu, gi, be = NAT.unique_inverse(k)
            assert be == "native", be
            t = time.perf_counter()
            gu, gi, be = NAT.unique_inverse(k)
            cold_nat = (time.perf_counter() - t) * 1e3
            assert be == "native", be
            runs_b, runs_n = [], []
            for _ in range(3):
                t = time.perf_counter()
                eu, ei = np.unique(k, return_inverse=True)
                runs_b.append((time.perf_counter() - t) * 1e3)
                t = time.perf_counter()
                gu, gi, be = NAT.unique_inverse(k)
                runs_n.append((time.perf_counter() - t) * 1e3)
                assert be == "native", be
            assert gu.dtype == eu.dtype and np.array_equal(gu, eu), "uniq mismatch"
            assert np.array_equal(gi, ei.astype(np.int64)), "inverse mismatch"
            assert np.array_equal(gu[gi], k), "reconstruct"
            wb_stat = wasm_bench(k, kind)
            best_b, best_n = min(runs_b), min(runs_n)
            gb = (n * wb + gu.size * wb + n * 4) / 1073741824.0
            key = f"n{n}_{kind}"
            rows[key] = {"n": n, "ng": int(gu.size),
                         "gen_ms": round(gen_ms, 1),
                         "mem_native_mb": round(mem_mb, 1),
                         "cold_base": round(cold_base, 1),
                         "cold_nat": round(cold_nat, 1),
                         "runs_base": [round(r, 1) for r in runs_b],
                         "runs_nat": [round(r, 1) for r in runs_n],
                         "best_base": round(best_b, 1),
                         "best_nat": round(best_n, 1),
                         "ratio": round(best_b / best_n, 2),
                         "gb_per_s_nat": round(gb / (best_n / 1000.0), 2),
                         "wasm_median_ms": wb_stat["median_ms"],
                         "wasm_runs": [round(r, 3) for r in wb_stat["runs"]]}
            print(f"bench {key} ng={gu.size} base={best_b:.1f} nat={best_n:.1f} "
                  f"x{best_b / best_n:.2f} wasm_med={wb_stat['median_ms']:.3f} "
                  f"mem={mem_mb:.0f}MB gen={gen_ms:.0f}", flush=True)
            del k, gu, gi, eu, ei
    json.dump({"seed": SEED, **rows},
              open(os.path.join(RES, "bench_unique.json"), "w"), indent=1)
    shutil.rmtree(TMP, ignore_errors=True)
    print("BENCH OK")


if __name__ == "__main__":
    main()
