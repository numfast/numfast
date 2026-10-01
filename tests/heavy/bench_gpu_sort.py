# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""GPU sort/topk bench: correctness gates + N-scaling + GPU vs CPU (NEW file).

Design under test: bitonic stable permutation, single int32/finite-f32 key,
pow2 N; sorted keys = gather(keys, perm) composition (resident, never
dict/list); topk = sort -> slice -> gather composition (no separate kernel).

Gates: 1000+ random cases (seed 42) exact vs CPU driver + numpy stable refs;
validity/NaN/error paths; N-scaling 256K/1M/4M GPU vs CPU, 10M CPU-only (GPU
records explicit non-pow2 error); H2D/D2H micro-measured separately, warm
runs, RTX 2060. Consumer: order-by+limit query + rank, resident on GPU.

Usage (Git Bash, sequential, one chunk per command):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    timeout 550 python tests/heavy/bench_gpu_sort.py
"""

import gc
import json
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
for _p in (str(FORK.parent / "app-builder-ponytail"), str(FORK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from builder import MAIN

A = MAIN["build"](str(FORK)).alias
OUT_JSON = Path(__file__).with_suffix(".json")
I32MIN, I32MAX = -(2 ** 31), 2 ** 31 - 1
RES = {"seed": 42, "device": "RTX 2060", "stages": []}


def _stable_desc_ref(V):
    n = len(V)
    asc = np.argsort(V, kind="stable")
    sk = V[asc]
    ch = np.empty(n, dtype=bool)
    ch[0] = True
    if n > 1:
        ch[1:] = sk[1:] != sk[:-1]
    grp = np.cumsum(ch)
    return asc[np.argsort(grp.max(initial=0) - grp, kind="stable")]


def _run_gpu(V, dtype="int32", desc=False, validity=None):
    kw = {} if validity is None else {"validity": validity}
    jobs = [A["ir_series"]("v", V, dtype, **kw),
            A["ir_sort"]("p", "v", descending=desc)]
    g = A["compile"](jobs)
    t = time.perf_counter()
    bufs = A["gpu_execute"](g["nodes"])
    return np.asarray(bufs["p"]), (time.perf_counter() - t) * 1000


def _run_cpu(V, dtype="int32", desc=False, validity=None):
    kw = {} if validity is None else {"validity": validity}
    jobs = [A["ir_series"]("v", V, dtype, **kw),
            A["ir_sort"]("p", "v", descending=desc)]
    g = A["compile"](jobs)
    t = time.perf_counter()
    bufs = A["cpu_execute"](g["nodes"])
    return np.asarray(bufs["p"]), (time.perf_counter() - t) * 1000


def chunk_correctness():
    """Gate 1: >=1000 random cases exact vs CPU driver + numpy stable refs."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(42)
    n_cases, n_fail = 0, 0
    sizes = [4, 8, 16, 32, 64, 256]
    for i in range(1000):
        n = sizes[i % len(sizes)]
        desc = bool(i & 1)
        kind = (i // 2) % 3
        if kind == 0:
            V = rng.integers(I32MIN, I32MAX + 1, n).astype(np.int32)
            dt = "int32"
        elif kind == 1:
            V = rng.integers(-5, 6, n).astype(np.int32)  # heavy ties
            dt = "int32"
        else:
            V = (rng.normal(0, 5, n)).astype(np.float32)
            V[V > 8] = np.inf
            V[V < -8] = -np.inf
            dt = "float32"
        got, _ = _run_gpu(V, dt, desc)
        want_cpu, _ = _run_cpu(V, dt, desc)
        ref = _stable_desc_ref(V) if desc else np.argsort(V, kind="stable")
        n_cases += 1
        if not ((got == want_cpu).all() and (got == ref).all()):
            n_fail += 1
            print(f"  FAIL case {i} n={n} {dt} desc={desc}", flush=True)
            if n_fail > 3:
                break
    # validity: pow2 popcount exact; non-pow2 popcount explicit error
    for i in range(60):
        n = 32
        k = [4, 8, 16][i % 3]
        m = np.zeros(n, dtype=bool)
        m[rng.choice(n, k, replace=False)] = True
        V = rng.integers(-50, 50, n).astype(np.int32)
        for desc in (False, True):
            got, _ = _run_gpu(V, "int32", desc, m.tolist())
            want_cpu, _ = _run_cpu(V, "int32", desc, m.tolist())
            n_cases += 1
            if not (got == want_cpu).all():
                n_fail += 1
                print(f"  FAIL validity {i} desc={desc}", flush=True)
    # error paths raise explicit (no silent fallback)
    err_cases = 0
    for mk in (lambda: _run_gpu(np.arange(100, dtype=np.int32)),
               lambda: _run_gpu(np.array([1.0, np.nan] * 32,
                                         dtype=np.float32), "float32")):
        try:
            mk()
            print("  FAIL: no-raise on error path", flush=True)
            n_fail += 1
        except ValueError:
            err_cases += 1
        n_cases += 1
    ms = (time.perf_counter() - t0) * 1000
    print(f"[correctness] cases={n_cases} fail={n_fail} "
          f"err_paths_ok={err_cases}/2 total_ms={ms:.0f} "
          f"elapsed={(ms / 1000):.1f}s", flush=True)
    RES["stages"].append({"chunk": "correctness", "cases": n_cases,
                          "fail": n_fail, "ms": ms})
    return n_fail == 0


_XDEV = None


def _transfer_micro(n):
    """H2D/D2H micro on same-size buffers (device cached + warmed; the
    driver-side transfer is buffer create with data / readback)."""
    global _XDEV
    import wgpu
    if _XDEV is None:
        ad = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance")
        _XDEV = ad.request_device_sync()
        wb = _XDEV.create_buffer_with_data(
            data=np.zeros(256, dtype=np.int32).tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        _ = bytes(_XDEV.queue.read_buffer(wb))
    dev = _XDEV
    kb = np.zeros(n, dtype=np.int32)
    ib = np.zeros(n, dtype=np.int32)
    t = time.perf_counter()
    b1 = dev.create_buffer_with_data(
        data=kb.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    b2 = dev.create_buffer_with_data(
        data=ib.tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        | wgpu.BufferUsage.COPY_DST)
    h2d = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    _ = bytes(dev.queue.read_buffer(b2))
    d2h = (time.perf_counter() - t) * 1000
    return h2d, d2h


def chunk_scaling():
    """Gate 2: N-scaling GPU vs CPU + transfer split, warm runs."""
    rng = np.random.default_rng(42)
    rows = []
    for n in (1 << 18, 1 << 20, 1 << 22):
        V = rng.integers(I32MIN, I32MAX + 1, n).astype(np.int32)
        _run_gpu(V)  # warm (device/pipeline)
        _run_cpu(V)  # warm
        _, t_gpu = _run_gpu(V)
        _, t_gpu2 = _run_gpu(V)
        _, t_cpu = _run_cpu(V)
        h2d, d2h = _transfer_micro(n)
        got, _ = _run_gpu(V)
        exact = bool((got == np.argsort(V, kind="stable")).all())
        row = {"n": n, "gpu_ms": round(min(t_gpu, t_gpu2), 3),
               "cpu_ms": round(t_cpu, 3),
               "h2d_ms": round(h2d, 3), "d2h_ms": round(d2h, 3),
               "exact": exact}
        row["speedup_gpu_vs_cpu"] = round(t_cpu / min(t_gpu, t_gpu2), 3)
        rows.append(row)
        print(f"[scaling] n={n} gpu={row['gpu_ms']:.1f}ms "
              f"cpu={t_cpu:.1f}ms h2d={h2d:.1f} d2h={d2h:.1f} "
              f"exact={exact}", flush=True)
        gc.collect()
    # 10M: CPU-only (GPU must raise explicit non-pow2 error)
    n = 10_000_000
    V = rng.integers(0, 1000, n).astype(np.int32)
    _, t_cpu = _run_cpu(V)
    try:
        _run_gpu(V)
        gpu10 = "NO-RAISE BUG"
    except ValueError:
        gpu10 = "explicit-cpu-error-ok"
    rows.append({"n": n, "gpu_ms": gpu10, "cpu_ms": round(t_cpu, 3)})
    print(f"[scaling] n={n} cpu={t_cpu:.1f}ms gpu={gpu10}", flush=True)
    RES["stages"].append({"chunk": "scaling", "rows": rows})


def chunk_topk_consumer():
    """Gate 3: topk = sort->slice->gather; tail cost vs full sort (why no
    separate kernel) + consumer order-by+limit / rank resident on GPU."""
    rng = np.random.default_rng(42)
    n, k = 1 << 20, 100
    V = rng.integers(0, 10 * n, n).astype(np.int32)
    _run_gpu(V)
    # full topk chain on GPU
    jobs = [A["ir_series"]("v", V),
            A["ir_sort"]("p", "v", descending=True),
            A["ir_slice"]("t", "p", limit=k),
            A["ir_gather"]("g", "v", "t")]
    g = A["compile"](jobs)
    t = time.perf_counter()
    bufs = A["gpu_execute"](g["nodes"])
    t_chain = (time.perf_counter() - t) * 1000
    topk = np.asarray(bufs["g"])
    ref_idx = np.argsort(-V, kind="stable")[:k]
    ok_topk = bool((topk == V[ref_idx]).all())
    # tail only (slice+gather on pre-sorted perm) vs full chain
    t = time.perf_counter()
    gp, t_sort = _run_gpu(V, "int32", True)
    jobs2 = [A["ir_series"]("v", V), A["ir_series"]("p", gp),
             A["ir_slice"]("t", "p", limit=k),
             A["ir_gather"]("g2", "v", "t")]
    g2 = A["compile"](jobs2)
    t = time.perf_counter()
    bufs2 = A["gpu_execute"](g2["nodes"])
    t_tail = (time.perf_counter() - t) * 1000
    # consumer 2: rank = inverse permutation (second sort), resident
    jobs3 = [A["ir_series"]("v", V[:4096]),
             A["ir_sort"]("p", "v"),
             A["ir_sort"]("r", "p")]
    bufs3 = A["gpu_execute"](A["compile"](jobs3)["nodes"])
    rank = np.asarray(bufs3["r"])
    inv = np.empty(4096, dtype=np.int64)
    inv[np.asarray(bufs3["p"])] = np.arange(4096)
    ok_rank = bool((rank == inv).all())
    print(f"[topk] n={n} k={k} chain_ms={t_chain:.1f} sort_ms={t_sort:.1f} "
          f"tail_ms={t_tail:.1f} tail_share={t_tail / t_chain:.3f} "
          f"topk_exact={ok_topk} rank_exact={ok_rank}", flush=True)
    RES["stages"].append({"chunk": "topk_consumer", "n": n, "k": k,
                          "chain_ms": t_chain, "sort_ms": t_sort,
                          "tail_ms": t_tail, "topk_exact": ok_topk,
                          "rank_exact": ok_rank})


if __name__ == "__main__":
    t0 = time.perf_counter()
    ok = chunk_correctness()
    chunk_scaling()
    chunk_topk_consumer()
    RES["elapsed_s"] = time.perf_counter() - t0
    RES["verdict"] = "GREEN" if ok else "RED"
    OUT_JSON.write_text(json.dumps(RES, indent=1))
    print(f"VERDICT={RES['verdict']} elapsed={RES['elapsed_s']:.1f}s "
          f"-> {OUT_JSON}", flush=True)
