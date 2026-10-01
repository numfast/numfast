# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU profile Q1/Q2 warm 10M G1_1e7_1e2_0_0, RTX 2060 (NEW file, bench-only).

Methodology: data load + shader warmup OUTSIDE timer. Inside timer only query.
Stages: host prep (outside) / upload H->D (bytes) / dispatch setup (counts,
workgroups, chunk) / GPU kernel per kernel (pack vs groupby; tiled/shared vs
global atomics; Planner path actually chosen) / sync / readback D->H (bytes) /
host merge / materialization / peak GPU mem estimate + RSS.
Correctness exact (chk total + ngroups). No optimization, no 1B.

Controls: (1) Q2 usual (np.load inside loop) vs resident (host arrays reused);
(2) Q2 N-scaling 256K/512K/1M/2M/4M/10M total/kernel/merge vs N.

Usage (Git Bash, sequential, timeout):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    timeout 550 /c/Users/Mikech/AppData/Local/Programs/Python/Python314/python.exe \
    tests/heavy/bench_gpu_profile_q1q2.py
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
import psutil

PROC = psutil.Process()
SNAP = FORK / "scratch" / "snap_G1_1e7_1e2_0_0"
OUT = FORK / "tests" / "heavy" / "bench_gpu_profile_q1q2.json"
N_FULL = 10_000_000
WG = 256
GB_TILED_M_MAX = 2048
GB_TILE = 2048
CHUNK = int(__import__("os").environ.get("NUMFAST_GPU_GROUPBY_CHUNK", str(1 << 20)))


def rss():
    return PROC.memory_info().rss / 1e9


def staged_run(dev, src, io_specs, n_threads, out_sizes):
    """Mirror Drivers/GPU _run with per-stage timers. No logic change."""
    import wgpu
    st = {}
    t = time.perf_counter()
    bufs = []
    h2d = 0
    for data, ro in io_specs:
        h2d += len(data)
        b = dev.create_buffer_with_data(
            data=data,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
            | wgpu.BufferUsage.COPY_SRC)
        bufs.append((b, ro))
    for sz in out_sizes:
        h2d += 0
        bufs_out = dev.create_buffer(
            size=max(1, int(sz)),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST)
        bufs.append((bufs_out, False))
        outs_n = len(out_sizes)
    st["upload_ms"] = (time.perf_counter() - t) * 1000
    st["h2d_bytes"] = h2d
    t = time.perf_counter()
    n_in = len(io_specs)
    allb = [b for b, _ in bufs]
    entries = []
    for i, (b, ro) in enumerate(bufs[:n_in]):
        typ = (wgpu.BufferBindingType.read_only_storage if ro
               else wgpu.BufferBindingType.storage)
        entries.append({"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": typ}})
    for j in range(len(bufs) - n_in):
        entries.append({"binding": n_in + j,
                        "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": wgpu.BufferBindingType.storage}})
    bgl = dev.create_bind_group_layout(entries=entries)
    bg = dev.create_bind_group(
        layout=bgl,
        entries=[{"binding": i, "resource": {"buffer": b, "offset": 0,
                                            "size": b.size}}
                 for i, b in enumerate(allb)])
    sm = dev.create_shader_module(code=src)
    pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])
    pipe = dev.create_compute_pipeline(
        layout=pl, compute={"module": sm, "entry_point": "main"})
    nw = max(1, (int(n_threads) + WG - 1) // WG)
    st["setup_ms"] = (time.perf_counter() - t) * 1000
    st["workgroups"] = nw
    t = time.perf_counter()
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe)
    cp.set_bind_group(0, bg, [], 0, 0)
    cp.dispatch_workgroups(nw, 1, 1)
    cp.end()
    dev.queue.submit([enc.finish()])
    st["dispatch_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    res = []
    d2h = 0
    for b, ro in bufs:
        if not ro:
            rb = bytes(dev.queue.read_buffer(b))
            d2h += len(rb)
            res.append(rb)
    st["sync_readback_ms"] = (time.perf_counter() - t) * 1000
    st["d2h_bytes"] = d2h
    st["peak_bytes"] = sum(b.size for b, _ in bufs)
    return res, st


def pack_staged(dev, k1, k2, m2):
    n = int(k1.size)
    src = f"""@group(0) @binding(0) var<storage,read> k1: array<i32>;
@group(0) @binding(1) var<storage,read> k2: array<i32>;
@group(0) @binding(2) var<storage,read_write> o: array<i32>;
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
  let i: u32 = g.x;
  if (i < {n}u) {{ o[i] = k1[i] * {int(m2)}i + k2[i]; }}
}}"""
    out = np.zeros(n, dtype=np.int32)
    (rb,), st = staged_run(dev, src, [(k1.tobytes(), True),
                                      (k2.tobytes(), True),
                                      (out.tobytes(), False)], n, [])
    st["dispatches"] = 1
    st["kernel"] = "pack_i32(elementwise,global)"
    return np.frombuffer(rb, dtype=np.int32).copy(), st


def gb_tiled_staged(dev, keys, hi, lo, m):
    c = int(keys.size)
    w = (c + GB_TILE - 1) // GB_TILE
    src = f"""@group(0) @binding(0) var<storage,read> keys: array<i32>;
@group(0) @binding(1) var<storage,read> hi: array<i32>;
@group(0) @binding(2) var<storage,read> lo: array<u32>;
@group(0) @binding(3) var<storage,read_write> phi: array<i32>;
@group(0) @binding(4) var<storage,read_write> plo: array<u32>;
var<workgroup> shi: array<atomic<i32>, {m}>;
var<workgroup> slo: array<atomic<u32>, {m}>;
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {{
  for (var j: u32 = l.x; j < {m}u; j += {WG}u) {{
    atomicStore(&shi[j], 0i);
    atomicStore(&slo[j], 0u);
  }}
  workgroupBarrier();
  let n: u32 = arrayLength(&keys);
  let base: u32 = w.x * {GB_TILE}u;
  for (var r: u32 = l.x; r < {GB_TILE}u; r += {WG}u) {{
    let i: u32 = base + r;
    if (i < n) {{
      let k: u32 = u32(keys[i]);
      atomicAdd(&shi[k], hi[i]);
      atomicAdd(&slo[k], lo[i]);
    }}
  }}
  workgroupBarrier();
  for (var j: u32 = l.x; j < {m}u; j += {WG}u) {{
    phi[w.x * {m}u + j] = atomicLoad(&shi[j]);
    plo[w.x * {m}u + j] = atomicLoad(&slo[j]);
  }}
}}"""
    phi0 = np.zeros(w * m, dtype=np.int32)
    plo0 = np.zeros(w * m, dtype=np.uint32)
    (phi_rb, plo_rb), st = staged_run(
        dev, src, [(keys.tobytes(), True), (hi.tobytes(), True),
                   (lo.tobytes(), True), (phi0.tobytes(), False),
                   (plo0.tobytes(), False)], c, [])
    st["dispatches"] = 1
    st["workgroups_tiled"] = w
    st["kernel"] = "tiled_shared_atomics"
    phi = np.frombuffer(phi_rb, dtype=np.int32).astype(np.int64)
    plo = np.frombuffer(plo_rb, dtype=np.uint32).astype(np.int64)
    return phi.reshape(w, m).sum(axis=0), plo.reshape(w, m).sum(axis=0), st


GB_GLOBAL_WGSL = """@group(0) @binding(0) var<storage,read> keys: array<i32>;
@group(0) @binding(1) var<storage,read> hi: array<i32>;
@group(0) @binding(2) var<storage,read> lo: array<u32>;
@group(0) @binding(3) var<storage,read_write> ghi: array<atomic<i32>>;
@group(0) @binding(4) var<storage,read_write> glo: array<atomic<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
  let n: u32 = arrayLength(&keys);
  let i: u32 = g.x;
  if (i < n) {{
    let k: u32 = u32(keys[i]);
    atomicAdd(&ghi[k], hi[i]);
    atomicAdd(&glo[k], lo[i]);
  }}
}}"""


def gb_global_accum(dev, keys, hi, lo, bhi, blo):
    """One chunk accumulate into persistent accums, staged timers."""
    import wgpu
    t = time.perf_counter()
    in_bufs = []
    h2d = 0
    for b in (keys.tobytes(), hi.tobytes(), lo.tobytes()):
        h2d += len(b)
        in_bufs.append(dev.create_buffer_with_data(
            data=b, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
            | wgpu.BufferUsage.COPY_SRC))
    up_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    allb = in_bufs + [bhi, blo]
    entries = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage}}
               for i in range(3)]
    for j in range(2):
        entries.append({"binding": 3 + j,
                        "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": wgpu.BufferBindingType.storage}})
    bgl = dev.create_bind_group_layout(entries=entries)
    bg = dev.create_bind_group(
        layout=bgl, entries=[{"binding": i, "resource": {"buffer": b,
                                                         "offset": 0,
                                                         "size": b.size}}
                             for i, b in enumerate(allb)])
    sm = dev.create_shader_module(code=GB_GLOBAL_WGSL)
    pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])
    pipe = dev.create_compute_pipeline(
        layout=pl, compute={"module": sm, "entry_point": "main"})
    n = int(keys.size)
    nw = max(1, (n + WG - 1) // WG)
    setup_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe)
    cp.set_bind_group(0, bg, [], 0, 0)
    cp.dispatch_workgroups(nw, 1, 1)
    cp.end()
    dev.queue.submit([enc.finish()])
    disp_ms = (time.perf_counter() - t) * 1000
    return {"upload_ms": up_ms, "h2d_bytes": h2d, "setup_ms": setup_ms,
            "workgroups": nw, "dispatch_ms": disp_ms,
            "sync_readback_ms": 0.0, "d2h_bytes": 0,
            "dispatches": 1, "kernel": "global_atomics",
            "peak_bytes": sum(b.size for b in allb)}


def q1_profile(dev, K, V):
    """Q1: dense groupby M=100 -> tiled path. Returns result + stages."""
    m = int(K.max()) + 1
    path = "tiled_shared" if m <= GB_TILED_M_MAX else "global_atomics"
    assert path == "tiled_shared", (m, path)
    n = int(K.size)
    nchunks = (n + CHUNK - 1) // CHUNK
    t = time.perf_counter()
    counts = np.bincount(K.astype(np.int64), minlength=m).astype(np.int64)
    counts_ms = (time.perf_counter() - t) * 1000
    hi = np.right_shift(V, np.int32(16))
    lo = np.bitwise_and(V, np.int32(0xFFFF)).astype(np.uint32)
    agg = np.zeros(m, dtype=np.int64)
    tot = {"upload_ms": 0.0, "h2d_bytes": 0, "setup_ms": 0.0,
           "dispatch_ms": 0.0, "sync_readback_ms": 0.0, "d2h_bytes": 0,
           "dispatches": 0, "peak_bytes": 0, "workgroups": 0}
    t = time.perf_counter()
    k_ms = 0.0
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        shi, slo, st = gb_tiled_staged(
            dev, np.ascontiguousarray(K[a:b]),
            np.ascontiguousarray(hi[a:b]),
            np.ascontiguousarray(lo[a:b]), m)
        for k in ("upload_ms", "setup_ms", "dispatch_ms",
                  "sync_readback_ms"):
            tot[k] += st[k]
        tot["h2d_bytes"] += st["h2d_bytes"]
        tot["d2h_bytes"] += st["d2h_bytes"]
        tot["dispatches"] += st["dispatches"]
        tot["workgroups"] += st["workgroups"]
        tot["peak_bytes"] = max(tot["peak_bytes"], st["peak_bytes"])
        agg += shi * np.int64(65536) + slo
    kernel_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    nz = np.flatnonzero(counts > 0)
    d = {int(k): int(v) for k, v in zip(nz.astype(np.int64).tolist(),
                                        agg[nz].tolist())}
    mat_ms = (time.perf_counter() - t) * 1000
    tot.update({"counts_ms": counts_ms, "kernel_ms": kernel_ms,
                "merge_in_kernel": True, "mat_ms": mat_ms,
                "nchunks": nchunks, "chunk": CHUNK, "m": m,
                "planner_path": path,
                "runtime_dispatches_nodes": 3})
    return d, tot


def q2_profile(dev, K1, K2, V):
    """Q2: pack + dense groupby M=10000 -> global path."""
    m2 = int(K2.max()) + 1
    t = time.perf_counter()
    P, pst = pack_staged(dev, K1, K2, m2)
    pack_ms = (time.perf_counter() - t) * 1000
    m = int(P.max()) + 1
    path = "tiled_shared" if m <= GB_TILED_M_MAX else "global_atomics"
    assert path == "global_atomics", (m, path)
    import wgpu
    t = time.perf_counter()
    counts = np.bincount(P.astype(np.int64), minlength=m).astype(np.int64)
    counts_ms = (time.perf_counter() - t) * 1000
    n = int(P.size)
    nchunks = (n + CHUNK - 1) // CHUNK
    t = time.perf_counter()
    bhi = dev.create_buffer(
        size=m * 4, usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
    blo = dev.create_buffer(
        size=m * 4, usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
    alloc_ms = (time.perf_counter() - t) * 1000
    tot = {"upload_ms": 0.0, "h2d_bytes": 0, "setup_ms": 0.0,
           "dispatch_ms": 0.0, "sync_readback_ms": 0.0, "d2h_bytes": 0,
           "dispatches": 0, "peak_bytes": 0, "workgroups": 0}
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        kk = np.ascontiguousarray(P[a:b])
        vv = np.ascontiguousarray(V[a:b])
        hi = np.right_shift(vv, np.int32(16))
        lo = np.bitwise_and(vv, np.int32(0xFFFF)).astype(np.uint32)
        st = gb_global_accum(dev, kk, hi, lo, bhi, blo)
        for k in ("upload_ms", "setup_ms", "dispatch_ms"):
            tot[k] += st[k]
        tot["h2d_bytes"] += st["h2d_bytes"]
        tot["dispatches"] += st["dispatches"]
        tot["workgroups"] += st["workgroups"]
        tot["peak_bytes"] = max(tot["peak_bytes"], st["peak_bytes"])
    t2 = time.perf_counter()
    ghi = np.frombuffer(bytes(dev.queue.read_buffer(bhi)),
                        dtype=np.int32).astype(np.int64).copy()
    glo = np.frombuffer(bytes(dev.queue.read_buffer(blo)),
                        dtype=np.uint32).astype(np.int64).copy()
    read_ms = (time.perf_counter() - t2) * 1000
    tot["sync_readback_ms"] = read_ms
    tot["d2h_bytes"] = 2 * m * 4
    tot["peak_bytes"] = max(tot["peak_bytes"], 2 * m * 4)
    t2 = time.perf_counter()
    agg = ghi * np.int64(65536) + glo
    merge_ms = (time.perf_counter() - t2) * 1000
    kernel_ms = pack_ms + tot["upload_ms"] + tot["setup_ms"] \
        + tot["dispatch_ms"] + read_ms + merge_ms + counts_ms
    t2 = time.perf_counter()
    nz = np.flatnonzero(counts > 0)
    d = {int(k): int(v) for k, v in zip(nz.astype(np.int64).tolist(),
                                        agg[nz].tolist())}
    mat_ms = (time.perf_counter() - t2) * 1000
    tot.update({"pack_ms": pack_ms, "pack_stages": pst,
                "counts_ms": counts_ms, "alloc_ms": alloc_ms,
                "merge_ms": merge_ms, "read_ms": read_ms,
                "kernel_ms": kernel_ms, "mat_ms": mat_ms,
                "nchunks": nchunks, "chunk": CHUNK, "m": m, "m2": m2,
                "planner_path": path,
                "runtime_dispatches_nodes": 5})
    return d, tot


def main():
    import wgpu
    print("=== gpu-profile-q1q2 N=10M ===", flush=True)
    print(f"RSS start {rss():.2f}GB", flush=True)
    t = time.perf_counter()
    K1 = np.load(SNAP / "K1.npy")
    K2 = np.load(SNAP / "K2.npy")
    V1 = np.load(SNAP / "V1.npy")
    load_ms = (time.perf_counter() - t) * 1000
    K1 = np.ascontiguousarray(K1, dtype=np.int32)
    K2 = np.ascontiguousarray(K2, dtype=np.int32)
    V1 = np.ascontiguousarray(V1, dtype=np.int32)
    nbytes = int(K1.nbytes + K2.nbytes + V1.nbytes)
    print(f"host_prep(load snap) {load_ms:.0f}ms bytes={nbytes} "
          f"RSS {rss():.2f}GB", flush=True)
    dev = wgpu.gpu.request_adapter_sync(
        power_preference="high-performance").request_device_sync()
    print(f"GPU {wgpu.gpu.request_adapter_sync().info.device}",
          flush=True)
    t = time.perf_counter()
    k = np.arange(64, dtype=np.int32)
    v = np.arange(64, dtype=np.int32)
    m2w = int(k.max()) + 1
    pack_staged(dev, k, k, m2w)
    gb_tiled_staged(dev, np.ascontiguousarray(k % 8),
                    np.ascontiguousarray(v >> 16),
                    np.ascontiguousarray((v & 0xFFFF).astype(np.uint32)), 8)
    warm_ms = (time.perf_counter() - t) * 1000
    print(f"warmup_compile {warm_ms:.0f}ms RSS {rss():.2f}GB", flush=True)
    out = {"N": N_FULL, "host_prep_ms": load_ms,
           "host_prep_bytes": nbytes, "warmup_ms": warm_ms,
           "chunk": CHUNK, "rss_gb": rss()}
    for name, fn in (("Q1", lambda: q1_profile(dev, K1, V1)),
                     ("Q2", lambda: q2_profile(dev, K1, K2, V1))):
        d, st = fn()
        tot = int(sum(d.values()))
        assert len(d) == (100 if name == "Q1" else 10000), len(d)
        assert tot == 29998789, tot
        d2, st2 = fn()
        assert int(sum(d2.values())) == 29998789
        assert len(d2) == len(d)
        inside = (st2["upload_ms"] + st2["setup_ms"] + st2["dispatch_ms"]
                  + st2["sync_readback_ms"] + st2.get("counts_ms", 0)
                  + st2.get("merge_ms", 0) + st2.get("mat_ms", 0)
                  + st2.get("pack_ms", 0 if name == "Q1" else 0))
        st2["warm_total_inside_ms"] = inside
        st2["chk_total"] = tot
        st2["chk_ngroups"] = len(d)
        out[name] = st2
        print(f"{name} warm chk={tot} groups={len(d)} path={st2['planner_path']} "
              f"disp={st2['dispatches']} wg={st2['workgroups']} "
              f"chunks={st2['nchunks']}x{st2['chunk']} "
              f"up={st2['upload_ms']:.1f}ms({st2['h2d_bytes']}) "
              f"setup={st2['setup_ms']:.1f}ms disp={st2['dispatch_ms']:.1f}ms "
              f"read={st2['sync_readback_ms']:.1f}ms({st2['d2h_bytes']}) "
              f"counts={st2.get('counts_ms',0):.1f}ms "
              f"merge={st2.get('merge_ms',0):.1f}ms mat={st2['mat_ms']:.1f}ms "
              f"RSS {rss():.2f}GB", flush=True)
        if name == "Q2":
            ps = st2["pack_stages"]
            print(f"  pack: up={ps['upload_ms']:.1f}ms({ps['h2d_bytes']}) "
                  f"setup={ps['setup_ms']:.1f}ms disp={ps['dispatch_ms']:.1f}ms "
                  f"read={ps['sync_readback_ms']:.1f}ms({ps['d2h_bytes']}) "
                  f"wg={ps['workgroups']}", flush=True)
    t = time.perf_counter()
    _ = np.load(SNAP / "K1.npy")
    reload_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    d, st = q2_profile(dev, K1, K2, V1)
    resident_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    a1 = np.load(SNAP / "K1.npy")
    a2 = np.load(SNAP / "K2.npy")
    av = np.load(SNAP / "V1.npy")
    d0, _ = q2_profile(dev, a1, a2, av)
    usual_ms = (time.perf_counter() - t) * 1000
    assert int(sum(d.values())) == int(sum(d0.values())) == 29998789
    out["resident_ctrl"] = {"resident_ms": resident_ms,
                            "usual_ms": usual_ms,
                            "reload_one_col_ms": reload_ms}
    print(f"resident Q2 {resident_ms:.0f}ms usual(reload+query) "
          f"{usual_ms:.0f}ms RSS {rss():.2f}GB", flush=True)
    scale = {}
    for N in (256_000, 512_000, 1_000_000, 2_000_000, 4_000_000, 10_000_000):
        k1, k2, vv = K1[:N], K2[:N], V1[:N]
        ref = {}
        import collections
        m2 = int(k2.max()) + 1
        P = k1.astype(np.int64) * m2 + k2.astype(np.int64)
        uk, inv = np.unique(P, return_inverse=True)
        ss = np.bincount(inv, weights=vv.astype(np.float64)).astype(np.int64)
        ref_tot = int(ss.sum())
        t = time.perf_counter()
        d, st = q2_profile(dev, np.ascontiguousarray(k1),
                           np.ascontiguousarray(k2),
                           np.ascontiguousarray(vv))
        tot_ms = (time.perf_counter() - t) * 1000
        assert int(sum(d.values())) == ref_tot, (N, int(sum(d.values())),
                                                 ref_tot)
        kern = st["kernel_ms"]
        merge = st.get("merge_ms", 0) + st.get("counts_ms", 0)
        scale[str(N)] = {"total_ms": tot_ms, "kernel_ms": kern,
                         "merge_ms": merge, "mat_ms": st["mat_ms"],
                         "groups": len(d), "ref_total": ref_tot,
                         "dispatches": st["dispatches"],
                         "h2d_bytes": st["h2d_bytes"],
                         "d2h_bytes": st["d2h_bytes"]}
        print(f"N={N} total={tot_ms:.1f}ms kernel={kern:.1f}ms "
              f"merge={merge:.1f}ms mat={st['mat_ms']:.1f}ms "
              f"groups={len(d)} RSS {rss():.2f}GB", flush=True)
        gc.collect()
    out["n_scale"] = scale
    out["rss_gb"] = rss()
    with open(OUT, "w") as f:
        json.dump(json.loads(json.dumps(out, default=float)), f, indent=1)
    print(f"JSON {OUT} RSS {rss():.2f}GB", flush=True)


if __name__ == "__main__":
    main()
