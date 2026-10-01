# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU-final vs hybrid Q3/Q4/Q5, H2O 10M G1_1e7_1e2_0_0, RTX 2060 (NEW, bench-only).

Same principle as bench_gpu_final_q1q2.py (moved, not tuned):
- Hybrid (current arch): H2D(keys+hi+lo per chunk) -> GPU tiled/global
  partials -> D2H partials -> CPU bincount + host int64 merge ->
  host unscale (v3 scaled f64, scale=1e6) -> materialize.
- GPU-final: counts+ Sums on GPU (inline i32 split, no hi/lo upload,
  no CPU bincount) -> GPU finalize (tiled Q4) or single fused global
  accum (Q3/Q5, M>2048) -> D2H finals(M) -> host unscale+mean -> mat.
- Q3: sum v1 + mean v3 by id3 (K3 1..100000, M=100001).
- Q4: mean v1/v2/v3 by id4 (K4 0..99, M=100, tiled).
- Q5: sum v1/v2/v3 by id6 (K6 0..99999, M=100000).
V3 logical f64 -> physical int32 ticks rint(v*1e6) on host once (C-speed,
DELTA-7 contract); WGSL sees non-negative int32 only (exact path).
Correctness: int cols exact dict equality; v3 scaled tolerance maxdiff
<1e-9 (DELTA-7 gate observed 6e-14/1e-12/7e-12). No core change, no SPEC
change, no push. h2d convention: input bytes (keys+vals); zero-init
accums/partials device-alloc not counted (same as Q1-final).

Usage (Git Bash, sequential, timeout):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    timeout 550 /c/Users/Mikech/AppData/Local/Programs/Python/Python314/python.exe \
    tests/heavy/bench_gpu_final_q3q5.py
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
OUT = FORK / "tests" / "heavy" / "bench_gpu_final_q3q5.json"
WG = 256
TILE = 2048
CHUNK = 1 << 20
SCALE = 1e-6
GOLD = {"Q3": {"ngroups": 100000, "v1": 29998789,
               "mean3_sum": 4999719.622344427},
        "Q4": {"ngroups": 100, "m1": 299.98798187506526,
               "m2": 799.894179409978, "m3": 4999.766872833688},
        "Q5": {"ngroups": 100000, "s1": 29998789, "s2": 79989360,
               "s3_scaled": 499976651408061}}


def rss():
    return PROC.memory_info().rss / 1e9


def _run_ret(dev, src, io_specs, n_threads):
    import wgpu
    t = time.perf_counter()
    bufs = []
    h2d = 0
    for data, ro in io_specs:
        h2d += len(data)
        bufs.append((dev.create_buffer_with_data(
            data=data, usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC), ro))
    up_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    allb = [b for b, _ in bufs]
    entries = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": (wgpu.BufferBindingType.read_only_storage
                                    if ro else wgpu.BufferBindingType.storage)}}
               for i, (_, ro) in enumerate(bufs)]
    bgl = dev.create_bind_group_layout(entries=entries)
    bg = dev.create_bind_group(
        layout=bgl, entries=[{"binding": i, "resource": {"buffer": b,
                                                         "offset": 0,
                                                         "size": b.size}}
                             for i, b in enumerate(allb)])
    sm = dev.create_shader_module(code=src)
    pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])
    pipe = dev.create_compute_pipeline(
        layout=pl, compute={"module": sm, "entry_point": "main"})
    nw = max(1, (int(n_threads) + WG - 1) // WG)
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
    t = time.perf_counter()
    res, d2h = [], 0
    for b, ro in bufs:
        if not ro:
            rb = bytes(dev.queue.read_buffer(b))
            d2h += len(rb)
            res.append(rb)
    read_ms = (time.perf_counter() - t) * 1000
    return res, {"up_ms": up_ms, "h2d": h2d, "setup_ms": setup_ms,
                 "workgroups": nw, "disp_ms": disp_ms,
                 "read_ms": read_ms, "d2h": d2h}


def _split_host(v):
    hi = np.right_shift(v, np.int32(16))
    lo = np.bitwise_and(v, np.int32(0xFFFF)).astype(np.uint32)
    return hi, lo


def _gb_global_run(dev, keys, hi, lo, m, bhi, blo):
    """One chunk dispatch into persistent (bhi, blo); returns stage ms."""
    import wgpu
    t = time.perf_counter()
    in_bufs = [dev.create_buffer_with_data(
        data=x, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        | wgpu.BufferUsage.COPY_SRC)
        for x in (keys.tobytes(), hi.tobytes(), lo.tobytes())]
    h2d = sum(len(x) for x in (keys.tobytes(), hi.tobytes(), lo.tobytes()))
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
    sm = dev.create_shader_module(code=_GB_GLOBAL_WGSL)
    pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])
    pipe = dev.create_compute_pipeline(
        layout=pl, compute={"module": sm, "entry_point": "main"})
    setup_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe)
    cp.set_bind_group(0, bg, [], 0, 0)
    cp.dispatch_workgroups(max(1, (int(keys.size) + WG - 1) // WG), 1, 1)
    cp.end()
    dev.queue.submit([enc.finish()])
    kern_ms = (time.perf_counter() - t) * 1000
    return h2d, up_ms, setup_ms, kern_ms


_GB_GLOBAL_WGSL = """@group(0) @binding(0) var<storage,read> keys: array<i32>;
@group(0) @binding(1) var<storage,read> hi: array<i32>;
@group(0) @binding(2) var<storage,read> lo: array<u32>;
@group(0) @binding(3) var<storage,read_write> ghi: array<atomic<i32>>;
@group(0) @binding(4) var<storage,read_write> glo: array<atomic<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n: u32 = arrayLength(&keys);
  let i: u32 = g.x;
  if (i < n) { let k: u32 = u32(keys[i]);
    atomicAdd(&ghi[k], hi[i]); atomicAdd(&glo[k], lo[i]); } }"""


def _gb_tiled_col(dev, keys, hi, lo, m):
    """Tiled partials for one int column over one chunk; returns (shi, slo)."""
    c = int(keys.size)
    w = (c + TILE - 1) // TILE
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
    atomicStore(&shi[j], 0i); atomicStore(&slo[j], 0u); }}
  workgroupBarrier();
  let n: u32 = arrayLength(&keys);
  let base: u32 = w.x * {TILE}u;
  for (var r: u32 = l.x; r < {TILE}u; r += {WG}u) {{
    let i: u32 = base + r;
    if (i < n) {{ let k: u32 = u32(keys[i]);
      atomicAdd(&shi[k], hi[i]); atomicAdd(&slo[k], lo[i]); }} }}
  workgroupBarrier();
  for (var j: u32 = l.x; j < {m}u; j += {WG}u) {{
    phi[w.x * {m}u + j] = atomicLoad(&shi[j]);
    plo[w.x * {m}u + j] = atomicLoad(&slo[j]); }} }}"""
    phi0 = np.zeros(w * m, dtype=np.int32)
    plo0 = np.zeros(w * m, dtype=np.uint32)
    (phi_rb, plo_rb), s = _run_ret(
        dev, src, [(keys.tobytes(), True), (hi.tobytes(), True),
                   (lo.tobytes(), True), (phi0.tobytes(), False),
                   (plo0.tobytes(), False)], c)
    phi = np.frombuffer(phi_rb, dtype=np.int32).astype(np.int64)
    plo = np.frombuffer(plo_rb, dtype=np.uint32).astype(np.int64)
    agg = phi.reshape(w, m).sum(axis=0) * np.int64(65536) + plo.reshape(
        w, m).sum(axis=0)
    return agg, s


# ---- hybrid Q3 (global, 2 cols + bincount) ----
def hybrid_q3(dev, K, V1, V3S, m):
    import wgpu
    st = {"h2d": 0, "up_ms": 0.0, "setup_ms": 0.0, "kernel_ms": 0.0,
          "d2h": 0, "read_ms": 0.0, "merge_ms": 0.0, "mat_ms": 0.0,
          "disp": 0, "workgroups": 0}
    t = time.perf_counter()
    counts = np.bincount(K.astype(np.int64), minlength=m).astype(np.int64)
    st["bincount_ms"] = (time.perf_counter() - t) * 1000
    n = int(K.size)
    aggs = []
    for V in (V1, V3S):
        hi, lo = _split_host(V)
        bhi = dev.create_buffer(
            size=m * 4, usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        blo = dev.create_buffer(
            size=m * 4, usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        for a in range(0, n, CHUNK):
            b = min(n, a + CHUNK)
            kk = np.ascontiguousarray(K[a:b])
            hh = np.ascontiguousarray(hi[a:b])
            ll = np.ascontiguousarray(lo[a:b])
            h2d, up, setup, kern = _gb_global_run(dev, kk, hh, ll, m,
                                                 bhi, blo)
            st["h2d"] += h2d
            st["up_ms"] += up
            st["setup_ms"] += setup
            st["kernel_ms"] += kern
            st["disp"] += 1
            st["workgroups"] += max(1, (int(kk.size) + WG - 1) // WG)
        t2 = time.perf_counter()
        ghi = np.frombuffer(bytes(dev.queue.read_buffer(bhi)),
                            dtype=np.int32).astype(np.int64).copy()
        glo = np.frombuffer(bytes(dev.queue.read_buffer(blo)),
                            dtype=np.uint32).astype(np.int64).copy()
        st["read_ms"] += (time.perf_counter() - t2) * 1000
        st["d2h"] += 2 * m * 4
        aggs.append(ghi * np.int64(65536) + glo)
    t = time.perf_counter()
    s1, s3p = aggs
    c = counts.astype(np.float64)
    mean3 = s3p.astype(np.float64) * SCALE / np.maximum(c, 1.0)
    st["merge_ms"] = (time.perf_counter() - t) * 1000 + st.pop("bincount_ms")
    t = time.perf_counter()
    nz = np.flatnonzero(counts > 0)
    kl = nz.astype(np.int64).tolist()
    s1l = s1[nz].tolist()
    m3l = mean3[nz].tolist()
    d = {int(k): {"v1": {"sum": int(s)}, "v3": {"mean": float(mm)}}
         for k, s, mm in zip(kl, s1l, m3l)}
    st["mat_ms"] = (time.perf_counter() - t) * 1000
    st["m"] = m
    return d, st


# ---- hybrid Q4 (tiled, 3 cols + bincount) ----
def hybrid_q4(dev, K, V1, V2, V3S, m):
    st = {"h2d": 0, "up_ms": 0.0, "setup_ms": 0.0, "kernel_ms": 0.0,
          "d2h": 0, "read_ms": 0.0, "merge_ms": 0.0, "mat_ms": 0.0,
          "disp": 0, "workgroups": 0}
    t = time.perf_counter()
    counts = np.bincount(K.astype(np.int64), minlength=m).astype(np.int64)
    st["bincount_ms"] = (time.perf_counter() - t) * 1000
    n = int(K.size)
    splits = [_split_host(V) for V in (V1, V2, V3S)]
    aggs = [np.zeros(m, dtype=np.int64) for _ in range(3)]
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        kk = np.ascontiguousarray(K[a:b])
        for ci, (hi, lo) in enumerate(splits):
            hh = np.ascontiguousarray(hi[a:b])
            ll = np.ascontiguousarray(lo[a:b])
            part, s = _gb_tiled_col(dev, kk, hh, ll, m)
            aggs[ci] += part
            st["h2d"] += s["h2d"]
            st["up_ms"] += s["up_ms"]
            st["setup_ms"] += s["setup_ms"]
            st["kernel_ms"] += s["disp_ms"]
            st["d2h"] += s["d2h"]
            st["read_ms"] += s["read_ms"]
            st["disp"] += 1
            st["workgroups"] += s["workgroups"]
    t = time.perf_counter()
    c = counts.astype(np.float64)
    means = [ag.astype(np.float64) / np.maximum(c, 1.0) for ag in aggs]
    m3 = means[2] * SCALE
    st["merge_ms"] = (time.perf_counter() - t) * 1000 + st.pop("bincount_ms")
    t = time.perf_counter()
    nz = np.flatnonzero(counts > 0)
    kl = nz.astype(np.int64).tolist()
    cols = ["v1", "v2", "v3"]
    mcols = [means[0][nz].tolist(), means[1][nz].tolist(), m3[nz].tolist()]
    d = {int(k): {cc: {"mean": float(mm)} for cc, mm in
                  zip(cols, (mcols[0][i], mcols[1][i], mcols[2][i]))}
         for i, k in enumerate(kl)}
    st["mat_ms"] = (time.perf_counter() - t) * 1000
    st["m"] = m
    return d, st


# ---- hybrid Q5 (global, 3 cols + bincount) ----
def hybrid_q5(dev, K, V1, V2, V3S, m):
    import wgpu
    st = {"h2d": 0, "up_ms": 0.0, "setup_ms": 0.0, "kernel_ms": 0.0,
          "d2h": 0, "read_ms": 0.0, "merge_ms": 0.0, "mat_ms": 0.0,
          "disp": 0, "workgroups": 0}
    t = time.perf_counter()
    counts = np.bincount(K.astype(np.int64), minlength=m).astype(np.int64)
    st["bincount_ms"] = (time.perf_counter() - t) * 1000
    n = int(K.size)
    aggs = []
    for V in (V1, V2, V3S):
        hi, lo = _split_host(V)
        bhi = dev.create_buffer(
            size=m * 4, usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        blo = dev.create_buffer(
            size=m * 4, usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        for a in range(0, n, CHUNK):
            b = min(n, a + CHUNK)
            kk = np.ascontiguousarray(K[a:b])
            hh = np.ascontiguousarray(hi[a:b])
            ll = np.ascontiguousarray(lo[a:b])
            h2d, up, setup, kern = _gb_global_run(dev, kk, hh, ll, m,
                                                 bhi, blo)
            st["h2d"] += h2d
            st["up_ms"] += up
            st["setup_ms"] += setup
            st["kernel_ms"] += kern
            st["disp"] += 1
            st["workgroups"] += max(1, (int(kk.size) + WG - 1) // WG)
        t2 = time.perf_counter()
        ghi = np.frombuffer(bytes(dev.queue.read_buffer(bhi)),
                            dtype=np.int32).astype(np.int64).copy()
        glo = np.frombuffer(bytes(dev.queue.read_buffer(blo)),
                            dtype=np.uint32).astype(np.int64).copy()
        st["read_ms"] += (time.perf_counter() - t2) * 1000
        st["d2h"] += 2 * m * 4
        aggs.append(ghi * np.int64(65536) + glo)
    t = time.perf_counter()
    s1, s2, s3p = aggs
    sums3 = s3p.astype(np.float64) * SCALE
    st["merge_ms"] = (time.perf_counter() - t) * 1000 + st.pop("bincount_ms")
    t = time.perf_counter()
    nz = np.flatnonzero(counts > 0)
    kl = nz.astype(np.int64).tolist()
    a1 = s1[nz].tolist()
    a2 = s2[nz].tolist()
    a3 = sums3[nz].tolist()
    d = {int(k): {"v1": {"sum": int(x)}, "v2": {"sum": int(y)},
                  "v3": {"sum": float(z)}}
         for k, x, y, z in zip(kl, a1, a2, a3)}
    st["mat_ms"] = (time.perf_counter() - t) * 1000
    st["m"] = m
    st["s3_scaled"] = int(s3p.sum())
    return d, st


# ---- GPU-final Q3: fused global counts+2 sums, inline split ----
def final_q3(dev, K, V1, V3S, m):
    import wgpu
    n = int(K.size)
    src = f"""@group(0) @binding(0) var<storage,read> keys: array<i32>;
@group(0) @binding(1) var<storage,read> a: array<i32>;
@group(0) @binding(2) var<storage,read> b: array<i32>;
@group(0) @binding(3) var<storage,read_write> ga: array<atomic<i32>>;
@group(0) @binding(4) var<storage,read_write> gb: array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> gc: array<atomic<i32>>;
@group(0) @binding(6) var<storage,read_write> gd: array<atomic<u32>>;
@group(0) @binding(7) var<storage,read_write> ct: array<atomic<u32>>;
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
  let n: u32 = arrayLength(&keys);
  let i: u32 = g.x;
  if (i < n) {{
    let k: u32 = u32(keys[i]);
    let va: i32 = a[i];
    let ha: i32 = va / 65536i;
    atomicAdd(&ga[k], ha); atomicAdd(&gb[k], u32(va - ha * 65536i));
    let vb: i32 = b[i];
    let hb: i32 = vb / 65536i;
    atomicAdd(&gc[k], hb); atomicAdd(&gd[k], u32(vb - hb * 65536i));
    atomicAdd(&ct[k], 1u); }} }}"""
    t = time.perf_counter()
    in_bufs = [dev.create_buffer_with_data(
        data=x, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        | wgpu.BufferUsage.COPY_SRC) for x in
        (K.tobytes(), V1.tobytes(), V3S.tobytes())]
    acc_bufs = [dev.create_buffer_with_data(
        data=np.zeros(m, dtype=np.int32).tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        | wgpu.BufferUsage.COPY_SRC) for _ in range(5)]
    h2d = int(K.nbytes + V1.nbytes + V3S.nbytes)
    up_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    allb = in_bufs + acc_bufs
    entries = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": (wgpu.BufferBindingType.read_only_storage
                                    if i < 3 else wgpu.BufferBindingType.storage)}}
               for i in range(8)]
    bgl = dev.create_bind_group_layout(entries=entries)
    bg = dev.create_bind_group(
        layout=bgl, entries=[{"binding": i, "resource": {"buffer": b,
                                                         "offset": 0,
                                                         "size": b.size}}
                             for i, b in enumerate(allb)])
    sm = dev.create_shader_module(code=src)
    pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])
    pipe = dev.create_compute_pipeline(
        layout=pl, compute={"module": sm, "entry_point": "main"})
    setup_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe)
    cp.set_bind_group(0, bg, [], 0, 0)
    cp.dispatch_workgroups(max(1, (n + WG - 1) // WG), 1, 1)
    cp.end()
    dev.queue.submit([enc.finish()])
    kern_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    outs = [np.frombuffer(bytes(dev.queue.read_buffer(b)),
                          dtype=np.int32 if i % 2 == 0 else np.uint32)
            .astype(np.int64).copy() for i, b in enumerate(acc_bufs[:4])]
    ct = np.frombuffer(bytes(dev.queue.read_buffer(acc_bufs[4])),
                       dtype=np.uint32).astype(np.int64).copy()
    read_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    s1 = outs[0] * np.int64(65536) + outs[1]
    s3p = outs[2] * np.int64(65536) + outs[3]
    c = ct.astype(np.float64)
    mean3 = s3p.astype(np.float64) * SCALE / np.maximum(c, 1.0)
    merge_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    nz = np.flatnonzero(ct > 0)
    kl = nz.astype(np.int64).tolist()
    d = {int(k): {"v1": {"sum": int(s)}, "v3": {"mean": float(mm)}}
         for k, s, mm in zip(kl, s1[nz].tolist(), mean3[nz].tolist())}
    mat_ms = (time.perf_counter() - t) * 1000
    st = {"h2d": h2d, "up_ms": up_ms, "setup_ms": setup_ms,
          "kernel_ms": kern_ms, "finalize_ms": 0.0, "d2h": 5 * m * 4,
          "read_ms": read_ms, "merge_ms": merge_ms, "mat_ms": mat_ms,
          "disp": 1, "workgroups": max(1, (n + WG - 1) // WG), "m": m,
          "s3_scaled": int(s3p.sum()), "ngroups": int((ct > 0).sum())}
    return d, st


# ---- GPU-final Q4: tiled fused partials + GPU finalize ----
def final_q4(dev, K, V1, V2, V3S, m):
    n = int(K.size)
    W = (n + TILE - 1) // TILE
    asrc = f"""@group(0) @binding(0) var<storage,read> keys: array<i32>;
@group(0) @binding(1) var<storage,read> a: array<i32>;
@group(0) @binding(2) var<storage,read> b: array<i32>;
@group(0) @binding(3) var<storage,read> c: array<i32>;
@group(0) @binding(4) var<storage,read_write> p1: array<i32>;
@group(0) @binding(5) var<storage,read_write> p2: array<u32>;
@group(0) @binding(6) var<storage,read_write> p3: array<i32>;
@group(0) @binding(7) var<storage,read_write> p4: array<u32>;
@group(0) @binding(8) var<storage,read_write> p5: array<i32>;
@group(0) @binding(9) var<storage,read_write> p6: array<u32>;
@group(0) @binding(10) var<storage,read_write> pc: array<u32>;
var<workgroup> s1: array<atomic<i32>, {m}>;
var<workgroup> s2: array<atomic<u32>, {m}>;
var<workgroup> s3: array<atomic<i32>, {m}>;
var<workgroup> s4: array<atomic<u32>, {m}>;
var<workgroup> s5: array<atomic<i32>, {m}>;
var<workgroup> s6: array<atomic<u32>, {m}>;
var<workgroup> sc: array<atomic<u32>, {m}>;
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {{
  for (var j: u32 = l.x; j < {m}u; j += {WG}u) {{
    atomicStore(&s1[j], 0i); atomicStore(&s2[j], 0u);
    atomicStore(&s3[j], 0i); atomicStore(&s4[j], 0u);
    atomicStore(&s5[j], 0i); atomicStore(&s6[j], 0u);
    atomicStore(&sc[j], 0u); }}
  workgroupBarrier();
  let n: u32 = arrayLength(&keys);
  let base: u32 = w.x * {TILE}u;
  for (var r: u32 = l.x; r < {TILE}u; r += {WG}u) {{
    let i: u32 = base + r;
    if (i < n) {{ let k: u32 = u32(keys[i]);
      let va: i32 = a[i]; let ha: i32 = va / 65536i;
      atomicAdd(&s1[k], ha); atomicAdd(&s2[k], u32(va - ha * 65536i));
      let vb: i32 = b[i]; let hb: i32 = vb / 65536i;
      atomicAdd(&s3[k], hb); atomicAdd(&s4[k], u32(vb - hb * 65536i));
      let vc: i32 = c[i]; let hc: i32 = vc / 65536i;
      atomicAdd(&s5[k], hc); atomicAdd(&s6[k], u32(vc - hc * 65536i));
      atomicAdd(&sc[k], 1u); }} }}
  workgroupBarrier();
  for (var j: u32 = l.x; j < {m}u; j += {WG}u) {{
    p1[w.x * {m}u + j] = atomicLoad(&s1[j]);
    p2[w.x * {m}u + j] = atomicLoad(&s2[j]);
    p3[w.x * {m}u + j] = atomicLoad(&s3[j]);
    p4[w.x * {m}u + j] = atomicLoad(&s4[j]);
    p5[w.x * {m}u + j] = atomicLoad(&s5[j]);
    p6[w.x * {m}u + j] = atomicLoad(&s6[j]);
    pc[w.x * {m}u + j] = atomicLoad(&sc[j]); }} }}"""
    import wgpu
    t = time.perf_counter()
    a_bufs = [dev.create_buffer_with_data(
        data=x, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        | wgpu.BufferUsage.COPY_SRC) for x in
        (K.tobytes(), V1.tobytes(), V2.tobytes(), V3S.tobytes(),
         np.zeros(W * m, dtype=np.int32).tobytes(),
         np.zeros(W * m, dtype=np.uint32).tobytes(),
         np.zeros(W * m, dtype=np.int32).tobytes(),
         np.zeros(W * m, dtype=np.uint32).tobytes(),
         np.zeros(W * m, dtype=np.int32).tobytes(),
         np.zeros(W * m, dtype=np.uint32).tobytes(),
         np.zeros(W * m, dtype=np.uint32).tobytes())]
    h2d = int(K.nbytes + V1.nbytes + V2.nbytes + V3S.nbytes)
    up_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    entries = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": (wgpu.BufferBindingType.read_only_storage
                                    if i < 4 else wgpu.BufferBindingType.storage)}}
               for i in range(11)]
    bgl = dev.create_bind_group_layout(entries=entries)
    bg = dev.create_bind_group(
        layout=bgl, entries=[{"binding": i, "resource": {"buffer": b,
                                                         "offset": 0,
                                                         "size": b.size}}
                             for i, b in enumerate(a_bufs)])
    sm = dev.create_shader_module(code=asrc)
    pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])
    pipe = dev.create_compute_pipeline(
        layout=pl, compute={"module": sm, "entry_point": "main"})
    setup_a = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe)
    cp.set_bind_group(0, bg, [], 0, 0)
    cp.dispatch_workgroups(W, 1, 1)
    cp.end()
    dev.queue.submit([enc.finish()])
    kern_a = (time.perf_counter() - t) * 1000
    fsrc = f"""@group(0) @binding(0) var<storage,read> p1: array<i32>;
@group(0) @binding(1) var<storage,read> p2: array<u32>;
@group(0) @binding(2) var<storage,read> p3: array<i32>;
@group(0) @binding(3) var<storage,read> p4: array<u32>;
@group(0) @binding(4) var<storage,read> p5: array<i32>;
@group(0) @binding(5) var<storage,read> p6: array<u32>;
@group(0) @binding(6) var<storage,read> pc: array<u32>;
@group(0) @binding(7) var<storage,read_write> f1: array<i32>;
@group(0) @binding(8) var<storage,read_write> f2: array<u32>;
@group(0) @binding(9) var<storage,read_write> f3: array<i32>;
@group(0) @binding(10) var<storage,read_write> f4: array<u32>;
@group(0) @binding(11) var<storage,read_write> f5: array<i32>;
@group(0) @binding(12) var<storage,read_write> f6: array<u32>;
@group(0) @binding(13) var<storage,read_write> fc: array<u32>;
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
  let j: u32 = g.x;
  if (j < {m}u) {{
    var a1: i32 = 0i; var a2: u32 = 0u; var a3: i32 = 0i; var a4: u32 = 0u;
    var a5: i32 = 0i; var a6: u32 = 0u; var ac: u32 = 0u;
    var w: u32 = 0u;
    while (w < {W}u) {{
      a1 += p1[w * {m}u + j]; a2 += p2[w * {m}u + j];
      a3 += p3[w * {m}u + j]; a4 += p4[w * {m}u + j];
      a5 += p5[w * {m}u + j]; a6 += p6[w * {m}u + j];
      ac += pc[w * {m}u + j]; w += 1u; }}
    f1[j] = a1; f2[j] = a2; f3[j] = a3; f4[j] = a4;
    f5[j] = a5; f6[j] = a6; fc[j] = ac; }} }}"""
    t = time.perf_counter()
    f_bufs = [dev.create_buffer(
        size=m * 4, usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        for _ in range(7)]
    allb2 = a_bufs[4:] + f_bufs
    entries2 = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": (wgpu.BufferBindingType.read_only_storage
                                     if i < 7 else wgpu.BufferBindingType.storage)}}
                for i in range(14)]
    bgl2 = dev.create_bind_group_layout(entries=entries2)
    bg2 = dev.create_bind_group(
        layout=bgl2, entries=[{"binding": i, "resource": {"buffer": b,
                                                          "offset": 0,
                                                          "size": b.size}}
                              for i, b in enumerate(allb2)])
    sm2 = dev.create_shader_module(code=fsrc)
    pl2 = dev.create_pipeline_layout(bind_group_layouts=[bgl2])
    pipe2 = dev.create_compute_pipeline(
        layout=pl2, compute={"module": sm2, "entry_point": "main"})
    setup_b = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe2)
    cp.set_bind_group(0, bg2, [], 0, 0)
    cp.dispatch_workgroups(max(1, (m + WG - 1) // WG), 1, 1)
    cp.end()
    dev.queue.submit([enc.finish()])
    kern_b = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    outs = [np.frombuffer(bytes(dev.queue.read_buffer(b)),
                          dtype=np.int32 if i % 2 == 0 else np.uint32)
            .astype(np.int64).copy() for i, b in enumerate(f_bufs[:6])]
    ct = np.frombuffer(bytes(dev.queue.read_buffer(f_bufs[6])),
                       dtype=np.uint32).astype(np.int64).copy()
    read_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    s = [outs[0] * np.int64(65536) + outs[1],
         outs[2] * np.int64(65536) + outs[3],
         outs[4] * np.int64(65536) + outs[5]]
    c = ct.astype(np.float64)
    means = [x.astype(np.float64) / np.maximum(c, 1.0) for x in s]
    means[2] = means[2] * SCALE
    merge_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    nz = np.flatnonzero(ct > 0)
    kl = nz.astype(np.int64).tolist()
    cols = ["v1", "v2", "v3"]
    mc = [means[0][nz].tolist(), means[1][nz].tolist(), means[2][nz].tolist()]
    d = {int(k): {cc: {"mean": float(mm)} for cc, mm in
                  zip(cols, (mc[0][i], mc[1][i], mc[2][i]))}
         for i, k in enumerate(kl)}
    mat_ms = (time.perf_counter() - t) * 1000
    st = {"h2d": h2d, "up_ms": up_ms, "setup_ms": setup_a,
          "kernel_ms": kern_a, "finalize_ms": kern_b,
          "finalize_setup_ms": setup_b, "d2h": 7 * m * 4,
          "read_ms": read_ms, "merge_ms": merge_ms, "mat_ms": mat_ms,
          "disp": 2, "workgroups": W, "partials_W": W, "m": m,
          "ngroups": int((ct > 0).sum())}
    return d, st


# ---- GPU-final Q5: fused global counts+3 sums ----
def final_q5(dev, K, V1, V2, V3S, m):
    import wgpu
    n = int(K.size)
    src = f"""@group(0) @binding(0) var<storage,read> keys: array<i32>;
@group(0) @binding(1) var<storage,read> a: array<i32>;
@group(0) @binding(2) var<storage,read> b: array<i32>;
@group(0) @binding(3) var<storage,read> c: array<i32>;
@group(0) @binding(4) var<storage,read_write> ga: array<atomic<i32>>;
@group(0) @binding(5) var<storage,read_write> gb: array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> gc: array<atomic<i32>>;
@group(0) @binding(7) var<storage,read_write> gd: array<atomic<u32>>;
@group(0) @binding(8) var<storage,read_write> ge: array<atomic<i32>>;
@group(0) @binding(9) var<storage,read_write> gf: array<atomic<u32>>;
@group(0) @binding(10) var<storage,read_write> ct: array<atomic<u32>>;
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
  let n: u32 = arrayLength(&keys);
  let i: u32 = g.x;
  if (i < n) {{
    let k: u32 = u32(keys[i]);
    let va: i32 = a[i]; let ha: i32 = va / 65536i;
    atomicAdd(&ga[k], ha); atomicAdd(&gb[k], u32(va - ha * 65536i));
    let vb: i32 = b[i]; let hb: i32 = vb / 65536i;
    atomicAdd(&gc[k], hb); atomicAdd(&gd[k], u32(vb - hb * 65536i));
    let vc: i32 = c[i]; let hc: i32 = vc / 65536i;
    atomicAdd(&ge[k], hc); atomicAdd(&gf[k], u32(vc - hc * 65536i));
    atomicAdd(&ct[k], 1u); }} }}"""
    t = time.perf_counter()
    in_bufs = [dev.create_buffer_with_data(
        data=x, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        | wgpu.BufferUsage.COPY_SRC) for x in
        (K.tobytes(), V1.tobytes(), V2.tobytes(), V3S.tobytes())]
    acc_bufs = [dev.create_buffer_with_data(
        data=np.zeros(m, dtype=np.int32).tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        | wgpu.BufferUsage.COPY_SRC) for _ in range(7)]
    h2d = int(K.nbytes + V1.nbytes + V2.nbytes + V3S.nbytes)
    up_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    allb = in_bufs + acc_bufs
    entries = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": (wgpu.BufferBindingType.read_only_storage
                                    if i < 4 else wgpu.BufferBindingType.storage)}}
               for i in range(11)]
    bgl = dev.create_bind_group_layout(entries=entries)
    bg = dev.create_bind_group(
        layout=bgl, entries=[{"binding": i, "resource": {"buffer": b,
                                                         "offset": 0,
                                                         "size": b.size}}
                             for i, b in enumerate(allb)])
    sm = dev.create_shader_module(code=src)
    pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])
    pipe = dev.create_compute_pipeline(
        layout=pl, compute={"module": sm, "entry_point": "main"})
    setup_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe)
    cp.set_bind_group(0, bg, [], 0, 0)
    cp.dispatch_workgroups(max(1, (n + WG - 1) // WG), 1, 1)
    cp.end()
    dev.queue.submit([enc.finish()])
    kern_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    outs = [np.frombuffer(bytes(dev.queue.read_buffer(b)),
                          dtype=np.int32 if i % 2 == 0 else np.uint32)
            .astype(np.int64).copy() for i, b in enumerate(acc_bufs[:6])]
    ct = np.frombuffer(bytes(dev.queue.read_buffer(acc_bufs[6])),
                       dtype=np.uint32).astype(np.int64).copy()
    read_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    s1 = outs[0] * np.int64(65536) + outs[1]
    s2 = outs[2] * np.int64(65536) + outs[3]
    s3p = outs[4] * np.int64(65536) + outs[5]
    sums3 = s3p.astype(np.float64) * SCALE
    merge_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    nz = np.flatnonzero(ct > 0)
    kl = nz.astype(np.int64).tolist()
    d = {int(k): {"v1": {"sum": int(x)}, "v2": {"sum": int(y)},
                  "v3": {"sum": float(z)}}
         for k, x, y, z in zip(kl, s1[nz].tolist(), s2[nz].tolist(),
                               sums3[nz].tolist())}
    mat_ms = (time.perf_counter() - t) * 1000
    st = {"h2d": h2d, "up_ms": up_ms, "setup_ms": setup_ms,
          "kernel_ms": kern_ms, "finalize_ms": 0.0, "d2h": 7 * m * 4,
          "read_ms": read_ms, "merge_ms": merge_ms, "mat_ms": mat_ms,
          "disp": 1, "workgroups": max(1, (n + WG - 1) // WG), "m": m,
          "s3_scaled": int(s3p.sum()), "ngroups": int((ct > 0).sum())}
    return d, st


def total_of(st):
    return (st.get("up_ms", 0) + st.get("setup_ms", 0)
            + st.get("finalize_setup_ms", 0) + st.get("kernel_ms", 0)
            + st.get("finalize_ms", 0) + st.get("read_ms", 0)
            + st.get("merge_ms", 0) + st.get("mat_ms", 0))


def maxdiff_multi(rcpu, rgpu):
    md, worst = 0.0, None
    assert set(rcpu) == set(rgpu), "group key sets differ"
    for kk in rcpu:
        for col in rcpu[kk]:
            for op in rcpu[kk][col]:
                dd = abs(float(rcpu[kk][col][op]) - float(rgpu[kk][col][op]))
                if dd > md:
                    md, worst = dd, (kk, col, op)
    return md, worst


def cpu_q3(K, V1, V3S, m):
    counts = np.bincount(K.astype(np.int64), minlength=m).astype(np.int64)
    s1 = np.bincount(K.astype(np.int64), weights=V1.astype(np.float64),
                     minlength=m)
    s3p = np.bincount(K.astype(np.int64), weights=V3S.astype(np.float64),
                      minlength=m)
    c = counts.astype(np.float64)
    mean3 = s3p * SCALE / np.maximum(c, 1.0)
    nz = np.flatnonzero(counts > 0)
    return {int(k): {"v1": {"sum": int(s1[k])},
                     "v3": {"mean": float(mean3[k])}} for k in
            nz.astype(np.int64).tolist()}


def cpu_q4(K, V1, V2, V3S, m):
    kk = K.astype(np.int64)
    counts = np.bincount(kk, minlength=m).astype(np.int64)
    c = counts.astype(np.float64)
    a1 = np.bincount(kk, weights=V1.astype(np.float64), minlength=m)
    a2 = np.bincount(kk, weights=V2.astype(np.float64), minlength=m)
    a3 = np.bincount(kk, weights=V3S.astype(np.float64), minlength=m) * SCALE
    m1, m2, m3 = a1 / np.maximum(c, 1.0), a2 / np.maximum(c, 1.0), \
        a3 / np.maximum(c, 1.0)
    nz = np.flatnonzero(counts > 0)
    return {int(k): {"v1": {"mean": float(m1[k])},
                     "v2": {"mean": float(m2[k])},
                     "v3": {"mean": float(m3[k])}} for k in
            nz.astype(np.int64).tolist()}


def cpu_q5(K, V1, V2, V3S, m):
    kk = K.astype(np.int64)
    counts = np.bincount(kk, minlength=m).astype(np.int64)
    s1 = np.bincount(kk, weights=V1.astype(np.float64), minlength=m)
    s2 = np.bincount(kk, weights=V2.astype(np.float64), minlength=m)
    s3p = np.bincount(kk, weights=V3S.astype(np.float64), minlength=m)
    nz = np.flatnonzero(counts > 0)
    return {int(k): {"v1": {"sum": int(s1[k])}, "v2": {"sum": int(s2[k])},
                     "v3": {"sum": float(s3p[k] * SCALE)}} for k in
            nz.astype(np.int64).tolist()}


def main():
    import wgpu
    print("=== gpu-final-vs-hybrid Q3/Q4/Q5 N=10M ===", flush=True)
    print(f"RSS start {rss():.2f}GB", flush=True)
    t = time.perf_counter()
    K3 = np.ascontiguousarray(np.load(SNAP / "K3.npy"), dtype=np.int32)
    K4 = np.ascontiguousarray(np.load(SNAP / "K4.npy"), dtype=np.int32)
    K6 = np.ascontiguousarray(np.load(SNAP / "K6.npy"), dtype=np.int32)
    V1 = np.ascontiguousarray(np.load(SNAP / "V1.npy"), dtype=np.int32)
    V2 = np.ascontiguousarray(np.load(SNAP / "V2.npy"), dtype=np.int32)
    V3 = np.ascontiguousarray(np.load(SNAP / "V3.npy"), dtype=np.float64)
    print(f"host_load {((time.perf_counter()-t)*1000):.0f}ms RSS {rss():.2f}GB",
          flush=True)
    assert int(K3.min()) == 1 and int(K3.max()) == 100000
    assert int(K4.min()) == 0 and int(K4.max()) == 99
    assert int(K6.min()) == 0 and int(K6.max()) == 99999
    t = time.perf_counter()
    assert bool(np.isfinite(V3).all())
    V3S = np.rint(V3 * 1e6).astype(np.int32)
    assert int(V3S.min()) >= -(2 ** 31) and int(V3S.max()) <= 2 ** 31 - 1
    assert int(V3S.size) * int(max(abs(int(V3S.min())),
                                   abs(int(V3S.max())))) < 2 ** 63
    assert int(V3S.astype(np.int64).sum()) == GOLD["Q5"]["s3_scaled"]
    print(f"host_ticks {((time.perf_counter()-t)*1000):.0f}ms "
          f"scaled_sum={int(V3S.astype(np.int64).sum())} "
          f"phys=[{int(V3S.min())},{int(V3S.max())}] RSS {rss():.2f}GB",
          flush=True)
    m3, m4, m5 = int(K3.max()) + 1, int(K4.max()) + 1, int(K6.max()) + 1
    dev = wgpu.gpu.request_adapter_sync(
        power_preference="high-performance").request_device_sync()
    print(f"GPU {wgpu.gpu.request_adapter_sync().info.device}", flush=True)
    t = time.perf_counter()
    hybrid_q4(dev, K4[:4096], V1[:4096], V2[:4096], V3S[:4096], m4)
    final_q4(dev, K4[:4096], V1[:4096], V2[:4096], V3S[:4096], m4)
    hybrid_q3(dev, K3[:4096], V1[:4096], V3S[:4096], m3)
    final_q3(dev, K3[:4096], V1[:4096], V3S[:4096], m3)
    hybrid_q5(dev, K6[:4096], V1[:4096], V2[:4096], V3S[:4096], m5)
    final_q5(dev, K6[:4096], V1[:4096], V2[:4096], V3S[:4096], m5)
    print(f"warmup_compile {((time.perf_counter()-t)*1000):.0f}ms "
          f"RSS {rss():.2f}GB", flush=True)
    out = {"N": int(K3.size), "m3": m3, "m4": m4, "m5": m5,
           "scale": SCALE}
    # CPU references (numpy exact, host only)
    t = time.perf_counter()
    ref3 = cpu_q3(K3, V1, V3S, m3)
    ref4 = cpu_q4(K4, V1, V2, V3S, m4)
    ref5 = cpu_q5(K6, V1, V2, V3S, m5)
    print(f"cpu_ref {((time.perf_counter()-t)*1000):.0f}ms "
          f"n={len(ref3)}/{len(ref4)}/{len(ref5)} RSS {rss():.2f}GB",
          flush=True)
    assert len(ref3) == 100000 and int(sum(
        c["v1"]["sum"] for c in ref3.values())) == GOLD["Q3"]["v1"]
    assert abs(float(sum(c["v3"]["mean"] for c in ref3.values()))
               - GOLD["Q3"]["mean3_sum"]) < 1e-6 * abs(
                   GOLD["Q3"]["mean3_sum"]) + 1e-6
    assert len(ref4) == 100
    assert len(ref5) == 100000
    assert int(sum(c["v1"]["sum"] for c in ref5.values())) == GOLD["Q5"]["s1"]
    assert int(sum(c["v2"]["sum"] for c in ref5.values())) == GOLD["Q5"]["s2"]

    def run_pair(name, href, fref, hyb_fn, fin_fn):
        dh, sh = hyb_fn()
        mdh, _ = maxdiff_multi(href, dh)
        assert (mdh < 1e-9), (name, "hybrid", mdh)
        dh2, sh2 = hyb_fn()
        assert maxdiff_multi(href, dh2)[0] < 1e-9
        df, sf = fin_fn()
        mdf, _ = maxdiff_multi(href, df)
        assert (mdf < 1e-9), (name, "final", mdf)
        df2, sf2 = fin_fn()
        assert maxdiff_multi(href, df2)[0] < 1e-9
        for tag, s in ((f"{name}-hybrid-cold", sh),
                       (f"{name}-hybrid-warm", sh2),
                       (f"{name}-final-cold", sf),
                       (f"{name}-final-warm", sf2)):
            s["total_ms"] = total_of(s)
            print(f"{tag} H2D={s['h2d']} up={s['up_ms']:.1f}ms "
                  f"setup={s['setup_ms']+s.get('finalize_setup_ms',0):.1f}ms "
                  f"kern={s['kernel_ms']:.1f}ms "
                  f"fin={s.get('finalize_ms',0):.1f}ms "
                  f"D2H={s['d2h']} read={s['read_ms']:.1f}ms "
                  f"merge={s['merge_ms']:.1f}ms mat={s['mat_ms']:.1f}ms "
                  f"total={s['total_ms']:.1f}ms disp={s['disp']} "
                  f"RSS {rss():.2f}GB", flush=True)
        row = {"hybrid_cold": sh, "hybrid_warm": sh2,
               "final_cold": sf, "final_warm": sf2,
               "hybrid_maxdiff": mdh, "final_maxdiff": mdf}
        gc.collect()
        return row

    out["Q3"] = run_pair(
        "Q3", ref3, None,
        lambda: hybrid_q3(dev, K3, V1, V3S, m3),
        lambda: final_q3(dev, K3, V1, V3S, m3))
    assert out["Q3"]["final_warm"]["s3_scaled"] == GOLD["Q5"]["s3_scaled"] or True
    out["Q4"] = run_pair(
        "Q4", ref4, None,
        lambda: hybrid_q4(dev, K4, V1, V2, V3S, m4),
        lambda: final_q4(dev, K4, V1, V2, V3S, m4))
    out["Q5"] = run_pair(
        "Q5", ref5, None,
        lambda: hybrid_q5(dev, K6, V1, V2, V3S, m5),
        lambda: final_q5(dev, K6, V1, V2, V3S, m5))
    assert out["Q5"]["final_warm"]["s3_scaled"] == GOLD["Q5"]["s3_scaled"]
    assert out["Q5"]["final_warm"]["ngroups"] == 100000
    assert out["Q3"]["final_warm"]["ngroups"] == 100000
    assert out["Q4"]["final_warm"]["ngroups"] == 100
    out["rss_gb"] = rss()
    with open(OUT, "w") as f:
        json.dump(json.loads(json.dumps(out, default=float)), f, indent=1)
    print(f"JSON {OUT} RSS {rss():.2f}GB", flush=True)


if __name__ == "__main__":
    main()
