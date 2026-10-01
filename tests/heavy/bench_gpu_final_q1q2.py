# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU-final vs hybrid Q1/Q2, H2O 10M G1_1e7_1e2_0_0, RTX 2060 (NEW, bench-only).

Hybrid (current arch): H2D(keys+hi+lo) -> GPU tiled/global partials ->
D2H partials -> CPU bincount + host int64 merge -> materialize.
GPU-final (experimental, same WGSL arithmetic/atomics, moved, not tuned):
  Q1: H2D(keys+vals) -> GPU tiled partials (inline split + counts) ->
      GPU finalize -> D2H finals(M) -> tiny host combine -> materialize.
  Q2: H2D(c1+c2+vals) -> GPU fused pack+global accum (inline split+counts) ->
      D2H finals(M) -> tiny host combine -> materialize.
No pack P round-trip, no hi/lo upload, no CPU bincount in final path.
Correctness exact (V1 in 1..5 non-negative -> hv==0 path exact; dict equality
vs hybrid + total/ngroups asserts). Stages: H2D|kernel|finalize|D2H|merge|
materialize|total. Resident warm saved separately.

Usage (Git Bash, sequential, timeout):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    timeout 550 /c/Users/Mikech/AppData/Local/Programs/Python/Python314/python.exe \
    tests/heavy/bench_gpu_final_q1q2.py
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
OUT = FORK / "tests" / "heavy" / "bench_gpu_final_q1q2.json"
WG = 256
TILE = 2048
CHUNK = 1 << 20


def rss():
    return PROC.memory_info().rss / 1e9


def _run_ret(dev, src, io_specs, n_threads):
    """Single dispatch, raw readbacks. Timers split by caller stages."""
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


# ---- hybrid Q1 (mirror Drivers/GPU tiled path, chunked) ----
def hybrid_q1(dev, K, V, m):
    st = {"h2d": 0, "up_ms": 0.0, "setup_ms": 0.0, "kernel_ms": 0.0,
          "d2h": 0, "read_ms": 0.0, "merge_ms": 0.0, "mat_ms": 0.0,
          "disp": 0}
    t = time.perf_counter()
    counts = np.bincount(K.astype(np.int64), minlength=m).astype(np.int64)
    st["bincount_ms"] = (time.perf_counter() - t) * 1000
    hi, lo = _split_host(V)
    st["hsplit_ms"] = 0.0
    agg = np.zeros(m, dtype=np.int64)
    t = time.perf_counter()
    n = int(K.size)
    wsum = 0
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        kk = np.ascontiguousarray(K[a:b])
        hh = np.ascontiguousarray(hi[a:b])
        ll = np.ascontiguousarray(lo[a:b])
        c = int(kk.size)
        w = (c + TILE - 1) // TILE
        wsum += w
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
            dev, src, [(kk.tobytes(), True), (hh.tobytes(), True),
                       (ll.tobytes(), True), (phi0.tobytes(), False),
                       (plo0.tobytes(), False)], c)
        st["h2d"] += s["h2d"]
        st["up_ms"] += s["up_ms"]
        st["setup_ms"] += s["setup_ms"]
        st["kernel_ms"] += s["disp_ms"]
        st["d2h"] += s["d2h"]
        st["read_ms"] += s["read_ms"]
        st["disp"] += 1
        t2 = time.perf_counter()
        phi = np.frombuffer(phi_rb, dtype=np.int32).astype(np.int64)
        plo = np.frombuffer(plo_rb, dtype=np.uint32).astype(np.int64)
        agg += phi.reshape(w, m).sum(axis=0) * np.int64(65536) + plo.reshape(
            w, m).sum(axis=0)
        st["merge_ms"] += (time.perf_counter() - t2) * 1000
    st["merge_ms"] += st.pop("bincount_ms")
    st["workgroups"] = wsum
    t = time.perf_counter()
    nz = np.flatnonzero(counts > 0)
    d = {int(k): int(v) for k, v in zip(nz.astype(np.int64).tolist(),
                                        agg[nz].tolist())}
    st["mat_ms"] = (time.perf_counter() - t) * 1000
    return d, st


# ---- hybrid Q2 (pack + global accum, chunked; mirror current) ----
def hybrid_q2(dev, K1, K2, V, m2):
    st = {"h2d": 0, "up_ms": 0.0, "setup_ms": 0.0, "kernel_ms": 0.0,
          "d2h": 0, "read_ms": 0.0, "merge_ms": 0.0, "mat_ms": 0.0,
          "disp": 0}
    import wgpu
    n = int(K1.size)
    psrc = f"""@group(0) @binding(0) var<storage,read> k1: array<i32>;
@group(0) @binding(1) var<storage,read> k2: array<i32>;
@group(0) @binding(2) var<storage,read_write> o: array<i32>;
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
  let i: u32 = g.x;
  if (i < {n}u) {{ o[i] = k1[i] * {int(m2)}i + k2[i]; }} }}"""
    t = time.perf_counter()
    (p_rb,), s = _run_ret(dev, psrc,
                           [(K1.tobytes(), True), (K2.tobytes(), True),
                            (np.zeros(n, dtype=np.int32).tobytes(), False)],
                           n)
    st["pack_ms"] = (time.perf_counter() - t) * 1000
    st["pack_h2d"] = s["h2d"]
    st["pack_d2h"] = s["d2h"]
    st["pack_read_ms"] = s["read_ms"]
    st["h2d"] += s["h2d"]
    st["up_ms"] += s["up_ms"]
    st["setup_ms"] += s["setup_ms"]
    st["kernel_ms"] += s["disp_ms"]
    st["d2h"] += s["d2h"]
    st["read_ms"] += s["read_ms"]
    st["disp"] += 1
    P = np.frombuffer(p_rb, dtype=np.int32).copy()
    m = int(P.max()) + 1
    t = time.perf_counter()
    counts = np.bincount(P.astype(np.int64), minlength=m).astype(np.int64)
    st["bincount_ms"] = (time.perf_counter() - t) * 1000
    gsrc = """@group(0) @binding(0) var<storage,read> keys: array<i32>;
@group(0) @binding(1) var<storage,read> hi: array<i32>;
@group(0) @binding(2) var<storage,read> lo: array<u32>;
@group(0) @binding(3) var<storage,read_write> ghi: array<atomic<i32>>;
@group(0) @binding(4) var<storage,read_write> glo: array<atomic<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
  let n: u32 = arrayLength(&keys);
  let i: u32 = g.x;
  if (i < n) {{ let k: u32 = u32(keys[i]);
    atomicAdd(&ghi[k], hi[i]); atomicAdd(&glo[k], lo[i]); }} }}"""
    bhi = dev.create_buffer(
        size=m * 4, usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
    blo = dev.create_buffer(
        size=m * 4, usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        kk = np.ascontiguousarray(P[a:b])
        vv = np.ascontiguousarray(V[a:b])
        hh, ll = _split_host(vv)
        t2 = time.perf_counter()
        in_bufs = [dev.create_buffer_with_data(
            data=x, usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
            for x in (kk.tobytes(), hh.tobytes(), ll.tobytes())]
        st["h2d"] += sum(len(x) for x in (kk.tobytes(), hh.tobytes(),
                                          ll.tobytes()))
        st["up_ms"] += (time.perf_counter() - t2) * 1000
        t2 = time.perf_counter()
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
        sm = dev.create_shader_module(code=gsrc)
        pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])
        pipe = dev.create_compute_pipeline(
            layout=pl, compute={"module": sm, "entry_point": "main"})
        st["setup_ms"] += (time.perf_counter() - t2) * 1000
        t2 = time.perf_counter()
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, bg, [], 0, 0)
        cp.dispatch_workgroups(max(1, (int(kk.size) + WG - 1) // WG), 1, 1)
        cp.end()
        dev.queue.submit([enc.finish()])
        st["kernel_ms"] += (time.perf_counter() - t2) * 1000
        st["workgroups"] = st.get("workgroups", 0) + max(
            1, (int(kk.size) + WG - 1) // WG)
        st["disp"] += 1
    t = time.perf_counter()
    ghi = np.frombuffer(bytes(dev.queue.read_buffer(bhi)),
                        dtype=np.int32).astype(np.int64).copy()
    glo = np.frombuffer(bytes(dev.queue.read_buffer(blo)),
                        dtype=np.uint32).astype(np.int64).copy()
    st["read_ms"] += (time.perf_counter() - t) * 1000
    st["d2h"] += 2 * m * 4
    t = time.perf_counter()
    agg = ghi * np.int64(65536) + glo
    st["merge_ms"] = (time.perf_counter() - t) * 1000 + st.pop("bincount_ms")
    t = time.perf_counter()
    nz = np.flatnonzero(counts > 0)
    d = {int(k): int(v) for k, v in zip(nz.astype(np.int64).tolist(),
                                        agg[nz].tolist())}
    st["mat_ms"] = (time.perf_counter() - t) * 1000
    st["m"] = m
    return d, st


# ---- GPU-final Q1: tiled partials (inline split+counts) + GPU finalize ----
def final_q1(dev, K, V, m):
    import wgpu
    n = int(K.size)
    W = (n + TILE - 1) // TILE
    st = {}
    asrc = f"""@group(0) @binding(0) var<storage,read> keys: array<i32>;
@group(0) @binding(1) var<storage,read> vals: array<i32>;
@group(0) @binding(2) var<storage,read_write> phi: array<i32>;
@group(0) @binding(3) var<storage,read_write> plo: array<u32>;
@group(0) @binding(4) var<storage,read_write> pct: array<u32>;
var<workgroup> shi: array<atomic<i32>, {m}>;
var<workgroup> slo: array<atomic<u32>, {m}>;
var<workgroup> sct: array<atomic<u32>, {m}>;
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {{
  for (var j: u32 = l.x; j < {m}u; j += {WG}u) {{
    atomicStore(&shi[j], 0i); atomicStore(&slo[j], 0u);
    atomicStore(&sct[j], 0u); }}
  workgroupBarrier();
  let n: u32 = arrayLength(&keys);
  let base: u32 = w.x * {TILE}u;
  for (var r: u32 = l.x; r < {TILE}u; r += {WG}u) {{
    let i: u32 = base + r;
    if (i < n) {{ let k: u32 = u32(keys[i]);
      let vv: i32 = vals[i];
      let hv: i32 = vv / 65536i;
      let lv: u32 = u32(vv - hv * 65536i);
      atomicAdd(&shi[k], hv); atomicAdd(&slo[k], lv);
      atomicAdd(&sct[k], 1u); }} }}
  workgroupBarrier();
  for (var j: u32 = l.x; j < {m}u; j += {WG}u) {{
    phi[w.x * {m}u + j] = atomicLoad(&shi[j]);
    plo[w.x * {m}u + j] = atomicLoad(&slo[j]);
    pct[w.x * {m}u + j] = atomicLoad(&sct[j]); }} }}"""
    t = time.perf_counter()
    a_bufs = [dev.create_buffer_with_data(
        data=x, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        | wgpu.BufferUsage.COPY_SRC) for x in
        (K.tobytes(), V.tobytes(),
         np.zeros(W * m, dtype=np.int32).tobytes(),
         np.zeros(W * m, dtype=np.uint32).tobytes(),
         np.zeros(W * m, dtype=np.uint32).tobytes())]
    h2d_a = int(K.nbytes + V.nbytes)
    up_a = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    allb = a_bufs
    entries = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": (wgpu.BufferBindingType.read_only_storage
                                    if i < 2 else wgpu.BufferBindingType.storage)}}
               for i in range(5)]
    bgl = dev.create_bind_group_layout(entries=entries)
    bg = dev.create_bind_group(
        layout=bgl, entries=[{"binding": i, "resource": {"buffer": b,
                                                         "offset": 0,
                                                         "size": b.size}}
                             for i, b in enumerate(allb)])
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
    # finalize on GPU: partials resident -> finals M
    fsrc = f"""@group(0) @binding(0) var<storage,read> phi: array<i32>;
@group(0) @binding(1) var<storage,read> plo: array<u32>;
@group(0) @binding(2) var<storage,read> pct: array<u32>;
@group(0) @binding(3) var<storage,read_write> fhi: array<i32>;
@group(0) @binding(4) var<storage,read_write> flo: array<u32>;
@group(0) @binding(5) var<storage,read_write> fct: array<u32>;
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
  let j: u32 = g.x;
  if (j < {m}u) {{
    var ah: i32 = 0i; var al: u32 = 0u; var ac: u32 = 0u;
    var w: u32 = 0u;
    while (w < {W}u) {{
      ah += phi[w * {m}u + j]; al += plo[w * {m}u + j];
      ac += pct[w * {m}u + j]; w += 1u; }}
    fhi[j] = ah; flo[j] = al; fct[j] = ac; }} }}"""
    t = time.perf_counter()
    f_bufs = [dev.create_buffer(
        size=m * 4, usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        for _ in range(3)]
    allb2 = a_bufs[2:] + f_bufs
    entries2 = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": (wgpu.BufferBindingType.read_only_storage
                                     if i < 3 else wgpu.BufferBindingType.storage)}}
                for i in range(6)]
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
    fhi = np.frombuffer(bytes(dev.queue.read_buffer(f_bufs[0])),
                        dtype=np.int32).astype(np.int64).copy()
    flo = np.frombuffer(bytes(dev.queue.read_buffer(f_bufs[1])),
                        dtype=np.uint32).astype(np.int64).copy()
    fct = np.frombuffer(bytes(dev.queue.read_buffer(f_bufs[2])),
                        dtype=np.uint32).astype(np.int64).copy()
    read_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    agg = fhi * np.int64(65536) + flo
    merge_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    nz = np.flatnonzero(fct > 0)
    d = {int(k): int(v) for k, v in zip(nz.astype(np.int64).tolist(),
                                        agg[nz].tolist())}
    mat_ms = (time.perf_counter() - t) * 1000
    st = {"h2d": h2d_a, "up_ms": up_a, "setup_ms": setup_a,
          "kernel_ms": kern_a, "finalize_ms": kern_b,
          "finalize_setup_ms": setup_b, "d2h": 3 * m * 4,
          "read_ms": read_ms, "merge_ms": merge_ms, "mat_ms": mat_ms,
          "disp": 2, "workgroups": W,
          "partials_W": W, "m": m,
          "counts": fct}
    return d, st


# ---- GPU-final Q2: fused pack + global accum (inline split+counts) ----
def final_q2(dev, K1, K2, V, m2, m):
    import wgpu
    n = int(K1.size)
    src = f"""@group(0) @binding(0) var<storage,read> c1: array<i32>;
@group(0) @binding(1) var<storage,read> c2: array<i32>;
@group(0) @binding(2) var<storage,read> vals: array<i32>;
@group(0) @binding(3) var<storage,read_write> ghi: array<atomic<i32>>;
@group(0) @binding(4) var<storage,read_write> glo: array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> gct: array<atomic<u32>>;
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
  let n: u32 = arrayLength(&vals);
  let i: u32 = g.x;
  if (i < n) {{
    let k: u32 = u32(c1[i] * {int(m2)}i + c2[i]);
    let vv: i32 = vals[i];
    let hv: i32 = vv / 65536i;
    let lv: u32 = u32(vv - hv * 65536i);
    atomicAdd(&ghi[k], hv); atomicAdd(&glo[k], lv);
    atomicAdd(&gct[k], 1u); }} }}"""
    t = time.perf_counter()
    in_bufs = [dev.create_buffer_with_data(
        data=x, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        | wgpu.BufferUsage.COPY_SRC) for x in
        (K1.tobytes(), K2.tobytes(), V.tobytes())]
    acc_bufs = [dev.create_buffer_with_data(
        data=np.zeros(m, dtype=np.int32).tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        | wgpu.BufferUsage.COPY_SRC) for _ in range(3)]
    h2d = int(K1.nbytes + K2.nbytes + V.nbytes) + 3 * m * 4
    up_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    allb = in_bufs + acc_bufs
    entries = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": (wgpu.BufferBindingType.read_only_storage
                                    if i < 3 else wgpu.BufferBindingType.storage)}}
               for i in range(6)]
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
    ghi = np.frombuffer(bytes(dev.queue.read_buffer(acc_bufs[0])),
                        dtype=np.int32).astype(np.int64).copy()
    glo = np.frombuffer(bytes(dev.queue.read_buffer(acc_bufs[1])),
                        dtype=np.uint32).astype(np.int64).copy()
    gct = np.frombuffer(bytes(dev.queue.read_buffer(acc_bufs[2])),
                        dtype=np.uint32).astype(np.int64).copy()
    read_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    agg = ghi * np.int64(65536) + glo
    merge_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    nz = np.flatnonzero(gct > 0)
    d = {int(k): int(v) for k, v in zip(nz.astype(np.int64).tolist(),
                                        agg[nz].tolist())}
    mat_ms = (time.perf_counter() - t) * 1000
    st = {"h2d": h2d, "up_ms": up_ms, "setup_ms": setup_ms,
          "kernel_ms": kern_ms, "finalize_ms": 0.0, "d2h": 3 * m * 4,
          "read_ms": read_ms, "merge_ms": merge_ms, "mat_ms": mat_ms,
          "disp": 1, "workgroups": max(1, (n + WG - 1) // WG), "m": m,
          "counts": gct}
    return d, st


def total_of(st):
    return (st.get("up_ms", 0) + st.get("setup_ms", 0)
            + st.get("finalize_setup_ms", 0) + st.get("kernel_ms", 0)
            + st.get("finalize_ms", 0) + st.get("read_ms", 0)
            + st.get("pack_read_ms", 0) + st.get("merge_ms", 0)
            + st.get("mat_ms", 0))


def main():
    import wgpu
    print("=== gpu-final-vs-hybrid Q1/Q2 N=10M ===", flush=True)
    print(f"RSS start {rss():.2f}GB", flush=True)
    t = time.perf_counter()
    K1 = np.ascontiguousarray(np.load(SNAP / "K1.npy"), dtype=np.int32)
    K2 = np.ascontiguousarray(np.load(SNAP / "K2.npy"), dtype=np.int32)
    V1 = np.ascontiguousarray(np.load(SNAP / "V1.npy"), dtype=np.int32)
    print(f"host_prep {((time.perf_counter()-t)*1000):.0f}ms "
          f"bytes={int(K1.nbytes+K2.nbytes+V1.nbytes)} RSS {rss():.2f}GB",
          flush=True)
    assert int(V1.astype(np.int64).sum()) == 29998789
    m1 = int(K1.max()) + 1
    m2 = int(K2.max()) + 1
    mq2 = m1 * m2 if int(K1.min()) >= 0 else None
    print(f"m1={m1} m2={m2} RSS {rss():.2f}GB", flush=True)
    dev = wgpu.gpu.request_adapter_sync(
        power_preference="high-performance").request_device_sync()
    print(f"GPU {wgpu.gpu.request_adapter_sync().info.device}", flush=True)
    # warmup each shader outside timers
    t = time.perf_counter()
    hybrid_q1(dev, K1[:4096], V1[:4096], m1)
    final_q1(dev, K1[:4096], V1[:4096], m1)
    hybrid_q2(dev, K1[:4096], K2[:4096], V1[:4096], m2)
    final_q2(dev, K1[:4096], K2[:4096], V1[:4096], m2, m1 * m2)
    print(f"warmup_compile {((time.perf_counter()-t)*1000):.0f}ms "
          f"RSS {rss():.2f}GB", flush=True)
    out = {"N": int(K1.size), "m1": m1, "m2": m2}
    # Q1 hybrid then final (cold, warm); Q2 same. Sequential.
    dh, sh = hybrid_q1(dev, K1, V1, m1)
    assert len(dh) == 100 and int(sum(dh.values())) == 29998789
    dh2, sh2 = hybrid_q1(dev, K1, V1, m1)
    assert dh2 == dh
    df, sf = final_q1(dev, K1, V1, m1)
    assert df == dh, (len(df), int(sum(df.values())))
    df2, sf2 = final_q1(dev, K1, V1, m1)
    assert df2 == dh
    for tag, s in (("Q1-hybrid-cold", sh), ("Q1-hybrid-warm", sh2),
                   ("Q1-final-cold", sf), ("Q1-final-warm", sf2)):
        s["total_ms"] = total_of(s)
        print(f"{tag} H2D={s['h2d']} up={s['up_ms']:.1f}ms "
              f"setup={s['setup_ms']+s.get('finalize_setup_ms',0):.1f}ms "
              f"kern={s['kernel_ms']:.1f}ms fin={s.get('finalize_ms',0):.1f}ms "
              f"D2H={s['d2h']} read={s['read_ms']+s.get('pack_read_ms',0):.1f}ms "
              f"merge={s['merge_ms']:.1f}ms mat={s['mat_ms']:.1f}ms "
              f"total={s['total_ms']:.1f}ms disp={s['disp']} "
              f"RSS {rss():.2f}GB", flush=True)
    out["Q1"] = {"hybrid_cold": sh, "hybrid_warm": sh2,
                 "final_cold": sf, "final_warm": sf2,
                 "chk_total": 29998789, "ngroups": 100}
    gc.collect()
    dh, sh = hybrid_q2(dev, K1, K2, V1, m2)
    assert len(dh) == 10000 and int(sum(dh.values())) == 29998789
    dh2, sh2 = hybrid_q2(dev, K1, K2, V1, m2)
    assert dh2 == dh
    df, sf = final_q2(dev, K1, K2, V1, m2, m1 * m2)
    # label check: fused pack k=c1*m2+c2 matches hybrid pack labels
    assert df == dh, (len(df), int(sum(df.values())))
    df2, sf2 = final_q2(dev, K1, K2, V1, m2, m1 * m2)
    assert df2 == dh
    for tag, s in (("Q2-hybrid-cold", sh), ("Q2-hybrid-warm", sh2),
                   ("Q2-final-cold", sf), ("Q2-final-warm", sf2)):
        s["total_ms"] = total_of(s)
        print(f"{tag} H2D={s['h2d']} up={s['up_ms']:.1f}ms "
              f"setup={s['setup_ms']+s.get('finalize_setup_ms',0):.1f}ms "
              f"kern={s['kernel_ms']:.1f}ms fin={s.get('finalize_ms',0):.1f}ms "
              f"D2H={s['d2h']} read={s['read_ms']+s.get('pack_read_ms',0):.1f}ms "
              f"merge={s['merge_ms']:.1f}ms mat={s['mat_ms']:.1f}ms "
              f"pack={s.get('pack_ms',0):.1f}ms total={s['total_ms']:.1f}ms "
              f"disp={s['disp']} RSS {rss():.2f}GB", flush=True)
    out["Q2"] = {"hybrid_cold": sh, "hybrid_warm": sh2,
                 "final_cold": sf, "final_warm": sf2,
                 "chk_total": 29998789, "ngroups": 10000}
    # resident warm: arrays resident, no reload; usual: reload one col + query
    gc.collect()
    t = time.perf_counter()
    d, s = final_q1(dev, K1, V1, m1)
    resident_q1 = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    d, s = final_q2(dev, K1, K2, V1, m2, m1 * m2)
    resident_q2 = (time.perf_counter() - t) * 1000
    assert int(sum(d.values())) == 29998789
    t = time.perf_counter()
    _r = np.load(SNAP / "K1.npy")
    reload_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    a1 = np.ascontiguousarray(np.load(SNAP / "K1.npy"), dtype=np.int32)
    a2 = np.ascontiguousarray(np.load(SNAP / "K2.npy"), dtype=np.int32)
    av = np.ascontiguousarray(np.load(SNAP / "V1.npy"), dtype=np.int32)
    d0, _ = final_q2(dev, a1, a2, av, m2, m1 * m2)
    usual_q2 = (time.perf_counter() - t) * 1000
    assert int(sum(d0.values())) == 29998789
    out["resident"] = {"resident_q1_ms": resident_q1,
                       "resident_q2_ms": resident_q2,
                       "reload_one_col_ms": reload_ms, "usual_q2_ms": usual_q2}
    print(f"resident Q1 {resident_q1:.0f}ms Q2 {resident_q2:.0f}ms "
          f"usual_Q2(reload+query) {usual_q2:.0f}ms RSS {rss():.2f}GB",
          flush=True)
    out["rss_gb"] = rss()
    for k in ("Q1", "Q2"):
        for rk in ("hybrid_cold", "hybrid_warm", "final_cold", "final_warm"):
            out[k][rk].pop("counts", None)
    with open(OUT, "w") as f:
        json.dump(json.loads(json.dumps(out, default=float)), f, indent=1)
    print(f"JSON {OUT} RSS {rss():.2f}GB", flush=True)


if __name__ == "__main__":
    main()
