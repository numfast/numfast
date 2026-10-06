# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""GPU driver: wgpu-py -> Vulkan/DX12, WGSL elementwise + reduce-partials.

Scope (v0.2 + DELTA-6 + SPEC-DELTA-7, honest): pure-GPU pack_keys/compare/mask/gather;
hybrid filter (CPU prefix + GPU gather) and reduce (GPU partials + CPU
int64/float64 final); groupby/groupby_multi DENSE hybrid (WGSL partial
histograms + host int64/f64 merge, DELTA-6); sort bitonic stable permutation
(single int32/finite-f32 key, pow2 N; multi-key/non-pow2/NaN/int64 raise an
explicit CPU-only error, spec 06 chunkable=false, no silent fallback).
Unscaled f64 rejected on GPU (MapF64 guard, spec 06): use float32 or CPU.
Scaled float64 (SPEC-DELTA-7: Schema scale/offset contract on series) runs as
exact integer WGSL on int32 physical ticks, f64 unscale on host for sum/mean.
No imports from other Extensions (flat assembly, Builder alias only).
"""

import os as _os

import numpy as _np

_WG = 256
_CTX = None


class DeviceContext:
    """SPEC-DELTA-9: persistent execution context (device+queue+pipelines).

    One adapter/device for process life; one compiled pipeline per
    (WGSL source, binding layout) -- compile once, dispatch many.
    Rule: N/K/M never baked into WGSL source on the uniform path;
    they ride a 16B uniform (runtime parameters), so the same pipeline
    serves any shape. Buffers outlive single dispatches (resident model).
    """

    def __init__(self):
        import wgpu
        a = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.dev = a.request_device_sync()
        self.pipes = {}
        self.hits = 0
        self.misses = 0

    def pipe(self, src, types):
        key = (src, tuple(types))
        p = self.pipes.get(key)
        if p is None:
            import wgpu
            dev = self.dev
            entries = [{"binding": i,
                        "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": t}} for i, t in enumerate(types)]
            bgl = dev.create_bind_group_layout(entries=entries)
            sm = dev.create_shader_module(code=src)
            pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])
            p = (dev.create_compute_pipeline(
                layout=pl, compute={"module": sm,
                                    "entry_point": "main"}), bgl)
            self.pipes[key] = p
            self.misses += 1
        else:
            self.hits += 1
        return p

    def stats(self):
        return {"pipelines": len(self.pipes), "hits": self.hits,
                "misses": self.misses}


def _ctx():
    global _CTX
    if _CTX is None:
        _CTX = DeviceContext()
    return _CTX


def pipeline_stats():
    """Cache evidence for benches: {pipelines, hits, misses}."""
    return _ctx().stats()


def _fallback_err(what, fix="", doc=""):
    msg = str(what)
    if fix:
        msg += f" Fix: {fix}."
    if doc:
        msg += f" See {doc}"
    return ValueError(msg)


def _device():
    return _ctx().dev


def _bufs(dev, parts):
    """parts: [(np_array_u8_bytes, readonly_bool)]. Returns wgpu buffers."""
    import wgpu
    out = []
    for data, ro in parts:
        b = dev.create_buffer_with_data(
            data=data,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
            | wgpu.BufferUsage.COPY_SRC,
        )
        out.append((b, ro))
    return out


def _bind_types(bufs, n_out=0, uniform_tail=0):
    import wgpu
    ts = [(wgpu.BufferBindingType.read_only_storage if ro
           else wgpu.BufferBindingType.storage) for _, ro in bufs]
    ts += [wgpu.BufferBindingType.storage] * n_out
    ts += [wgpu.BufferBindingType.uniform] * uniform_tail
    return ts


def _submit(dev, pipe, bgl, allb, n_threads):
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe)
    cp.set_bind_group(0, dev.create_bind_group(
        layout=bgl,
        entries=[{"binding": i, "resource": {"buffer": b, "offset": 0,
                                            "size": b.size}}
                 for i, b in enumerate(allb)]), [], 0, 0)
    cp.dispatch_workgroups(max(1, (int(n_threads) + _WG - 1) // _WG), 1, 1)
    cp.end()
    dev.queue.submit([enc.finish()])


def _run(dev, src, io_specs, n_threads, out_sizes):
    """io_specs: [(bytes, readonly)]. out_sizes: extra readback sizes.

    Returns list of readback bytes: one per io buffer that is writable,
    in order, plus trailing out-only buffers. Pipeline cached (Stage 1).
    """
    import wgpu
    bufs = _bufs(dev, io_specs)
    outs = []
    for sz in out_sizes:
        outs.append(dev.create_buffer(
            size=max(1, int(sz)),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        ))
    allb = [b for b, _ in bufs] + outs
    pipe, bgl = _ctx().pipe(src, _bind_types(bufs, len(outs)))
    _submit(dev, pipe, bgl, allb, n_threads)
    res = []
    for i, (b, ro) in enumerate(bufs):
        if not ro:
            res.append(bytes(dev.queue.read_buffer(b)))
    for b in outs:
        res.append(bytes(dev.queue.read_buffer(b)))
    return res


def _uni_bytes(vals):
    """4xu32 params -> 16B uniform payload (vec4<u32> compatible)."""
    return _np.ascontiguousarray(vals, dtype=_np.uint32).tobytes()


def _run_u(dev, src, io_specs, uni_vals, n_threads, out_sizes=()):
    """Cached dispatch with 16B uniform tail (N/K/M = runtime params).

    io_specs as _run; uni_vals: 4 uints; out_sizes: trailing out-only bufs.
    """
    import wgpu
    bufs = _bufs(dev, io_specs)
    outs = [dev.create_buffer(
        size=max(1, int(sz)),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        | wgpu.BufferUsage.COPY_DST) for sz in out_sizes]
    ub = dev.create_buffer_with_data(
        data=_uni_bytes(uni_vals),
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    allb = [b for b, _ in bufs] + outs + [ub]
    pipe, bgl = _ctx().pipe(src, _bind_types(bufs, len(outs), 1))
    _submit(dev, pipe, bgl, allb, n_threads)
    res = []
    for b, ro in bufs:
        if not ro:
            res.append(bytes(dev.queue.read_buffer(b)))
    for b in outs:
        res.append(bytes(dev.queue.read_buffer(b)))
    return res


def _dispatch_u(dev, src, bufs, uni_vals, n_threads):
    """Resident dispatch over live buffers + uniform (no alloc/readback)."""
    ub = dev.create_buffer_with_data(
        data=_uni_bytes(uni_vals),
        usage=__import__("wgpu").BufferUsage.UNIFORM
        | __import__("wgpu").BufferUsage.COPY_DST)
    allb = [b for b, _ in bufs] + [ub]
    import wgpu
    types = [(wgpu.BufferBindingType.read_only_storage if ro
              else wgpu.BufferBindingType.storage) for _, ro in bufs]
    types += [wgpu.BufferBindingType.uniform]
    pipe, bgl = _ctx().pipe(src, types)
    _submit(dev, pipe, bgl, allb, n_threads)


def r_upload(arr):
    """Resident upload: numpy -> device buffer (no readback, caller owns)."""
    import wgpu
    return _ctx().dev.create_buffer_with_data(
        data=_np.ascontiguousarray(arr).tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        | wgpu.BufferUsage.COPY_DST)


def r_alloc(nbytes):
    """Resident alloc: device-only buffer of nbytes (min 4, wgpu floor)."""
    import wgpu
    return _ctx().dev.create_buffer(
        size=max(4, int(nbytes)),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        | wgpu.BufferUsage.COPY_DST)


def r_download(buf, dtype, n=None):
    """Resident download: device buffer -> numpy copy (graph exit only)."""
    raw = bytes(_ctx().dev.queue.read_buffer(buf))
    a = _np.frombuffer(raw, dtype=dtype).copy()
    return a if n is None else a[:int(n)].copy()


def _i32(arr):
    return _np.ascontiguousarray(arr, dtype=_np.int32)


def _f32(arr):
    return _np.ascontiguousarray(arr, dtype=_np.float32)


def _mask_u32(m):
    a = _np.asarray(m)
    if a.dtype == _np.dtype(bool):
        return _np.ascontiguousarray(a.astype(_np.uint32)).view(_np.int32)
    if a.dtype == _np.dtype(_np.uint8):
        return _np.ascontiguousarray(a.astype(_np.int32))
    return _np.ascontiguousarray(a.astype(_np.int32))


_CMP_WGSL = {
    "==": "a[i] == bv", "!=": "a[i] != bv",
    "<": "a[i] < bv", "<=": "a[i] <= bv",
    ">": "a[i] > bv", ">=": "a[i] >= bv",
}


# ---- RNG CORE: counter-based Philox4x32-10 fill (int32, chunkable) ----
# Same rounds as numfast-native/src/rng.rs and CPU _lib/rng.py, u32
# mul/xor/shift/add ONLY (no u64 in WGSL: mul_hi via 16-bit halves;
# width < 2^32 always since lo/hi are int32, so Lemire v = mul_hi(r, w)
# is exact). Key (k0,k1) + stream words (c2,c3) derived COLD on host
# (local splitmix copy: GPU Extension never imports CPU _lib --
# flat-assembly rule, 1-2 helpers kept local). Lane i uses counter
# offset + i -> chunked (offset-adjusted) == unchunked bit-exact.
_RNG_FILL_I32 = """@group(0) @binding(0) var<storage,read> prm: array<u32>;
@group(0) @binding(1) var<storage,read_write> o: array<i32>;
fn mul_hi(a: u32, b: u32) -> u32 {
  let a0: u32 = a & 0xFFFFu;
  let a1: u32 = a >> 16u;
  let b0: u32 = b & 0xFFFFu;
  let b1: u32 = b >> 16u;
  let p0: u32 = a0 * b0;
  let p1: u32 = a0 * b1;
  let p2: u32 = a1 * b0;
  let p3: u32 = a1 * b1;
  let mid: u32 = (p0 >> 16u) + (p1 & 0xFFFFu) + (p2 & 0xFFFFu);
  return p3 + (p1 >> 16u) + (p2 >> 16u) + (mid >> 16u);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i: u32 = g.x;
  let n: u32 = arrayLength(&o);
  if (i >= n) { return; }
  let k0: u32 = prm[0];
  let k1: u32 = prm[1];
  let c2: u32 = prm[2];
  let c3: u32 = prm[3];
  let c0: u32 = i + prm[4];
  let carry: u32 = select(0u, 1u, c0 < i);
  let c1: u32 = prm[5] + carry;
  var a0: u32 = c0;
  var a1: u32 = c1;
  var a2: u32 = c2;
  var a3: u32 = c3;
  var kk0: u32 = k0;
  var kk1: u32 = k1;
  for (var r: u32 = 0u; r < 10u; r = r + 1u) {
    let hi0: u32 = mul_hi(a0, 0xD2511F53u);
    let lo0: u32 = a0 * 0xD2511F53u;
    let hi1: u32 = mul_hi(a2, 0xCD9E8D57u);
    let lo1: u32 = a2 * 0xCD9E8D57u;
    let b0: u32 = hi1 ^ a1 ^ kk0;
    let b1: u32 = lo1;
    let b2: u32 = hi0 ^ a3 ^ kk1;
    let b3: u32 = lo0;
    a0 = b0; a1 = b1; a2 = b2; a3 = b3;
    kk0 = kk0 + 0x9E3779B9u;
    kk1 = kk1 + 0xBB67AE85u;
  }
  let w: u32 = prm[7];
  let v: u32 = mul_hi(a0, w);
  o[i] = bitcast<i32>(prm[6]) + bitcast<i32>(v);
}"""


def _rng_splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = (((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF)
    z = (((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF)
    return state, (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def _rng_derive_key(seed, stream):
    st = ((int(seed) & 0xFFFFFFFFFFFFFFFF) + 0x9E3779B97F4A7C15
          + ((int(stream) & 0xFFFFFFFFFFFFFFFF) * 0xBF58476D1CE4E5B9
             & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    _, k = _rng_splitmix64(st)
    return (k & 0xFFFFFFFF, (k >> 32) & 0xFFFFFFFF)


def rng_fill_i32(n, seed, stream, offset, lo, hi):
    """WGSL Philox fill: out[i] = lo + mul_hi(draw, width) (== CPU bit-exact).

    Key/stream derived on host (COLD); per-lane counter offset + i.
    """
    n = int(n)
    if n == 0:
        return _np.zeros(0, dtype=_np.int32)
    width = int(hi) - int(lo)
    if width == 1:
        return _np.full(n, _np.int32(lo), dtype=_np.int32)
    k0, k1 = _rng_derive_key(seed, stream)
    s = int(stream) & 0xFFFFFFFFFFFFFFFF
    off = int(offset) & 0xFFFFFFFFFFFFFFFF
    import struct as _st
    prm = _np.array([k0, k1,
                     (s & 0xFFFFFFFF) ^ 0xC0FFEE00,
                     ((s >> 32) & 0xFFFFFFFF) ^ 0x9E3779B9,
                     off & 0xFFFFFFFF, (off >> 32) & 0xFFFFFFFF,
                     int(lo) & 0xFFFFFFFF, width & 0xFFFFFFFF],
                    dtype=_np.uint32)
    dev = _device()
    out = _np.zeros(n, dtype=_np.int32)
    (rb,) = _run(dev, _RNG_FILL_I32,
                 [(prm.tobytes(), True), (out.tobytes(), False)], n, [])
    return _np.frombuffer(rb, dtype=_np.int32).copy()


def pack_i32(k1, k2, m2):
    """WGSL pack: out[i] = k1[i]*m2 + k2[i] (int32-direct composite, uniform).

    SPEC-DELTA-9: N/m2 ride the uniform; pipeline cached across shapes.
    """
    k1, k2 = _i32(k1), _i32(k2)
    assert k1.size == k2.size
    n = int(k1.size)
    if n == 0:
        return _np.zeros(0, dtype=_np.int32)
    dev = _device()
    out = _np.zeros(n, dtype=_np.int32)
    (rb,) = _run_u(dev, _PACK_U, [(k1.tobytes(), True), (k2.tobytes(), True),
                                  (out.tobytes(), False)], (n, int(m2), 0, 0),
                   n)
    return _np.frombuffer(rb, dtype=_np.int32).copy()


def r_pack(b_k1, b_k2, m2, n, out=None):
    """Resident pack: device bufs -> device buf (no D2H, graph edge)."""
    dev = _device()
    b_o = out if out is not None else r_alloc(int(n) * 4)
    _dispatch_u(dev, _PACK_U, [(b_k1, True), (b_k2, True), (b_o, False)],
                (int(n), int(m2), 0, 0), int(n))
    return b_o


def compare(a, b_or_scalar, op, dtype):
    """WGSL compare -> u32 0/1 mask (i32/f32; f64 rejected). Uniform N.

    Scalar rides the uniform too (bitcast: i32 bits / f32 value), so the
    pipeline is shared across thresholds, not just shapes.
    """
    if op not in _CMP_WGSL:
        raise ValueError(f"unknown compare op '{op}'")
    n = int(_np.asarray(a).size)
    if n == 0:
        return _np.zeros(0, dtype=_np.uint32).view(_np.int32)
    dev = _device()
    expr = _CMP_WGSL[op]
    if dtype == "int32":
        A = _i32(a)
        if isinstance(b_or_scalar, _np.ndarray):
            B = _i32(b_or_scalar)
            src = _CMP_I_ARR_U.replace("__EXPR__",
                                       expr.replace("a[i]", "a[i]")
                                       .replace("bv", "b[i]"))
            ios = [(A.tobytes(), True), (B.tobytes(), True),
                   (_np.zeros(n, dtype=_np.int32).tobytes(), False)]
            uni = (n, 0, 0, 0)
        else:
            src = _CMP_I_SCAL_U.replace("__EXPR__", expr)
            ios = [(A.tobytes(), True),
                   (_np.zeros(n, dtype=_np.int32).tobytes(), False)]
            uni = (n, int(_np.int32(int(b_or_scalar)).view(_np.uint32)),
                   0, 0)
        (rb,) = _run_u(dev, src, ios, uni, n)
        return _np.frombuffer(rb, dtype=_np.int32).copy()
    if dtype == "float32":
        A = _f32(a)
        fexpr = expr.replace("a[i]", "a[i]")
        if isinstance(b_or_scalar, _np.ndarray):
            B = _f32(b_or_scalar)
            src = _CMP_F_ARR_U.replace("__EXPR__",
                                       fexpr.replace("bv", "b[i]"))
            ios = [(A.tobytes(), True), (B.tobytes(), True),
                   (_np.zeros(n, dtype=_np.int32).tobytes(), False)]
            # 2-uniform layout: p(n) + q unused; single-tail run needs the
            # array source with one uniform only -> use _run_u directly.
            (rb,) = _run_u(dev, src, ios, (n, 0, 0, 0), n)
            return _np.frombuffer(rb, dtype=_np.int32).copy()
        import struct as _st
        fbits = _st.unpack("<I", _st.pack("<f", float(b_or_scalar)))[0]
        # scalar-float source carries q uniform: custom two-tail dispatch.
        src = _CMP_F_SCAL_U.replace("__EXPR__", fexpr.replace("bv", "q.x"))
        import wgpu
        bufs = _bufs(dev, [(A.tobytes(), True),
                           (_np.zeros(n, dtype=_np.int32).tobytes(),
                            False)])
        ub1 = dev.create_buffer_with_data(
            data=_uni_bytes((n, 0, 0, 0)),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        ub2 = dev.create_buffer_with_data(
            data=_np.ascontiguousarray([float(b_or_scalar), 0, 0, 0],
                                       dtype=_np.float32).tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        allb = [b for b, _ in bufs] + [ub1, ub2]
        types = [wgpu.BufferBindingType.read_only_storage,
                 wgpu.BufferBindingType.storage,
                 wgpu.BufferBindingType.uniform,
                 wgpu.BufferBindingType.uniform]
        pipe, bgl = _ctx().pipe(src, types)
        _submit(dev, pipe, bgl, allb, n)
        _ = fbits
        res = []
        for b, ro in bufs:
            if not ro:
                res.append(bytes(dev.queue.read_buffer(b)))
        return _np.frombuffer(res[0], dtype=_np.int32).copy()
    raise ValueError(f"GPU compare: dtype '{dtype}' needs float32 or int32")


def mask_op(a, b, op):
    """WGSL BoolMask combine on 0/1 int32 (and/or/not). Uniform N."""
    A = _mask_u32(a)
    n = int(A.size)
    if n == 0:
        return _np.zeros(0, dtype=_np.int32)
    dev = _device()
    if op == "not":
        (rb,) = _run_u(dev, _MASK_NOT_U, [(A.tobytes(), True),
                                          (_np.zeros(n, dtype=_np.int32)
                                           .tobytes(), False)], (n, 0, 0, 0),
                       n)
    elif op in ("and", "or"):
        B = _mask_u32(b)
        assert B.size == n
        e = "min(a[i] + b[i], 1i)" if op == "or" else "a[i] * b[i]"
        src = _MASK_U.replace("__EXPR__", e)
        (rb,) = _run_u(dev, src, [(A.tobytes(), True), (B.tobytes(), True),
                                  (_np.zeros(n, dtype=_np.int32).tobytes(),
                                   False)], (n, 0, 0, 0), n)
    else:
        raise ValueError(f"unknown mask op '{op}'")
    return _np.frombuffer(rb, dtype=_np.int32).copy()


def r_compare_mask(a_buf, b_or_scalar_buf_or_int, op, dtype_is_f32, n,
                   out=None):
    """Resident compare->mask edge (no D2H). Scalar int only (graph path)."""
    dev = _device()
    b_o = out if out is not None else r_alloc(int(n) * 4)
    if isinstance(b_or_scalar_buf_or_int, int):
        src = (_CMP_I_SCAL_U if not dtype_is_f32 else None)
        if src is None:
            raise ValueError("resident float-scalar compare: use host path")
        sb = _np.int32(b_or_scalar_buf_or_int).view(_np.uint32)
        _dispatch_u(dev, src.replace("__EXPR__", _CMP_WGSL[op]),
                    [(a_buf, True), (b_o, False)],
                    (int(n), int(sb), 0, 0), int(n))
    else:
        t = _CMP_F_ARR_U if dtype_is_f32 else _CMP_I_ARR_U
        bv = "b[i]"
        _dispatch_u(dev, t.replace("__EXPR__",
                                   _CMP_WGSL[op].replace("bv", bv)),
                    [(a_buf, True), (b_or_scalar_buf_or_int, True),
                     (b_o, False)], (int(n), 0, 0, 0), int(n))
    return b_o


def r_mask(a_buf, b_buf_or_none, op, n, out=None):
    """Resident mask edge (no D2H)."""
    dev = _device()
    b_o = out if out is not None else r_alloc(int(n) * 4)
    if op == "not":
        _dispatch_u(dev, _MASK_NOT_U, [(a_buf, True), (b_o, False)],
                    (int(n), 0, 0, 0), int(n))
    else:
        e = "min(a[i] + b[i], 1i)" if op == "or" else "a[i] * b[i]"
        _dispatch_u(dev, _MASK_U.replace("__EXPR__", e),
                    [(a_buf, True), (b_buf_or_none, True), (b_o, False)],
                    (int(n), 0, 0, 0), int(n))
    return b_o


def gather(values, indices, dtype):
    """WGSL gather: out[i] = values[indices[i]] (uniform M)."""
    idx = _np.ascontiguousarray(indices, dtype=_np.int32)
    m = int(idx.size)
    if m == 0:
        return _np.zeros(0, dtype=_np.int32 if dtype == "int32"
                        else _np.float32)
    dev = _device()
    if dtype == "int32":
        V = _i32(values)
        t, np_t = "i32", _np.int32
    elif dtype == "float32":
        V = _f32(values)
        t, np_t = "f32", _np.float32
    else:
        raise ValueError(f"GPU gather: dtype '{dtype}' needs int32/float32")
    src = _GATHER_U.replace("__T__", t)
    out = _np.zeros(m, dtype=np_t)
    (rb,) = _run_u(dev, src, [(V.tobytes(), True), (idx.tobytes(), True),
                              (out.tobytes(), False)], (m, 0, 0, 0), m)
    return _np.frombuffer(rb, dtype=np_t).copy()


def r_gather(v_buf, ix_buf, m, dtype_is_f32=False, out=None):
    """Resident gather edge (no D2H)."""
    dev = _device()
    b_o = out if out is not None else r_alloc(int(m) * 4)
    _dispatch_u(dev, _GATHER_U.replace("__T__",
                                       "f32" if dtype_is_f32 else "i32"),
                [(v_buf, True), (ix_buf, True), (b_o, False)],
                (int(m), 0, 0, 0), int(m))
    return b_o


def slice_take(values, limit, offset, dtype):
    """WGSL slice: out[i] = values[offset+i] (LIMIT/OFFSET, resident).

    limit=None reads to end; offset beyond n yields empty (SQL semantics,
    same as CPU). Keeps dtype; validity sidecar sliced on host by caller.
    """
    V = _np.ascontiguousarray(values)
    n = int(V.size)
    off = int(offset)
    if off >= n:
        return _np.zeros(0, dtype=V.dtype)
    end = n if limit is None else min(n, off + int(limit))
    m = end - off
    if m == 0:
        return _np.zeros(0, dtype=V.dtype)
    if dtype == "int32":
        A = _i32(V)
        t, np_t = "i32", _np.int32
    elif dtype == "float32":
        A = _f32(V)
        t, np_t = "f32", _np.float32
    else:
        raise ValueError(f"GPU slice: dtype '{dtype}' needs int32/float32")
    dev = _device()
    src = _SLICE_U.replace("__T__", t)
    out = _np.zeros(m, dtype=np_t)
    (rb,) = _run_u(dev, src, [(A.tobytes(), True), (out.tobytes(), False)],
                   (m, off, 0, 0), m)
    return _np.frombuffer(rb, dtype=np_t).copy()


def r_slice(v_buf, m, off, out=None):
    """Resident slice edge (LIMIT/OFFSET, no D2H)."""
    dev = _device()
    b_o = out if out is not None else r_alloc(max(1, int(m)) * 4)
    _dispatch_u(dev, _SLICE_U.replace("__T__", "i32"),
                [(v_buf, True), (b_o, False)],
                (int(m), int(off), 0, 0), max(1, int(m)))
    return b_o


def shift_take(values, periods, dtype):
    """WGSL shift: out[i] = 0 for i < periods else values[i-periods].

    Right-shift only (periods>=0, IR-validated); memcpy semantics, so
    NaN/Inf ride bit-exact. Validity sidecar handled on host by caller
    (same carry rule as CPU _shift_ref). Empty/identity/all-invalid
    shapes are host-side (no dispatch); only 0 < periods < n runs WGSL.
    """
    V = _np.ascontiguousarray(values)
    n = int(V.size)
    per = int(periods)
    if dtype == "int32":
        A = _i32(V)
        t, np_t, zero = "i32", _np.int32, "0i"
    elif dtype == "float32":
        A = _f32(V)
        t, np_t, zero = "f32", _np.float32, "0.0f"
    else:
        raise ValueError(f"GPU shift: dtype '{dtype}' needs int32/float32")
    if n == 0:
        return _np.zeros(0, dtype=np_t)
    if per <= 0:
        return _np.ascontiguousarray(A)
    if per >= n:
        return _np.zeros(n, dtype=np_t)
    dev = _device()
    src = _SHIFT_U.replace("__T__", t).replace("__ZERO__", zero)
    out = _np.zeros(n, dtype=np_t)
    (rb,) = _run_u(dev, src, [(A.tobytes(), True), (out.tobytes(), False)],
                   (n, per, 0, 0), n)
    return _np.frombuffer(rb, dtype=np_t).copy()


_MAP_I_OPS = ("add", "sub", "mul", "floor_div", "mod")
_MAP_F_OPS = ("add", "sub", "mul")


def map_elem(a, b_or_scalar, fn, dtype):
    """WGSL elementwise IR `map` (i32/f32; f64 rejected by the MapF64 guard).

    Covered (CPU bit-exact): i32 add/sub/mul (wrapping) + floor_div/mod
    (Python-floor sign correction, zero divisor -> 0, NumPy rule);
    f32 add/sub/mul (IEEE). Scalar rides a 1-lane storage buffer, so the
    pipeline is shared across scalar values, not just shapes.
    Explicit gaps (ValueError, never silent): i32 div/pow (need the f64
    round-trip, not expressible in WGSL f32); f32 div/pow (the CPU
    contract widens them to float64 output); f32 floor_div/mod (NumPy
    computes them wide, WGSL has no exact fmod); f64 everything.
    Validity sidecars travel host-side (AND of inputs, same as CPU
    `_valid_mask`); this entry never sees validity.
    """
    if dtype == "int32":
        if fn not in _MAP_I_OPS:
            raise ValueError(
                f"GPU map: i32 fn '{fn}' needs CPU (div/pow ride the float64 "
                "round-trip, not expressible in WGSL)")
        A = _i32(a)
    elif dtype == "float32":
        if fn not in _MAP_F_OPS:
            raise ValueError(
                f"GPU map: f32 fn '{fn}' needs CPU (div/pow widen to float64 "
                "output; floor_div/mod are computed wide by NumPy)")
        A = _f32(a)
    else:
        raise ValueError(f"GPU map: dtype '{dtype}' needs int32/float32")
    n = int(A.size)
    if n == 0:
        return _np.zeros(0, dtype=A.dtype)
    t = "i32" if dtype == "int32" else "f32"
    np_t = _np.int32 if dtype == "int32" else _np.float32
    if isinstance(b_or_scalar, _np.ndarray) and b_or_scalar.ndim > 0:
        B = (_i32(b_or_scalar) if dtype == "int32"
             else _f32(b_or_scalar))
        if int(B.size) != n:
            raise ValueError(f"GPU map length mismatch {n} != {int(B.size)}")
        src = (_MAP_I_ARR_U if dtype == "int32" else _MAP_F_ARR_U).replace(
            "__T__", t).replace("__BODY__", _MAP_BODY[(dtype, fn)])
        ios = [(A.tobytes(), True), (B.tobytes(), True),
               (_np.zeros(n, dtype=np_t).tobytes(), False)]
    else:
        s = b_or_scalar.item() if isinstance(b_or_scalar, _np.ndarray) \
            else b_or_scalar
        if dtype == "int32":
            if isinstance(s, (bool, _np.bool_)):
                s = int(s)
            if not isinstance(s, (int, _np.integer)) or isinstance(
                    s, (float, _np.floating)) or not (-2 ** 31 <= int(s) <= 2 ** 31 - 1):
                raise ValueError(
                    f"GPU map: i32 lane needs an int32 scalar, got {s!r} "
                    "(float scalars ride float64 on CPU)")
            S = _np.ascontiguousarray([int(s)], dtype=_np.int32)
        else:
            try:
                f = float(s)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    f"GPU map: f32 lane needs a real scalar, got {s!r}")
            S = _np.ascontiguousarray([f], dtype=_np.float32)
        src = (_MAP_I_SCAL_U if dtype == "int32" else _MAP_F_SCAL_U).replace(
            "__T__", t).replace("__BODY_SCAL__", _MAP_SCAL_BODY[(dtype, fn)])
        ios = [(A.tobytes(), True), (S.tobytes(), True),
               (_np.zeros(n, dtype=np_t).tobytes(), False)]
    dev = _device()
    out = _np.zeros(n, dtype=np_t)
    (rb,) = _run_u(dev, src, ios, (n, 0, 0, 0), n)
    return _np.frombuffer(rb, dtype=np_t).copy()


# ---- Uniform WGSL (SPEC-DELTA-9): N/K/M = runtime params, never literals.
# One source per (op, dtype, arity); pipeline cached across shapes/values.
# P = vec4<u32> tail uniform: meaning per kernel, documented at each site.
_PACK_U = """@group(0) @binding(0) var<storage,read> k1: array<i32>;
@group(0) @binding(1) var<storage,read> k2: array<i32>;
@group(0) @binding(2) var<storage,read_write> o: array<i32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i: u32 = g.x;
  if (i < p.x) { o[i] = k1[i] * i32(p.y) + k2[i]; }
}"""

_CMP_I_SCAL_U = """@group(0) @binding(0) var<storage,read> a: array<i32>;
@group(0) @binding(1) var<storage,read_write> o: array<i32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i: u32 = g.x;
  if (i < p.x) {
    let bv: i32 = bitcast<i32>(p.y);
    o[i] = select(0i, 1i, __EXPR__);
  }
}"""

_CMP_I_ARR_U = """@group(0) @binding(0) var<storage,read> a: array<i32>;
@group(0) @binding(1) var<storage,read> b: array<i32>;
@group(0) @binding(2) var<storage,read_write> o: array<i32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i: u32 = g.x;
  if (i < p.x) {
    o[i] = select(0i, 1i, __EXPR__);
  }
}"""

_CMP_F_SCAL_U = """@group(0) @binding(0) var<storage,read> a: array<f32>;
@group(0) @binding(1) var<storage,read_write> o: array<i32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;
@group(0) @binding(3) var<uniform> q: vec4<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i: u32 = g.x;
  if (i < p.x) {
    o[i] = select(0i, 1i, __EXPR__);
  }
}"""

_CMP_F_ARR_U = """@group(0) @binding(0) var<storage,read> a: array<f32>;
@group(0) @binding(1) var<storage,read> b: array<f32>;
@group(0) @binding(2) var<storage,read_write> o: array<i32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i: u32 = g.x;
  if (i < p.x) {
    o[i] = select(0i, 1i, __EXPR__);
  }
}"""

_MASK_U = """@group(0) @binding(0) var<storage,read> a: array<i32>;
@group(0) @binding(1) var<storage,read> b: array<i32>;
@group(0) @binding(2) var<storage,read_write> o: array<i32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i: u32 = g.x;
  if (i < p.x) { o[i] = __EXPR__; }
}"""

_MASK_NOT_U = """@group(0) @binding(0) var<storage,read> a: array<i32>;
@group(0) @binding(1) var<storage,read_write> o: array<i32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i: u32 = g.x;
  if (i < p.x) { o[i] = select(1i, 0i, a[i] == 1i); }
}"""

_GATHER_U = """@group(0) @binding(0) var<storage,read> v: array<__T__>;
@group(0) @binding(1) var<storage,read> ix: array<i32>;
@group(0) @binding(2) var<storage,read_write> o: array<__T__>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i: u32 = g.x;
  if (i < p.x) { o[i] = v[u32(ix[i])]; }
}"""

_SLICE_U = """@group(0) @binding(0) var<storage,read> v: array<__T__>;
@group(0) @binding(1) var<storage,read_write> o: array<__T__>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i: u32 = g.x;
  if (i < p.x) { o[i] = v[p.y + i]; }
}"""

_SHIFT_U = """@group(0) @binding(0) var<storage,read> v: array<__T__>;
@group(0) @binding(1) var<storage,read_write> o: array<__T__>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i: u32 = g.x;
  if (i < p.x) {
    if (i < p.y) { o[i] = __ZERO__; } else { o[i] = v[i - p.y]; }
  }
}"""

# ---- Elementwise IR `map` (SPEC-DELTA-9 uniform style, same as shift). ----
# One source per (dtype, arity); the op is baked as __BODY__ statements
# (floor_div/mod need multi-statement sign correction, so __EXPR__ does
# not suffice). Scalar rides a 1-lane storage buffer `s` (binding 1):
# the pipeline is shared across scalar values, not just shapes.
# i32 overflow wraps (WGSL two's complement); i32 MIN/-1 is guarded
# (`-2147483647i - 1i`: the literal `-2147483648i` does not parse).
_MAP_I_ARR_U = """@group(0) @binding(0) var<storage,read> a: array<__T__>;
@group(0) @binding(1) var<storage,read> b: array<__T__>;
@group(0) @binding(2) var<storage,read_write> o: array<__T__>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i: u32 = g.x;
  if (i < p.x) {
__BODY__
  }
}"""

_MAP_I_SCAL_U = """@group(0) @binding(0) var<storage,read> a: array<__T__>;
@group(0) @binding(1) var<storage,read> s: array<__T__>;
@group(0) @binding(2) var<storage,read_write> o: array<__T__>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i: u32 = g.x;
  if (i < p.x) {
__BODY_SCAL__
  }
}"""

_MAP_F_ARR_U = _MAP_I_ARR_U
_MAP_F_SCAL_U = _MAP_I_SCAL_U

_MAP_I_ARR_BODY = {
    "add": "    o[i] = a[i] + b[i];",
    "sub": "    o[i] = a[i] - b[i];",
    "mul": "    o[i] = a[i] * b[i];",
    "floor_div": """    var q: i32 = 0;
    if (b[i] != 0i) {
      if (a[i] == (-2147483647i - 1i) && b[i] == -1i) { q = (-2147483647i - 1i); }
      else {
        let t: i32 = a[i] / b[i];
        let r: i32 = a[i] - t * b[i];
        q = t;
        if (r != 0i && ((r < 0i) != (b[i] < 0i))) { q = t - 1i; }
      }
    }
    o[i] = q;""",
    "mod": """    var m: i32 = 0;
    if (b[i] != 0i) {
      if (a[i] == (-2147483647i - 1i) && b[i] == -1i) { m = 0i; }
      else {
        let t: i32 = a[i] / b[i];
        var r: i32 = a[i] - t * b[i];
        if (r != 0i && ((r < 0i) != (b[i] < 0i))) { r = r + b[i]; }
        m = r;
      }
    }
    o[i] = m;""",
}

_MAP_I_SCAL_BODY = {
    k: v.replace("b[i]", "s[0]") for k, v in _MAP_I_ARR_BODY.items()
}

_MAP_F_ARR_BODY = {
    "add": "    o[i] = a[i] + b[i];",
    "sub": "    o[i] = a[i] - b[i];",
    "mul": "    o[i] = a[i] * b[i];",
}

_MAP_F_SCAL_BODY = {
    k: v.replace("b[i]", "s[0]") for k, v in _MAP_F_ARR_BODY.items()
}

_MAP_BODY = {}
_MAP_BODY.update({("int32", k): v for k, v in _MAP_I_ARR_BODY.items()})
_MAP_BODY.update({("float32", k): v for k, v in _MAP_F_ARR_BODY.items()})
_MAP_SCAL_BODY = {}
_MAP_SCAL_BODY.update({("int32", k): v for k, v in _MAP_I_SCAL_BODY.items()})
_MAP_SCAL_BODY.update({("float32", k): v for k, v in _MAP_F_SCAL_BODY.items()})

_FCOUNTS_U = """@group(0) @binding(0) var<storage,read> mk: array<i32>;
@group(0) @binding(1) var<storage,read_write> c: array<i32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;
var<workgroup> s: array<i32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {
  let i: u32 = g.x;
  let li: u32 = l.x;
  s[li] = select(0i, mk[i], i < p.x);
  workgroupBarrier();
  var stride: u32 = 128u;
  while (stride > 0u) {
    if (li < stride) { s[li] = s[li] + s[li + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (li == 0u) { c[w.x] = s[0]; }
}"""

_FSCAT_U = """@group(0) @binding(0) var<storage,read> v: array<__T__>;
@group(0) @binding(1) var<storage,read> mk: array<i32>;
@group(0) @binding(2) var<storage,read> off: array<i32>;
@group(0) @binding(3) var<storage,read_write> o: array<__T__>;
@group(0) @binding(4) var<uniform> p: vec4<u32>;
var<workgroup> s: array<i32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {
  let i: u32 = g.x;
  let li: u32 = l.x;
  let m: i32 = select(0i, mk[i], i < p.x);
  s[li] = m;
  workgroupBarrier();
  if (li == 0u) {
    var r: i32 = 0i;
    for (var k: u32 = 0u; k < 256u; k++) { let mm: i32 = s[k]; s[k] = r; r = r + mm; }
  }
  workgroupBarrier();
  if (i < p.x && m == 1i) { o[off[w.x] + s[li]] = v[i]; }
}"""

_FSCAT2_U = """@group(0) @binding(0) var<storage,read> v: array<i32>;
@group(0) @binding(1) var<storage,read> k: array<i32>;
@group(0) @binding(2) var<storage,read> mk: array<i32>;
@group(0) @binding(3) var<storage,read> off: array<i32>;
@group(0) @binding(4) var<storage,read_write> ov: array<i32>;
@group(0) @binding(5) var<storage,read_write> ok: array<i32>;
@group(0) @binding(6) var<uniform> p: vec4<u32>;
var<workgroup> s: array<i32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {
  let i: u32 = g.x;
  let li: u32 = l.x;
  let m: i32 = select(0i, mk[i], i < p.x);
  s[li] = m;
  workgroupBarrier();
  if (li == 0u) {
    var r: i32 = 0i;
    for (var k2: u32 = 0u; k2 < 256u; k2++) { let mm: i32 = s[k2]; s[k2] = r; r = r + mm; }
  }
  workgroupBarrier();
  if (i < p.x && m == 1i) { ov[off[w.x] + s[li]] = v[i]; ok[off[w.x] + s[li]] = k[i]; }
}"""

_FINDICES_U = """@group(0) @binding(0) var<storage,read> mk: array<i32>;
@group(0) @binding(1) var<storage,read> off: array<i32>;
@group(0) @binding(2) var<storage,read_write> o: array<i32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;
var<workgroup> s: array<i32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {
  let i: u32 = g.x;
  let li: u32 = l.x;
  let m: i32 = select(0i, mk[i], i < p.x);
  s[li] = m;
  workgroupBarrier();
  if (li == 0u) {
    var r: i32 = 0i;
    for (var k: u32 = 0u; k < 256u; k++) { let mm: i32 = s[k]; s[k] = r; r = r + mm; }
  }
  workgroupBarrier();
  if (i < p.x && m == 1i) { o[off[w.x] + s[li]] = i32(i); }
}"""

_REDUCE_U = """@group(0) @binding(0) var<storage,read> v: array<__T__>;
@group(0) @binding(1) var<storage,read_write> p: array<__T__>;
@group(0) @binding(2) var<uniform> q: vec4<u32>;
var<workgroup> s: array<__T__, 256>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {
  let i: u32 = g.x;
  let li: u32 = l.x;
  s[li] = select(__ZERO__, v[i], i < q.x);
  workgroupBarrier();
  var stride: u32 = 128u;
  while (stride > 0u) {
    if (li < stride) { s[li] = s[li] + s[li + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (li == 0u) { p[w.x] = s[0]; }
}"""

_GB_TILED_U = """@group(0) @binding(0) var<storage,read> keys: array<i32>;
@group(0) @binding(1) var<storage,read> hi: array<i32>;
@group(0) @binding(2) var<storage,read> lo: array<u32>;
@group(0) @binding(3) var<storage,read_write> phi: array<i32>;
@group(0) @binding(4) var<storage,read_write> plo: array<u32>;
@group(0) @binding(5) var<uniform> p: vec4<u32>;
var<workgroup> shi: array<atomic<i32>, 2048>;
var<workgroup> slo: array<atomic<u32>, 2048>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {
  let m: u32 = p.x;
  for (var j: u32 = l.x; j < m; j += 256u) {
    atomicStore(&shi[j], 0i);
    atomicStore(&slo[j], 0u);
  }
  workgroupBarrier();
  let n: u32 = arrayLength(&keys);
  let base: u32 = w.x * 2048u;
  for (var r: u32 = l.x; r < 2048u; r += 256u) {
    let i: u32 = base + r;
    if (i < n) {
      let k: u32 = u32(keys[i]);
      atomicAdd(&shi[k], hi[i]);
      atomicAdd(&slo[k], lo[i]);
    }
  }
  workgroupBarrier();
  for (var j: u32 = l.x; j < m; j += 256u) {
    phi[w.x * m + j] = atomicLoad(&shi[j]);
    plo[w.x * m + j] = atomicLoad(&slo[j]);
  }
}"""

_GB_F32_U = """@group(0) @binding(0) var<storage,read> keys: array<i32>;
@group(0) @binding(1) var<storage,read> vals: array<f32>;
@group(0) @binding(2) var<storage,read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;
var<workgroup> s: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {
  let n: u32 = arrayLength(&keys);
  let g: u32 = w.x + p.x;
  var acc: f32 = 0.0f;
  var i: u32 = l.x;
  while (i < n) {
    acc += select(0.0f, vals[i], u32(keys[i]) == g);
    i += 256u;
  }
  s[l.x] = acc;
  workgroupBarrier();
  var stride: u32 = 128u;
  while (stride > 0u) {
    if (l.x < stride) { s[l.x] = s[l.x] + s[l.x + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (l.x == 0u) { out[w.x] = s[0]; }
}"""

_SCAN_BLOCK_U = """@group(0) @binding(0) var<storage,read> x: array<u32>;
@group(0) @binding(1) var<storage,read_write> o: array<u32>;
@group(0) @binding(2) var<storage,read_write> p: array<u32>;
@group(0) @binding(3) var<uniform> q: vec4<u32>;
var<workgroup> s: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {
  let i: u32 = g.x;
  let li: u32 = l.x;
  s[li] = select(0u, x[i], i < q.x);
  workgroupBarrier();
  var off: u32 = 1u;
  while (off < 256u) {
    var v: u32 = 0u;
    if (li >= off) { v = s[li - off]; }
    workgroupBarrier();
    s[li] = s[li] + v;
    workgroupBarrier();
    off = off * 2u;
  }
  if (i < q.x) { o[i] = s[li]; }
  if (li == 0u) { p[w.x] = s[255u]; }
}"""

_SCAN_FIXUP_U = """@group(0) @binding(0) var<storage,read> x: array<u32>;
@group(0) @binding(1) var<storage,read_write> o: array<u32>;
@group(0) @binding(2) var<storage,read> poff: array<u32>;
@group(0) @binding(3) var<uniform> q: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {
  let i: u32 = g.x;
  if (i < q.x) {
    var v: u32 = o[i] + poff[w.x];
    if (q.y == 1u) { v = v - x[i]; }
    o[i] = v;
  }
}"""

_SCAN_EXCL1_U = """@group(0) @binding(0) var<storage,read> x: array<u32>;
@group(0) @binding(1) var<storage,read_write> o: array<u32>;
@group(0) @binding(2) var<uniform> q: vec4<u32>;
var<workgroup> s: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  let li: u32 = l.x;
  s[li] = select(0u, x[li], li < q.x);
  workgroupBarrier();
  var off: u32 = 1u;
  while (off < 256u) {
    var v: u32 = 0u;
    if (li >= off) { v = s[li - off]; }
    workgroupBarrier();
    s[li] = s[li] + v;
    workgroupBarrier();
    off = off * 2u;
  }
  if (li < q.x) {
    if (li == 0u) { o[li] = 0u; } else { o[li] = s[li - 1u]; }
  }
}"""

_REDUCE_SCAN_U = """@group(0) @binding(0) var<storage,read> v: array<__T__>;
@group(0) @binding(1) var<storage,read_write> p: array<__T__>;
@group(0) @binding(2) var<uniform> q: vec4<u32>;
var<workgroup> s: array<__T__, 256>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {
  let i: u32 = g.x;
  let li: u32 = l.x;
  s[li] = select(__NEUT__, v[i], i < q.x);
  workgroupBarrier();
  var stride: u32 = 128u;
  while (stride > 0u) {
    if (li < stride) { s[li] = __C__; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (li == 0u) { p[w.x] = s[0]; }
}"""

_LOOKUP_U = """@group(0) @binding(0) var<storage,read> p: array<i32>;
@group(0) @binding(1) var<storage,read> u: array<i32>;
@group(0) @binding(2) var<storage,read_write> m: array<i32>;
@group(0) @binding(3) var<uniform> q: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i: u32 = g.x;
  if (i < q.x) {
    let key: i32 = p[i];
    var lo: u32 = 0u;
    var hi: u32 = q.y;
    var found: i32 = 0i;
    while (lo < hi) {
      let mid: u32 = (lo + hi) / 2u;
      let v: i32 = u[mid];
      if (v == key) { found = 1i; break; }
      if (v < key) { lo = mid + 1u; } else { hi = mid; }
    }
    m[i] = found;
  }
}"""
# ---- Filter / Mask scan: block-partials compact (SPEC-DELTA-8) ----
# Predicate eval reuses compare/mask above (no new predicate code).
# Compact design (measured, bench_gpu_filter.json):
#   A host-assisted (baseline): full mask D2H (N i32) + host flatnonzero +
#     GPU gather. Mask crosses PCIe at full N.
#   B block-partials (new default, n > _FILTER_HOST_MAX): GPU per-workgroup
#     counts (mask resident) + tiny D2H (W=ceil(N/256) i32) + host exclusive
#     prefix over W (microseconds, bench proves negligible) + GPU scatter
#     with workgroup-local exclusive scan (single-thread 256-lane sweep in
#     shared mem: portable, no subgroups, no barrier hazards) + single D2H
#     of M kept rows. Order-preserving (stable) by construction.
# Rejected (no code, reasons):
#   per-row global atomicAdd tickets: ticket order != index order -> UNSTABLE
#     (wrong for filter contract); contention at high selectivity.
#   full-device Blelloch multi-level prefix: extra dispatches + temp buffers
#     + indirect dispatch to replace a microseconds host prefix over <=39063
#     ints (10M rows) -- measured negligible, complexity unjustified.
_FILTER_HOST_MAX = 2048  # n <= this: host-assisted wins (dispatch overhead);
# larger: block-partials (bench_gpu_filter.json crossover section).


def _filter_strategy(n):
    """Pure strategy choice (mirrors Planner.plan_filter_impl, same const)."""
    return "host" if int(n) <= _FILTER_HOST_MAX else "blocks"


_FILTER_COUNTS_WGSL = """@group(0) @binding(0) var<storage,read> mk: array<i32>;
@group(0) @binding(1) var<storage,read_write> c: array<i32>;
var<workgroup> s: array<i32, {_WG}>;
@compute @workgroup_size({_WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {{
  let i: u32 = g.x;
  let li: u32 = l.x;
  s[li] = select(0i, mk[i], i < {n}u);
  workgroupBarrier();
  var stride: u32 = {_WG2}u;
  while (stride > 0u) {{
    if (li < stride) {{ s[li] = s[li] + s[li + stride]; }}
    workgroupBarrier();
    stride = stride / 2u;
  }}
  if (li == 0u) {{ c[w.x] = s[0]; }}
}}"""

_FILTER_SCATTER_WGSL = """@group(0) @binding(0) var<storage,read> v: array<{t}>;
@group(0) @binding(1) var<storage,read> mk: array<i32>;
@group(0) @binding(2) var<storage,read> off: array<i32>;
@group(0) @binding(3) var<storage,read_write> o: array<{t}>;
var<workgroup> s: array<i32, {_WG}>;
@compute @workgroup_size({_WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {{
  let i: u32 = g.x;
  let li: u32 = l.x;
  let m: i32 = select(0i, mk[i], i < {n}u);
  s[li] = m;
  workgroupBarrier();
  if (li == 0u) {{
    var r: i32 = 0i;
    for (var k: u32 = 0u; k < {_WG}u; k++) {{ let mm: i32 = s[k]; s[k] = r; r = r + mm; }}
  }}
  workgroupBarrier();
  if (i < {n}u && m == 1i) {{ o[off[w.x] + s[li]] = v[i]; }}
}}"""

_FILTER_SCATTER2_WGSL = """@group(0) @binding(0) var<storage,read> v: array<i32>;
@group(0) @binding(1) var<storage,read> k: array<i32>;
@group(0) @binding(2) var<storage,read> mk: array<i32>;
@group(0) @binding(3) var<storage,read> off: array<i32>;
@group(0) @binding(4) var<storage,read_write> ov: array<i32>;
@group(0) @binding(5) var<storage,read_write> ok: array<i32>;
var<workgroup> s: array<i32, {_WG}>;
@compute @workgroup_size({_WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {{
  let i: u32 = g.x;
  let li: u32 = l.x;
  let m: i32 = select(0i, mk[i], i < {n}u);
  s[li] = m;
  workgroupBarrier();
  if (li == 0u) {{
    var r: i32 = 0i;
    for (var k2: u32 = 0u; k2 < {_WG}u; k2++) {{ let mm: i32 = s[k2]; s[k2] = r; r = r + mm; }}
  }}
  workgroupBarrier();
  if (i < {n}u && m == 1i) {{ ov[off[w.x] + s[li]] = v[i]; ok[off[w.x] + s[li]] = k[i]; }}
}}"""

_FILTER_INDICES_WGSL = """@group(0) @binding(0) var<storage,read> mk: array<i32>;
@group(0) @binding(1) var<storage,read> off: array<i32>;
@group(0) @binding(2) var<storage,read_write> o: array<i32>;
var<workgroup> s: array<i32, {_WG}>;
@compute @workgroup_size({_WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {{
  let i: u32 = g.x;
  let li: u32 = l.x;
  let m: i32 = select(0i, mk[i], i < {n}u);
  s[li] = m;
  workgroupBarrier();
  if (li == 0u) {{
    var r: i32 = 0i;
    for (var k: u32 = 0u; k < {_WG}u; k++) {{ let mm: i32 = s[k]; s[k] = r; r = r + mm; }}
  }}
  workgroupBarrier();
  if (i < {n}u && m == 1i) {{ o[off[w.x] + s[li]] = i32(i); }}
}}"""


def _filter_counts_gpu(dev, mask):
    """Per-workgroup kept-counts [W] (mask stays resident; only W cross)."""
    n = int(mask.size)
    w = (n + _WG - 1) // _WG
    out0 = _np.zeros(w, dtype=_np.int32)
    (rb,) = _run_u(dev, _FCOUNTS_U, [(mask.tobytes(), True),
                                     (out0.tobytes(), False)], (n, 0, 0, 0),
                   w * _WG)
    return _np.frombuffer(rb, dtype=_np.int32).copy()


def _filter_counts_resident(dev, b_mk, n, w):
    """Resident counts edge: mask buf -> counts buf (no D2H of mask)."""
    b_c = r_alloc(w * 4)
    _dispatch_u(dev, _FCOUNTS_U, [(b_mk, True), (b_c, False)],
                (int(n), 0, 0, 0), w * _WG)
    return b_c


def _filter_prefix(counts):
    """Host exclusive prefix over W block counts -> (offs int32[W], total)."""
    c = _np.ascontiguousarray(counts, dtype=_np.int64)
    offs = _np.zeros(c.size, dtype=_np.int64)
    if c.size > 1:
        offs[1:] = _np.cumsum(c[:-1])
    return offs.astype(_np.int32), int(c.sum())


def filter_compact(values, mask, dtype):
    """GPU compact: values[mask!=0], order kept, mask resident (path B).

    mask: 0/1 int32. Small n (<= _FILTER_HOST_MAX): host-assisted path A.
    Returns compacted array (single D2H of M rows; mask never crosses on B).
    int64 lane (honest, memcpy-exact): WGSL has no i64 storage type, so the
    int64 column is split host-side into lo/hi int32 halves (same as the
    groupby _gb_split pattern, C-speed reinterpret, no rounding) and each
    half compacts through the SAME i32 WGSL path (gather on A, counts +
    shared-offs scatter on B); halves rezip host-side over M kept rows
    only. Bit-exact incl. MIN/MAX (no arithmetic on values in WGSL).
    float64 values stay an explicit ValueError (honesty: WGSL portable
    path is f32-only, no f64 storage type; scaled-int ticks own groupby
    sums only per DELTA-7 sec.3 -- a filter is memcpy/order-preserving,
    not a linear sum, so unscale does not apply; DELTA-8 sec.3/5 bans
    scaled/unscaled f64 in filter -- needs backend='cpu').
    """
    mk = _np.ascontiguousarray(_np.asarray(mask).astype(_np.int32)).ravel()
    n = int(mk.size)
    if dtype == "int64":
        V = _np.ascontiguousarray(values, dtype=_np.int64).ravel()
        if V.size != n:
            raise ValueError(
                f"GPU filter values/mask size {V.size} != {n}")
        if n == 0:
            return _np.zeros(0, dtype=_np.int64)
        w = V.view(_np.int32)
        lo = _np.ascontiguousarray(w[0::2])
        hi = _np.ascontiguousarray(w[1::2])
        dev = _device()
        if _filter_strategy(n) == "host":
            eff = _np.flatnonzero(mk.astype(bool)).astype(_np.int32)
            lo_c = gather(lo, eff, "int32")
            hi_c = gather(hi, eff, "int32")
            out = _np.empty(int(eff.size), dtype=_np.int64)
            ow = out.view(_np.int32)
            ow[0::2] = lo_c
            ow[1::2] = hi_c
            return out
        counts = _filter_counts_gpu(dev, mk)
        offs, total = _filter_prefix(counts)
        if total == 0:
            return _np.zeros(0, dtype=_np.int64)
        src = _FSCAT_U.replace("__T__", "i32")
        out_lo = _np.zeros(total, dtype=_np.int32)
        out_hi = _np.zeros(total, dtype=_np.int32)
        (rb_lo,) = _run_u(dev, src, [(lo.tobytes(), True),
                                     (mk.tobytes(), True),
                                     (offs.tobytes(), True),
                                     (out_lo.tobytes(), False)],
                          (n, 0, 0, 0), n)
        (rb_hi,) = _run_u(dev, src, [(hi.tobytes(), True),
                                     (mk.tobytes(), True),
                                     (offs.tobytes(), True),
                                     (out_hi.tobytes(), False)],
                          (n, 0, 0, 0), n)
        out = _np.empty(total, dtype=_np.int64)
        ow = out.view(_np.int32)
        ow[0::2] = _np.frombuffer(rb_lo, dtype=_np.int32)
        ow[1::2] = _np.frombuffer(rb_hi, dtype=_np.int32)
        return out
    if dtype == "int32":
        V = _i32(values)
        t, np_t = "i32", _np.int32
    elif dtype == "float32":
        V = _f32(values)
        t, np_t = "f32", _np.float32
    else:
        raise ValueError(
            f"GPU filter: dtype '{dtype}' needs int32/float32/int64 "
            "(float64: WGSL portable path is f32-only, no f64 storage "
            "type; scaled-int ticks own groupby sums only per DELTA-7 "
            "sec.3 -- needs backend='cpu')")
    if V.size != n:
        raise ValueError(f"GPU filter values/mask size {V.size} != {n}")
    if n == 0:
        return _np.zeros(0, dtype=np_t)
    dev = _device()
    if _filter_strategy(n) == "host":
        eff = _np.flatnonzero(mk.astype(bool)).astype(_np.int32)
        return gather(V, eff, dtype)
    counts = _filter_counts_gpu(dev, mk)
    offs, total = _filter_prefix(counts)
    if total == 0:
        return _np.zeros(0, dtype=np_t)
    src = _FSCAT_U.replace("__T__", t)
    out0 = _np.zeros(total, dtype=np_t)
    (rb,) = _run_u(dev, src, [(V.tobytes(), True), (mk.tobytes(), True),
                              (offs.tobytes(), True),
                              (out0.tobytes(), False)], (n, 0, 0, 0), n)
    return _np.frombuffer(rb, dtype=np_t).copy()


def filter_indices(mask):
    """GPU mask -> int32 positions (compact path B; host path A if small n).

    Driver-internal stream primitive: feeds downstream gather without the
    caller materializing indices in IR (spec 02: filter never exposes them).
    """
    mk = _np.ascontiguousarray(_np.asarray(mask).astype(_np.int32)).ravel()
    n = int(mk.size)
    if n == 0:
        return _np.zeros(0, dtype=_np.int32)
    dev = _device()
    if _filter_strategy(n) == "host":
        return _np.flatnonzero(mk.astype(bool)).astype(_np.int32)
    counts = _filter_counts_gpu(dev, mk)
    offs, total = _filter_prefix(counts)
    if total == 0:
        return _np.zeros(0, dtype=_np.int32)
    out0 = _np.zeros(total, dtype=_np.int32)
    (rb,) = _run_u(dev, _FINDICES_U, [(mk.tobytes(), True),
                                      (offs.tobytes(), True),
                                      (out0.tobytes(), False)], (n, 0, 0, 0),
                   n)
    return _np.frombuffer(rb, dtype=_np.int32).copy()


def r_scatter(v_buf, mk_buf, off_buf, total, n, dtype_is_f32=False,
              out=None):
    """Resident scatter edge: compact values without mask D2H."""
    dev = _device()
    b_o = out if out is not None else r_alloc(max(1, int(total)) * 4)
    _dispatch_u(dev, _FSCAT_U.replace(
        "__T__", "f32" if dtype_is_f32 else "i32"),
        [(v_buf, True), (mk_buf, True), (off_buf, True), (b_o, False)],
        (int(n), 0, 0, 0), int(n))
    return b_o


def r_filter(values_buf, mask_buf, n, dtype_is_f32=False):
    """Resident filter: bufs in -> (compacted buf, total). W-only D2H.

    SPEC-DELTA-9 §3: N-sized mask/values never cross inside the graph;
    host sees only W counts, then M kept rows at graph exit.
    """
    dev = _device()
    w = (int(n) + _WG - 1) // _WG
    b_c = _filter_counts_resident(dev, mask_buf, int(n), w)
    counts = r_download(b_c, _np.int32)
    offs, total = _filter_prefix(counts)
    if total == 0:
        return r_alloc(4), 0
    b_off = r_upload(offs)
    return r_scatter(values_buf, mask_buf, b_off, total, int(n),
                     dtype_is_f32), total


def _chain_dispatch(dev, src, bufs, n_threads):
    """One compute dispatch over persistent device buffers (no alloc/readback).

    bufs: [(wgpu.Buffer, readonly_bool)]. Single submit; caller batches
    submits for host-dependent points (counts prefix split). Cached pipe.
    """
    import wgpu
    types = [(wgpu.BufferBindingType.read_only_storage if ro
              else wgpu.BufferBindingType.storage) for _, ro in bufs]
    pipe, bgl = _ctx().pipe(src, types)
    _submit(dev, pipe, bgl, [b for b, _ in bufs], n_threads)


def filter_groupby_resident(keys, values, mask, op="sum"):
    """Fused Filter->GroupBy with no N-sized round-trip (consumer, int32).

    Uploads keys/values/hi/lo/mask once; mask never crosses PCIe at full N
    (only W block-counts D2H + M-sized compacted readbacks); groupby
    accumulators are persistent device buffers (global-atomics path, any M).
    Returns (ukeys int64[M'], counts int64[M'], sums int64[M']) compacted.
    """
    import wgpu
    if op not in ("sum", "count", "mean"):
        raise ValueError(f"resident chain: op '{op}' needs sum/count/mean")
    k = _np.ascontiguousarray(_np.asarray(keys, dtype=_np.int32)).ravel()
    v = _np.ascontiguousarray(_np.asarray(values, dtype=_np.int32)).ravel()
    mk = _np.ascontiguousarray(_np.asarray(mask).astype(_np.int32)).ravel()
    n = int(k.size)
    if v.size != n or mk.size != n:
        raise ValueError(f"resident chain size {v.size}/{k.size}/{mk.size}")
    if n == 0:
        z = _np.zeros(0, dtype=_np.int64)
        return z, z, z
    dev = _device()
    w = (n + _WG - 1) // _WG
    b_k = r_upload(k)
    hi, lo = _gb_split(v)
    b_hi = r_upload(hi)
    b_lo = r_upload(lo.view(_np.int32))
    b_mk = r_upload(mk)
    b_cnt = _filter_counts_resident(dev, b_mk, n, w)
    counts = r_download(b_cnt, _np.int32)
    offs, total = _filter_prefix(counts)
    if total == 0:
        z = _np.zeros(0, dtype=_np.int64)
        return z, z, z
    b_off = r_upload(offs)
    b_cv = r_alloc(total * 4)
    b_ck = r_alloc(total * 4)
    # fused scatter hi+keys in one pass (two outputs, one mask read)
    _dispatch_u(dev, _FSCAT2_U,
                [(b_hi, True), (b_k, True), (b_mk, True), (b_off, True),
                 (b_cv, False), (b_ck, False)], (n, 0, 0, 0), n)
    ck = r_download(b_ck, _np.int32, total)
    # lo half needs its own compact pass (same offs, values=lo)
    b_cl = r_alloc(total * 4)
    _dispatch_u(dev, _FSCAT_U.replace("__T__", "u32"),
                [(b_lo, True), (b_mk, True), (b_off, True),
                 (b_cl, False)], (n, 0, 0, 0), n)
    cv = r_download(b_cv, _np.int32, total).astype(_np.int64)
    cl = r_download(b_cl, _np.uint32, total).astype(_np.int64)
    sums = cv * _np.int64(65536) + cl
    counts_h = _np.bincount(ck.astype(_np.int64),
                            minlength=int(ck.max()) + 1).astype(_np.int64)
    # dense sums via host bincount on compacted (M rows, selectivity-sized)
    m = int(counts_h.size)
    dense = _np.bincount(ck.astype(_np.int64), weights=sums,
                         minlength=m).astype(_np.int64)
    uk, cc, (ss,) = _gb_compact(counts_h, [dense])
    return uk, cc, ss


def reduce_partials(values, dtype, op):
    """WGSL workgroup partials (sum); CPU does exact final (int64/float64).

    Returns partials array (one per 256-block). op: sum only (count/mean
    derive on CPU: count=n, mean=sum/count).
    """
    if op != "sum":
        raise ValueError(f"GPU reduce: op '{op}' owns partials=sum only")
    n = int(_np.asarray(values).size)
    if n == 0:
        z = _np.zeros(1, dtype=_np.int32 if dtype == "int32" else _np.float32)
        return z
    dev = _device()
    nw = (n + _WG - 1) // _WG
    if dtype == "int32":
        V = _i32(values)
        t, np_t = "i32", _np.int32
        zero = "0i"
    elif dtype == "float32":
        V = _f32(values)
        t, np_t = "f32", _np.float32
        zero = "0.0f"
    else:
        raise ValueError(f"GPU reduce: dtype '{dtype}' needs int32/float32")
    src = _REDUCE_U.replace("__T__", t).replace("__ZERO__", zero)
    out = _np.zeros(nw, dtype=np_t)
    (rb,) = _run_u(dev, src, [(V.tobytes(), True), (out.tobytes(), False)],
                   (n, 0, 0, 0), n)
    return _np.frombuffer(rb, dtype=np_t).copy()


# ---- GroupBy dense hybrid (DELTA-6) ----
# counts: host bincount (exact probe-class op); sums: WGSL partials.
# int32 values split hi=(v>>16 arithmetic)/lo=(v&0xFFFF): each partial sum
# over <=256 rows fits int32, host merge hi*65536+lo in int64 is bit-exact
# (WGSL has no i64). float32: strided per-group f32 reduction + f64 merge
# (conformance tolerance, never claimed exact).
_GB_TILED_M_MAX = 2048  # shared-mem tiled partials iff M <= this
_GB_TILE = 2048         # rows per workgroup in tiled kernel (256 x 8)
_GB_DISPATCH_X = 65535  # dispatch limit (mirrors capability max_dispatch)


def _gb_chunk_rows():
    try:
        c = int(_os.environ.get("NUMFAST_GPU_GROUPBY_CHUNK", str(1 << 20)))
    except (TypeError, ValueError):
        return 1 << 20
    return max(1, c)


def _gb_split(v):
    """int32 -> (hi int32 arithmetic, lo uint32) with v == hi*65536+lo."""
    v = _np.ascontiguousarray(v, dtype=_np.int32)
    hi = _np.right_shift(v, _np.int32(16))
    lo = _np.bitwise_and(v, _np.int32(0xFFFF)).astype(_np.uint32)
    return hi, lo


def _gb_run_global_bufs(dev, src, in_bufs, accums, n_threads):
    """_gb_run_global over live device buffers (no upload, no readback)."""
    import wgpu
    allb = [b for b, _ in in_bufs] + list(accums)
    entries = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage}}
               for i in range(len(in_bufs))]
    for j in range(len(accums)):
        entries.append({"binding": len(in_bufs) + j,
                        "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": wgpu.BufferBindingType.storage}})
    types = [wgpu.BufferBindingType.read_only_storage] * len(in_bufs)
    types += [wgpu.BufferBindingType.storage] * len(accums)
    pipe, bgl = _ctx().pipe(src, types)
    _submit(dev, pipe, bgl, allb, n_threads)


def _gb_run_global(dev, src, inputs, accums, n_threads):
    """One dispatch accumulating into persistent accum buffers (no readback)."""
    in_bufs = _bufs(dev, [(b, True) for b in inputs])
    _gb_run_global_bufs(dev, src, in_bufs, accums, n_threads)


def _gb_int_tiled(dev, keys, hi, lo, m):
    """Tiled shared-mem partials (M <= 2048): returns (phi, plo) int64-ready.

    Per row-chunk: W workgroups x M groups int32 partials; host merges.
    SPEC-DELTA-9: shared arrays are static 2048 (WGSL needs const size);
    live M rides the uniform as loop bound -- one pipeline for all M.
    Dispatch is exactly W workgroups (W*256 threads): one workgroup owns
    rows [w*2048, w*2048+2048), tail guarded by i<n.
    """
    if int(m) > _GB_TILED_M_MAX:
        raise ValueError(f"tiled groupby: M={m} > cap {_GB_TILED_M_MAX}")
    c = int(keys.size)
    w = (c + _GB_TILE - 1) // _GB_TILE
    phi0 = _np.zeros(w * m, dtype=_np.int32)
    plo0 = _np.zeros(w * m, dtype=_np.uint32)
    (phi_rb, plo_rb) = _run_u(dev, _GB_TILED_U,
                              [(keys.tobytes(), True), (hi.tobytes(), True),
                               (lo.tobytes(), True), (phi0.tobytes(), False),
                               (plo0.tobytes(), False)], (int(m), 0, 0, 0),
                              w * _WG)
    phi = _np.frombuffer(phi_rb, dtype=_np.int32).astype(_np.int64)
    plo = _np.frombuffer(plo_rb, dtype=_np.uint32).astype(_np.int64)
    return phi.reshape(w, m).sum(axis=0), plo.reshape(w, m).sum(axis=0)


def _gb_int_tiled_resident(dev, b_keys, b_hi, b_lo, m, n_rows):
    """Resident tiled edge: live bufs -> (phi buf, plo buf), no N D2H."""
    w = (int(n_rows) + _GB_TILE - 1) // _GB_TILE
    b_phi = r_alloc(w * int(m) * 4)
    b_plo = r_alloc(w * int(m) * 4)
    _dispatch_u(dev, _GB_TILED_U,
                [(b_keys, True), (b_hi, True), (b_lo, True),
                 (b_phi, False), (b_plo, False)], (int(m), 0, 0, 0),
                w * _WG)
    return b_phi, b_plo, w


_GB_GLOBAL_WGSL = """@group(0) @binding(0) var<storage,read> keys: array<i32>;
@group(0) @binding(1) var<storage,read> hi: array<i32>;
@group(0) @binding(2) var<storage,read> lo: array<u32>;
@group(0) @binding(3) var<storage,read_write> ghi: array<atomic<i32>>;
@group(0) @binding(4) var<storage,read_write> glo: array<atomic<u32>>;
@compute @workgroup_size({_WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
  let n: u32 = arrayLength(&keys);
  let i: u32 = g.x;
  if (i < n) {{
    let k: u32 = u32(keys[i]);
    atomicAdd(&ghi[k], hi[i]);
    atomicAdd(&glo[k], lo[i]);
  }}
}}""".replace("{_WG}", str(_WG))


def _gb_f32_strided(dev, keys, vals, m, goff, gcount):
    """One group-block of f32 strided reduction -> f32 partials [gcount]."""
    out0 = _np.zeros(gcount, dtype=_np.float32)
    # NOTE: one workgroup per group -> threads = groups x 256 (_run divides).
    (rb,) = _run_u(dev, _GB_F32_U, [(keys.tobytes(), True),
                                    (vals.tobytes(), True),
                                    (out0.tobytes(), False)],
                   (int(goff), 0, 0, 0), gcount * _WG)
    return _np.frombuffer(rb, dtype=_np.float32).copy()


def groupby_dense_impl(keys, vcols, max_buffer_bytes=None):
    """Dense groupby sums (DELTA-6 hybrid): keys int32 >= 0.

    vcols: [(values, kind)] with kind in {int32, float32} (f64 rejected
    by caller with MapF64 guard). Returns (counts int64[M],
    sums [int64|float64 per col]). counts via host bincount (exact);
    int sums bit-exact via hi/lo split; f32 sums within tolerance.
    Op-internal row chunking (NUMFAST_GPU_GROUPBY_CHUNK) is associative:
    chunked == unchunked.
    """
    k = _np.ascontiguousarray(keys)
    if k.dtype.kind not in "iu":
        raise ValueError("GPU groupby: needs integer keys "
                         "(encode categoricals to int codes first)")
    if k.size == 0:
        zc = _np.zeros(0, dtype=_np.int64)
        return zc, [(_np.zeros(0, dtype=_np.int64)
                     if kind == "int32" else _np.zeros(0, dtype=_np.float64))
                    for _, kind in vcols]
    kmin, kmax = int(k.min()), int(k.max())
    if kmin < 0:
        raise ValueError(f"GPU groupby: dense-only, negative keys "
                         f"(kmin={kmin}) need backend='cpu'")
    m = kmax + 1
    if m > 2 ** 31 - 1:
        raise ValueError("GPU groupby: key span exceeds int32, "
                         "needs backend='cpu'")
    lim = 2 ** 31 - 1 if max_buffer_bytes is None else int(max_buffer_bytes)
    if 12 * m > lim:
        raise ValueError(f"GPU groupby: M={m} groups need {12 * m}B dense "
                         f"state > backend limit: needs backend='cpu'")
    k32 = _np.ascontiguousarray(k, dtype=_np.int32)
    counts = _np.bincount(k32.astype(_np.int64), minlength=m).astype(_np.int64)
    cols = []
    for v, kind in vcols:
        if kind == "int32":
            a = _np.ascontiguousarray(v, dtype=_np.int32)
            if a.size != k.size:
                raise ValueError("GPU groupby: values/keys size mismatch")
            cols.append(("int", a))
        elif kind == "float32":
            a = _np.ascontiguousarray(v, dtype=_np.float32)
            if a.size != k.size:
                raise ValueError("GPU groupby: values/keys size mismatch")
            cols.append(("f32", a))
        else:
            raise ValueError(f"GPU groupby: dtype '{kind}' needs int32/"
                             "float32 (float64: MapF64 guard, use CPU)")
    if not cols:
        return counts, []
    dev = _device()
    acc = [(_np.zeros(m, dtype=_np.int64)
            if kind == "int" else _np.zeros(m, dtype=_np.float64))
           for kind, _ in cols]
    chunk = _gb_chunk_rows()
    if m <= _GB_TILED_M_MAX:
        for a in range(0, k.size, chunk):
            b = min(k.size, a + chunk)
            kk = _np.ascontiguousarray(k32[a:b])
            for (kind, arr), ag in zip(
                    [(kd, c[a:b]) for kd, c in cols], acc):
                if kind == "int":
                    hi, lo = _gb_split(arr)
                    shi, slo = _gb_int_tiled(dev, kk, hi, lo, m)
                    ag += shi * _np.int64(65536) + slo
                else:
                    for goff in range(0, m, _GB_DISPATCH_X):
                        gc = min(_GB_DISPATCH_X, m - goff)
                        part = _gb_f32_strided(dev, kk, arr, m, goff, gc)
                        ag[goff:goff + gc] += part.astype(_np.float64)
    else:
        import wgpu
        # Per-int-column persistent accums: shared (bhi, blo) across columns
        # would merge their sums (wrong for multi-col M > tiled cap).
        for (kind, arr_full), ag in zip(cols, acc):
            if kind != "int":
                continue
            bhi = dev.create_buffer(
                size=m * 4, usage=wgpu.BufferUsage.STORAGE
                | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
            blo = dev.create_buffer(
                size=m * 4, usage=wgpu.BufferUsage.STORAGE
                | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
            for a in range(0, k.size, chunk):
                b = min(k.size, a + chunk)
                kk = _np.ascontiguousarray(k32[a:b])
                hi, lo = _gb_split(arr_full[a:b])
                _gb_run_global(dev, _GB_GLOBAL_WGSL,
                               [kk.tobytes(), hi.tobytes(), lo.tobytes()],
                               [bhi, blo], b - a)
            ghi = _np.frombuffer(bytes(dev.queue.read_buffer(bhi)),
                                 dtype=_np.int32).astype(_np.int64).copy()
            glo = _np.frombuffer(bytes(dev.queue.read_buffer(blo)),
                                 dtype=_np.uint32).astype(_np.int64).copy()
            ag += ghi * _np.int64(65536) + glo
    # f32 columns on the global path: strided per chunk (same as tiled path)
    if m > _GB_TILED_M_MAX:
        for a in range(0, k.size, chunk):
            b = min(k.size, a + chunk)
            kk = _np.ascontiguousarray(k32[a:b])
            for (kind, arr), ag in zip(
                    [(kd, c[a:b]) for kd, c in cols], acc):
                if kind != "f32":
                    continue
                for goff in range(0, m, _GB_DISPATCH_X):
                    gc = min(_GB_DISPATCH_X, m - goff)
                    part = _gb_f32_strided(dev, kk, arr, m, goff, gc)
                    ag[goff:goff + gc] += part.astype(_np.float64)
    return counts, acc


# ---- Bitonic sort -> stable int32 permutation (single key, pow2 N) ----
# Portable WGSL: no subgroups, no atomics, no workgroup memory, uniform-only
# control flow before the early-out return (no barriers at all). One pipeline
# (k/j/desc ride a 16B uniform), one dispatch per (k, j) stage, persistent
# buffers, single D2H of idx at the end. O(n log^2 n) compares; radix (linear)
# rejected for v1: needs atomics+prefix+scatter, more code/risk, and spec 06
# already gates GPU sort to power-of-two (non-pow2 owns CPU explicitly).
_SORT_WGSL = """struct P { k: u32, j: u32, n: u32 };
@group(0) @binding(0) var<storage,read> keys: array<u32>;
@group(0) @binding(1) var<storage,read_write> idx: array<i32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let ix: u32 = g.x;
  if (ix >= p.n) { return; }
  let l: u32 = ix ^ p.j;
  if (l > ix) {
    let up: bool = ((ix & p.k) == 0u);
    let pa: i32 = idx[ix];
    let pb: i32 = idx[l];
    let a: u32 = keys[u32(pa)];
    let b: u32 = keys[u32(pb)];
    var swap: bool = (a > b) || ((a == b) && (pa > pb));
    if (!up) { swap = (a < b) || ((a == b) && (pa < pb)); }
    if (swap) { idx[ix] = pb; idx[l] = pa; }
  }
}"""


def _is_pow2(n):
    n = int(n)
    return n > 0 and (n & (n - 1)) == 0


def _sort_u32(keys):
    """int32/finite-float32 -> order-preserving u32 (exact, host C-speed).

    int32: flip sign bit (no negation: INT32_MIN-safe). float32: standard
    sign-magnitude mapping; NaN rejected by caller (unordered, CPU owns it),
    +-inf map to extremes and sort exactly.
    """
    a = _np.ascontiguousarray(keys)
    if a.dtype == _np.dtype(_np.int32):
        return (a.view(_np.uint32) ^ _np.uint32(0x80000000))
    if a.dtype == _np.dtype(_np.float32):
        if bool(_np.isnan(a).any()):
            raise ValueError("nan")
        u = a.view(_np.uint32)
        neg = (u >> _np.uint32(31)) == _np.uint32(1)
        return _np.where(neg, ~u, u ^ _np.uint32(0x80000000)).astype(_np.uint32)
    raise ValueError(f"sort keys need int32/float32, got {a.dtype}")


def sort_perm_impl(keys, descending=False):
    """Bitonic stable permutation of one pow2 key column -> int32 positions.

    Stability = index tiebreak ((key, pos) total order, positions unique).
    Non-pow2 n and NaN are explicit errors (CPU path owns them, never silent).
    """
    a = _np.ascontiguousarray(keys)
    n = int(a.size)
    if n <= 1:
        return _np.arange(n, dtype=_np.int32)
    if not _is_pow2(n):
        raise ValueError(f"GPU sort: n={n} not power-of-two "
                         "(needs backend='cpu')")
    try:
        ku = _np.ascontiguousarray(_sort_u32(a), dtype=_np.uint32)
    except ValueError as e:
        if str(e) == "nan":
            raise ValueError("GPU sort: float32 NaN needs backend='cpu' "
                             "(unordered on GPU)") from None
        raise
    if descending:
        # Stable-desc = key-desc + input-order ties: invert keys, run the
        # single ascending network (phase flip would reverse ties too).
        ku = _np.bitwise_not(ku)
    import wgpu
    dev = _device()
    idx0 = _np.arange(n, dtype=_np.int32)
    bkeys = dev.create_buffer_with_data(
        data=ku.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    bidx = dev.create_buffer_with_data(
        data=idx0.tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        | wgpu.BufferUsage.COPY_DST)
    # SPEC-DELTA-9 §1: sort source already uniform (k/j/n ride P) -- the
    # pipeline is compiled once and shared across all N/directions.
    pipe, bgl = _ctx().pipe(
        _SORT_WGSL, (wgpu.BufferBindingType.read_only_storage,
                     wgpu.BufferBindingType.storage,
                     wgpu.BufferBindingType.uniform))
    nn = _np.uint32(n)
    k = 2
    while k <= n:
        j = k // 2
        while j >= 1:
            uni = dev.create_buffer_with_data(
                data=_np.array([k, j, nn],
                               dtype=_np.uint32).tobytes(),
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            bg = dev.create_bind_group(
                layout=bgl,
                entries=[{"binding": 0, "resource": {"buffer": bkeys,
                                                    "offset": 0,
                                                    "size": bkeys.size}},
                         {"binding": 1, "resource": {"buffer": bidx,
                                                    "offset": 0,
                                                    "size": bidx.size}},
                         {"binding": 2, "resource": {"buffer": uni,
                                                    "offset": 0,
                                                    "size": uni.size}}])
            enc = dev.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipe)
            cp.set_bind_group(0, bg, [], 0, 0)
            cp.dispatch_workgroups(max(1, (n + _WG - 1) // _WG), 1, 1)
            cp.end()
            dev.queue.submit([enc.finish()])
            j //= 2
        k *= 2
    return _np.frombuffer(bytes(dev.queue.read_buffer(bidx)),
                          dtype=_np.int32).copy()


def _gb_keep(names, bufs):
    """AND of validity sidecars (DELTA-3 mirror); None if no NA present."""
    masks = [bufs[n + "#validity"] for n in names if n + "#validity" in bufs]
    if not masks:
        return None
    m = _np.asarray(masks[0], dtype=bool).ravel()
    for extra in masks[1:]:
        m = m & _np.asarray(extra, dtype=bool).ravel()
    return m


# ---- Scaled float64 -> int32 fixed-point (SPEC-DELTA-7) ----
# WGSL/WebGPU portable path has no f64: float64 logical values with an
# explicit Schema scale/offset contract run as exact integer WGSL on int32
# physical values (same hi/lo dense hybrid as int32). f64 lives only on host
# for the final mean/sum untransform. Arbitrary f64->f32 casts stay banned:
# unscaled float64 still hits the MapF64 guard below.
_I32_MIN = -(2 ** 31)
_I32_MAX = 2 ** 31 - 1
_I64_MAX = 2 ** 63 - 1


def _is_scaled_params(params):
    try:
        sc = float(params.get("scale", 1))
        off = float(params.get("offset", 0))
    except (TypeError, ValueError):
        return False
    return not (sc == 1.0 and off == 0.0)


def _f64_to_physical(values, scale, offset, err, what):
    """Logical f64 -> int32 physical ticks (SPEC-DELTA-7, host numpy C-speed).

    Guards (explicit error = CPU path owns it, never silent):
    finite values, int32 physical range (mirrors Core Invariant #1),
    int64 total bound N*max|phys| (per-group sums must fit int64).
    """
    v = _np.ascontiguousarray(values, dtype=_np.float64)
    if v.size == 0:
        return _np.zeros(0, dtype=_np.int32)
    if not bool(_np.isfinite(v).all()):
        raise err(f"GPU driver: {what} scaled-float64 has non-finite values "
                  "(needs backend='cpu')",
                  fix="pass finite values or backend='cpu'",
                  doc="")
    phys = _np.rint((v - float(offset)) / float(scale))
    plo, phi = float(phys.min()), float(phys.max())
    if plo < _I32_MIN or phi > _I32_MAX:
        raise err(f"GPU driver: {what} scaled physical range [{plo:.0f}, "
                  f"{phi:.0f}] exceeds int32 (needs backend='cpu')",
                  fix="choose larger scale via Schema or backend='cpu'",
                  doc="")
    if int(v.size) * int(max(abs(plo), abs(phi))) >= 2 ** 63:
        raise err(f"GPU driver: {what} scaled int64 total bound exceeded "
                  "N*max|phys| >= 2**63 (needs backend='cpu')",
                  fix="use backend='cpu' for this range",
                  doc="")
    return phys.astype(_np.int32)


def _gb_unscale(sums_phys, counts, scale, offset):
    """Physical int64 sums -> (logical sums f64, logical means f64).

    sum_logical = sum_phys*scale + offset*count;
    mean_logical = sum_phys*scale/count + offset (f64 host-only).
    """
    sc, off = float(scale), float(offset)
    c = _np.ascontiguousarray(counts, dtype=_np.float64)
    s = _np.ascontiguousarray(sums_phys, dtype=_np.float64)
    sums = s * sc + off * c
    means = _np.ascontiguousarray(s * sc / _np.maximum(c, 1.0) + off)
    return sums, means


def _gb_scaled_sidecar(bufs, name):
    """Scale/offset contract sidecar for a series buffer (None if unscaled)."""
    sc = bufs.get(name + "#scaled")
    if sc is None:
        return None
    return float(sc[0]), float(sc[1])


def _gb_reject_scaled_keys(bufs, key_names, err, what):
    for kn in key_names:
        if bufs.get(kn + "#scaled") is not None:
            raise err(f"GPU driver: {what} scaled-float64 keys unsupported "
                      "(encode keys to int32 first or backend='cpu')",
                      fix="pass int32 key codes or backend='cpu'",
                      doc="")


def _gb_reject_scaled_op(bufs, names, err, what):
    bad = [n for n in names if bufs.get(n + "#scaled") is not None]
    if bad:
        raise err(f"GPU driver: {what} scaled-float64 owns groupby only "
                  f"(got {bad}, needs backend='cpu')",
                  fix="run this op with backend='cpu'",
                  doc="")


def _gb_vkind(values, err, what):
    dt = _np.asarray(values).dtype
    if dt == _np.dtype(_np.int32):
        return "int32"
    if dt == _np.dtype(_np.float32):
        return "float32"
    if dt == _np.dtype(_np.float64):
        raise err(f"GPU driver: {what} float64 needs CPU or float32 "
                  "(MapF64 guard, spec 06)",
                  fix="pass dtype='float32' or backend='cpu'",
                  doc="")
    raise err(f"GPU driver: {what} needs int32/float32 values, got {dt}",
              fix="pass int32/float32 columns or backend='cpu'",
              doc="")


def _gb_compact(counts, sums_list):
    """Dense states -> sorted (ukeys int64, counts, sums); absent groups out."""
    nz = _np.flatnonzero(counts > 0)
    uk = nz.astype(_np.int64)
    return uk, counts[nz], [s[nz] for s in sums_list]


def _gb_flat(ukeys, counts, sums, op):
    """Mirror ColumnCarry.to_dict_flat (threads=1): scalar types identical."""
    kl = ukeys.tolist()
    if op == "count":
        return {k: c for k, c in zip(kl, counts.tolist())}
    if op == "mean":
        ml = (sums.astype(_np.float64) / counts).tolist()
        return {k: float(m) for k, m in zip(kl, ml)}
    return {k: s for k, s in zip(kl, sums.tolist())}


def _gb_single(ukeys, counts, sums, means, ops):
    """Mirror ColumnCarry._dict_single_from (threads=1 branch shapes)."""
    kl = ukeys.tolist()
    sl = sums.tolist() if sums is not None else None
    cl = counts.tolist()
    ml = list(means.tolist()) if means is not None else None
    n = len(kl)
    if "sum" in ops and "count" in ops and "mean" in ops:
        return {kl[i]: {"sum": sl[i], "count": cl[i], "mean": ml[i]}
                for i in range(n)}
    if "sum" in ops and "count" in ops:
        return {kl[i]: {"sum": sl[i], "count": cl[i]} for i in range(n)}
    if "sum" in ops and "mean" in ops:
        return {kl[i]: {"sum": sl[i], "mean": ml[i]} for i in range(n)}
    if "count" in ops and "mean" in ops:
        return {kl[i]: {"count": cl[i], "mean": ml[i]} for i in range(n)}
    if "sum" in ops:
        return {kl[i]: {"sum": sl[i]} for i in range(n)}
    if "count" in ops:
        return {kl[i]: {"count": cl[i]} for i in range(n)}
    return {kl[i]: {"mean": ml[i]} for i in range(n)}


def _gb_multi(ukeys, counts, cols, sums_list, means_list, ops_map):
    """Mirror ColumnCarry._dict_multi_from (threads=1 branch shape)."""
    kl = ukeys.tolist()
    cl = counts.tolist()
    col_data = []
    for c, sums, means in zip(cols, sums_list, means_list):
        ops = ops_map[c]
        sl = sums.tolist() if "sum" in ops else None
        ml = list(means.tolist()) if means is not None else None
        col_data.append((c, sl, ml, ops))
    res = {}
    for i in range(len(kl)):
        cell = {}
        for c, sl, ml, ops in col_data:
            sub = {}
            if "sum" in ops:
                sub["sum"] = sl[i]
            if "count" in ops:
                sub["count"] = cl[i]
            if "mean" in ops:
                sub["mean"] = ml[i]
            cell[c] = sub
        res[kl[i]] = cell
    return res


def _gb_means(sums, counts, want):
    if not want:
        return None
    return _np.ascontiguousarray(sums.astype(_np.float64) / counts)


def gpu_capability_impl():
    return {
        "ops": ["series", "pack_keys", "compare", "mask", "filter",
                "gather", "reduce", "groupby", "groupby_multi", "sort",
                "slice", "shift", "map", "cumsum", "rng_fill_i32"],
        "max_dispatch": {"x": 65535, "y": 65535, "z": 65535},
        "max_buffer_bytes": 2 ** 31 - 1,
        "chunkable_hints": {"series": True, "pack_keys": True,
                            "compare": True, "mask": True, "filter": True,
                            "gather": True, "reduce": True,
                            "groupby": False, "groupby_multi": False,
                            "sort": False, "slice": False, "shift": False,
                            "map": True, "cumsum": False,
                            "rng_fill_i32": True},
        "note": ("wgpu-py RTX 2060 Vulkan: elementwise pure-GPU; "
                 "filter=GPU block-partials compact (SPEC-DELTA-8: "
                 "per-workgroup counts on GPU, mask resident, tiny W "
                 "prefix on host, workgroup-local scan scatter; "
                 "host-assisted gather only for n<=2048); "
                 "reduce=GPU-partials+CPU-int64/float64-final hybrid; "
                 "groupby/groupby_multi=dense-only hybrid (WGSL partial "
                 "histograms + host int64/f64 merge, DELTA-6; "
                 "non-negative int32 keys, int32/float32 values, "
                 "scaled-float64 (SPEC-DELTA-7 Schema scale/offset -> int32 "
                 "physical, exact integer WGSL, f64 unscale on host); "
                 "sort=bitonic stable permutation, single int32/finite-f32 "
                 "key, power-of-two N (spec 06/07), NaN/non-pow2/multi-key/ "
                 "int64 CPU-only explicit; slice=resident LIMIT/OFFSET take; "
                 "shift=positional right-shift (head invalid, memcpy-exact, "
                 "int32/float32, validity carry on host); "
                 "cumsum=inclusive prefix (int32 wraparound scan, "
                 "0-fill+carry on host; f32/f64 CPU-only explicit); "
                 "map=elementwise (i32 add/sub/mul wrap + floor_div/mod "
                 "floor-sign-corrected + f32 add/sub/mul IEEE, validity AND "
                 "on host; i32 div/pow + f32 div/pow/floor_div/mod + f64 + "
                 "mixed-dtype CPU-only explicit); "
                 "unscaled-f64/carry/sparse-keys CPU-only explicit); "
                 "encode_pattern CPU-only (explicit error); "
                 "lookup NOT on GPU: IR lookup (unique-keys contract, "
                 "positions+hit, dupe=error) has no measured GPU cost and no "
                 "positions kernel -- the existing lookup_mask/LookupTable "
                 "standalone is mask-only semi-lookup with dupe-collapse "
                 "set semantics (different contract), reusable as declared, "
                 "never as this node"),
    }


def _as_col(values, dtype, err, what):
    import numpy as _np2
    if dtype == "float64":
        raise err(f"GPU driver: {what} float64 needs CPU or float32 "
                  "(MapF64 guard, spec 06)",
                  fix="pass dtype='float32' or backend='cpu'",
                  doc="")
    try:
        return (_f32(values) if dtype == "float32" else _i32(values))
    except (TypeError, ValueError):
        raise err(f"GPU driver: {what} needs rank-1 numeric",
                  fix="pass a flat int32/float32 column",
                  doc="") from None


# ---- Standalone Scan / Reduction primitives (portable WGSL, no subgroups) ----
# Design (alternatives measured, not guessed):
#   block-scan (Hillis-Steele, 256-lane shared mem, uniform barriers only) ->
#   per-block totals [W] -> exclusive prefix over W -> fixup add (+optional
#   subtract of input for exclusive). Default hybrid scan_inclusive/
#   scan_exclusive: W-prefix on host (microseconds for W<=39063 at 10M rows;
#   one W-sized D2H + one W-sized H2D). Full-device scan_exclusive_gpu chains
#   the W-prefix on GPU (zero round trips, extra dispatches) to answer the
#   resident-graph question (replace host _filter_prefix or not).
# Rejected without code: per-row global atomicAdd tickets (ticket order !=
#   index order -> UNSTABLE for scan; contention); cross-workgroup sync in one
#   pass (no such primitive in portable WGSL); subgroup ops (not portable,
#   gated optional per R12); full Blelloch multi-level for the W-prefix (extra
#   dispatches/buffers to replace microseconds of host prefix, same argument
#   as DELTA-8).
# Overflow policy (EXPLICIT, tested): ALL int32/uint32 scan/reduce arithmetic
#   is wraparound mod 2**32 (u32 lanes; int32 = bit-identical reinterpret).
#   Host prefix/sum merges accumulate mod 2**32. scan_ref/reduce_ref compute
#   exact in uint64 then wrap. MIN/MAX/COUNT exact (no overflow possible);
#   min/max on empty -> explicit ValueError (numpy parity). No IR/Planner
#   wiring here: standalone primitive, GroupBy/Sort/Filter untouched.
_SCAN_BLOCK_WGSL = """@group(0) @binding(0) var<storage,read> x: array<u32>;
@group(0) @binding(1) var<storage,read_write> o: array<u32>;
@group(0) @binding(2) var<storage,read_write> p: array<u32>;
var<workgroup> s: array<u32, {WG}>;
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {{
  let i: u32 = g.x;
  let li: u32 = l.x;
  s[li] = select(0u, x[i], i < {N}u);
  workgroupBarrier();
  var off: u32 = 1u;
  while (off < {WG}u) {{
    var v: u32 = 0u;
    if (li >= off) {{ v = s[li - off]; }}
    workgroupBarrier();
    s[li] = s[li] + v;
    workgroupBarrier();
    off = off * 2u;
  }}
  if (i < {N}u) {{ o[i] = s[li]; }}
  if (li == 0u) {{ p[w.x] = s[{WGM1}u]; }}
}}"""

_SCAN_FIXUP_WGSL = """@group(0) @binding(0) var<storage,read> x: array<u32>;
@group(0) @binding(1) var<storage,read_write> o: array<u32>;
@group(0) @binding(2) var<storage,read> poff: array<u32>;
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {{
  let i: u32 = g.x;
  if (i < {N}u) {{
    var v: u32 = o[i] + poff[w.x];
    {EXCL}
    o[i] = v;
  }}
}}"""

_SCAN_EXCL1_WGSL = """@group(0) @binding(0) var<storage,read> x: array<u32>;
@group(0) @binding(1) var<storage,read_write> o: array<u32>;
var<workgroup> s: array<u32, {WG}>;
@compute @workgroup_size({WG})
fn main(@builtin(local_invocation_id) l: vec3<u32>) {{
  let li: u32 = l.x;
  s[li] = select(0u, x[li], li < {N}u);
  workgroupBarrier();
  var off: u32 = 1u;
  while (off < {WG}u) {{
    var v: u32 = 0u;
    if (li >= off) {{ v = s[li - off]; }}
    workgroupBarrier();
    s[li] = s[li] + v;
    workgroupBarrier();
    off = off * 2u;
  }}
  if (li < {N}u) {{
    if (li == 0u) {{ o[li] = 0u; }} else {{ o[li] = s[li - 1u]; }}
  }}
}}"""

_REDUCE_WGSL = """@group(0) @binding(0) var<storage,read> v: array<{T}>;
@group(0) @binding(1) var<storage,read_write> p: array<{T}>;
var<workgroup> s: array<{T}, {WG}>;
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {{
  let i: u32 = g.x;
  let li: u32 = l.x;
  s[li] = select({NEUT}, v[i], i < {N}u);
  workgroupBarrier();
  var stride: u32 = {WG2}u;
  while (stride > 0u) {{
    if (li < stride) {{ s[li] = {C}; }}
    workgroupBarrier();
    stride = stride / 2u;
  }}
  if (li == 0u) {{ p[w.x] = s[0]; }}
}}"""

_SCAN_MAX_RESIDENT_N = 16 * 1024 * 1024  # gpu variant needs W<=65536 (2 levels)


def _scan_lane_in(values, dtype):
    """int32/uint32 -> (u32 view, numpy out-dtype). No silent float cast."""
    if dtype == "int32":
        a = _np.ascontiguousarray(values, dtype=_np.int32).ravel()
        return a.view(_np.uint32), _np.int32
    if dtype == "uint32":
        a = _np.ascontiguousarray(values, dtype=_np.uint32).ravel()
        return a, _np.uint32
    raise ValueError(f"GPU scan: dtype '{dtype}' needs int32/uint32 "
                     "(exact integer lanes only)")


def _scan_block_gpu(dev, xu):
    """One dispatch: block-inclusive scan -> (incl[N] u32, totals[W] u32)."""
    n = int(xu.size)
    w = (n + _WG - 1) // _WG
    o0 = _np.zeros(n, dtype=_np.uint32)
    p0 = _np.zeros(w, dtype=_np.uint32)
    (o_rb, p_rb) = _run_u(dev, _SCAN_BLOCK_U, [(xu.tobytes(), True),
                                              (o0.tobytes(), False),
                                              (p0.tobytes(), False)],
                          (n, 0, 0, 0), w * _WG)
    return (_np.frombuffer(o_rb, dtype=_np.uint32).copy(),
            _np.frombuffer(p_rb, dtype=_np.uint32).copy())


def _prefix_excl_mod32(parts):
    """Host exclusive prefix over W block totals, wraparound mod 2**32."""
    c = _np.ascontiguousarray(parts, dtype=_np.uint64)
    o = _np.zeros(c.size, dtype=_np.uint64)
    if c.size > 1:
        o[1:] = _np.cumsum(c[:-1])
    return (o % _np.uint64(2 ** 32)).astype(_np.uint32)


def _scan_fixup_gpu(dev, xu, incl, poff, exclusive):
    """One dispatch: incl[i] + poff[block] (- x[i] if exclusive), in place."""
    n = int(xu.size)
    (rb,) = _run_u(dev, _SCAN_FIXUP_U,
                   [(xu.tobytes(), True), (incl.tobytes(), False),
                    (poff.tobytes(), True)], (n, 1 if exclusive else 0, 0, 0),
                   n)
    return _np.frombuffer(rb, dtype=_np.uint32).copy()


def scan_inclusive(values, dtype="int32"):
    """Portable WGSL inclusive prefix scan (hybrid: GPU blocks + host fixup).

    Exact vs wraparound reference (see policy above). n==0 -> empty, no
    dispatch. Single block (n<=256) skips the fixup dispatch.
    """
    xu, out_t = _scan_lane_in(values, dtype)
    n = int(xu.size)
    if n == 0:
        return _np.zeros(0, dtype=out_t)
    dev = _device()
    incl, parts = _scan_block_gpu(dev, xu)
    if parts.size == 1:
        return incl.view(out_t)
    return _scan_fixup_gpu(dev, xu, incl, _prefix_excl_mod32(parts),
                           False).view(out_t)


def scan_exclusive(values, dtype="int32"):
    """Portable WGSL exclusive prefix scan (hybrid). out[0]=0 identity."""
    xu, out_t = _scan_lane_in(values, dtype)
    n = int(xu.size)
    if n == 0:
        return _np.zeros(0, dtype=out_t)
    dev = _device()
    incl, parts = _scan_block_gpu(dev, xu)
    poff = (_np.zeros(1, dtype=_np.uint32) if parts.size == 1
            else _prefix_excl_mod32(parts))
    return _scan_fixup_gpu(dev, xu, incl, poff, True).view(out_t)


def scan_exclusive_gpu(values, dtype="int32"):
    """Full-device exclusive scan: 5 dispatches, zero round trips to host.

    Same result as scan_exclusive (bit-exact). N<=16M guard (2-level prefix
    covers W<=65536); larger -> explicit error, use hybrid scan_exclusive.
    Measurement variant for the resident-graph question, not the default.
    """
    import wgpu
    xu, out_t = _scan_lane_in(values, dtype)
    n = int(xu.size)
    if n == 0:
        return _np.zeros(0, dtype=out_t)
    if n > _SCAN_MAX_RESIDENT_N:
        raise ValueError(f"GPU scan resident: n={n} > 16M needs hybrid "
                         "scan_exclusive (2-level prefix capped at W=65536)")
    dev = _device()
    w = (n + _WG - 1) // _WG
    w2 = (w + _WG - 1) // _WG
    b_x = r_upload(xu)
    b_o = r_alloc(n * 4)
    b_p1 = r_alloc(w * 4)
    b_ip1 = r_alloc(w * 4)
    b_p2 = r_alloc(w2 * 4)
    b_off2 = r_alloc(w2 * 4)
    _dispatch_u(dev, _SCAN_BLOCK_U,
                [(b_x, True), (b_o, False), (b_p1, False)], (n, 0, 0, 0),
                w * _WG)
    _dispatch_u(dev, _SCAN_BLOCK_U,
                [(b_p1, True), (b_ip1, False), (b_p2, False)], (w, 0, 0, 0),
                w2 * _WG)
    _dispatch_u(dev, _SCAN_EXCL1_U, [(b_p2, True), (b_off2, False)],
                (w2, 0, 0, 0), _WG)
    _dispatch_u(dev, _SCAN_FIXUP_U,
                [(b_p1, True), (b_ip1, False), (b_off2, True)], (w, 1, 0, 0),
                w)
    _dispatch_u(dev, _SCAN_FIXUP_U,
                [(b_x, True), (b_o, False), (b_ip1, True)], (n, 1, 0, 0), n)
    return r_download(b_o, _np.uint32, n).view(out_t)


def reduce_full(values, dtype="int32", op="sum"):
    """Standalone WGSL reduction: sum/min/max/count (exact integer lanes).

    sum wraps mod 2**32 (policy above); min/max/count exact. count needs no
    dispatch. Empty: sum->0, count->0, min/max->ValueError. float dtypes ->
    explicit error (not exact, CPU owns them).
    """
    if op == "count":
        return int(_np.asarray(values).size)
    xu, out_t = _scan_lane_in(values, dtype)
    n = int(xu.size)
    if n == 0:
        if op == "sum":
            return 0
        raise ValueError(f"GPU reduce: op '{op}' of empty input undefined "
                         "(numpy parity: explicit error, no neutral leak)")
    dev = _device()
    nw = (n + _WG - 1) // _WG
    if op == "sum":
        src = _REDUCE_SCAN_U.replace("__T__", "u32").replace(
            "__NEUT__", "0u").replace("__C__", "s[li] + s[li + stride]")
        out0 = _np.zeros(nw, dtype=_np.uint32)
        (rb,) = _run_u(dev, src, [(xu.tobytes(), True),
                                  (out0.tobytes(), False)], (n, 0, 0, 0), n)
        tot = int(_np.frombuffer(rb, dtype=_np.uint32).astype(
            _np.uint64).sum() % _np.uint64(2 ** 32))
        return tot if out_t == _np.uint32 else int(
            _np.uint32(tot).view(_np.int32))
    if op in ("min", "max"):
        signed = out_t == _np.int32
        T = "i32" if signed else "u32"
        if signed:
            neut = ("2147483647i" if op == "min" else "-2147483647i - 1i")
            V = xu.view(_np.int32)
            np_t = _np.int32
        else:
            neut = "4294967295u" if op == "min" else "0u"
            V = xu
            np_t = _np.uint32
        fn = "min" if op == "min" else "max"
        src = _REDUCE_SCAN_U.replace("__T__", T).replace(
            "__NEUT__", neut).replace("__C__", f"{fn}(s[li], s[li + stride])")
        out0 = _np.zeros(nw, dtype=np_t)
        (rb,) = _run_u(dev, src, [(V.tobytes(), True),
                                  (out0.tobytes(), False)], (n, 0, 0, 0), n)
        part = _np.frombuffer(rb, dtype=np_t)
        return int(part.min() if op == "min" else part.max())
    raise ValueError(f"GPU reduce: op '{op}' needs sum/min/max/count "
                     "(mean/var/std CPU-only, not exact in one pass)")


def scan_ref(values, dtype="int32", inclusive=True):
    """CPU reference: exact uint64 cumsum wrapped mod 2**32 (policy oracle)."""
    xu, out_t = _scan_lane_in(values, dtype)
    c = _np.cumsum(xu.astype(_np.uint64)) % _np.uint64(2 ** 32)
    if not inclusive:
        c = (c - xu.astype(_np.uint64)) % _np.uint64(2 ** 32)
    return c.astype(_np.uint32).view(out_t)


def reduce_ref(values, dtype="int32", op="sum"):
    """CPU reference for reduce_full (same wrap policy; empty min/max err)."""
    xu, out_t = _scan_lane_in(values, dtype)
    n = int(xu.size)
    if op == "count":
        return n
    if n == 0:
        if op == "sum":
            return 0
        raise ValueError("ref: min/max of empty undefined")
    if op == "sum":
        tot = int(xu.astype(_np.uint64).sum() % _np.uint64(2 ** 32))
        return tot if out_t == _np.uint32 else int(
            _np.uint32(tot).view(_np.int32))
    v = xu.view(_np.int32) if out_t == _np.int32 else xu
    return int(v.min() if op == "min" else v.max())


# ---- Standalone Lookup / semi-join (resident int32, portable WGSL) ----
# Semantics (v1, semi-join, NOT full SQL JOIN):
#   build B[M] int32 (dupes allowed) -> resident sorted-unique dict U[K];
#   probe P[N] int32 -> mask[N] 0/1, order-preserving (probe order).
#   match = existence (probe value in set(B)). Build dupes collapse (set
#   semantics, no multiplicity expansion: B=[5,5,7],P=[5] -> mask=[1]).
#   Probe dupes independent (P=[5,5] -> [1,1]). M=0 -> all zeros; N=0 ->
#   empty. Output feeds Filter/Gather (mask -> filter_compact /
#   filter_indices -> gather). No IR/Planner wiring here: standalone
#   primitive, GroupBy/Sort/Filter/Scan untouched.
# Design (minimal: host build + GPU probe):
#   build on host (numpy unique+sort, O(M log M) once, amortized over
#   probes); probe binary-search on GPU (per-thread iterative, uniform
#   loop, no subgroups, no shared memory: U random-access, shared tile
#   buys nothing, avoids bank conflicts by construction; coalesced probe
#   reads). Workgroup 256, guard i<N. No hash table (no load-factor /
#   tombstone / divergence tuning; measured first, hash only on profile
#   need -- not now).
_LOOKUP_WGSL = """@group(0) @binding(0) var<storage,read> p: array<i32>;
@group(0) @binding(1) var<storage,read> u: array<i32>;
@group(0) @binding(2) var<storage,read_write> m: array<i32>;
@compute @workgroup_size({_WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
  let i: u32 = g.x;
  if (i < {N}u) {{
    let key: i32 = p[i];
    var lo: u32 = 0u;
    var hi: u32 = {K}u;
    var found: i32 = 0i;
    while (lo < hi) {{
      let mid: u32 = (lo + hi) / 2u;
      let v: i32 = u[mid];
      if (v == key) {{ found = 1i; break; }}
      if (v < key) {{ lo = mid + 1u; }} else {{ hi = mid; }}
    }}
    m[i] = found;
  }}
}}"""


def lookup_build(keys):
    """Build-side resident dict: sorted-unique int32 (host, exact).

    Dupes collapse (set semantics); empty -> empty int32. Caller keeps the
    array resident (upload once, probe many).
    """
    b = _np.ascontiguousarray(_np.asarray(keys).ravel(), dtype=_np.int32)
    if b.size == 0:
        return _np.zeros(0, dtype=_np.int32)
    return _np.ascontiguousarray(_np.unique(b))


def lookup_ref(build, probe):
    """CPU reference: mask[i] = probe[i] in set(build) (vectorized oracle)."""
    p = _np.ascontiguousarray(_np.asarray(probe).ravel(), dtype=_np.int32)
    n = int(p.size)
    if n == 0:
        return _np.zeros(0, dtype=_np.int32)
    u = lookup_build(build)
    if u.size == 0:
        return _np.zeros(n, dtype=_np.int32)
    return _np.ascontiguousarray(_np.isin(p, u).astype(_np.int32))


def _lookup_src(n, k):
    return _LOOKUP_WGSL.replace("{_WG}", str(_WG)).replace(
        "{N}", str(int(n))).replace("{K}", str(int(k)))


def lookup_mask(build, probe):
    """GPU probe: mask[N] 0/1 via binary search over resident U (one D2H)."""
    p = _np.ascontiguousarray(_np.asarray(probe).ravel(), dtype=_np.int32)
    n = int(p.size)
    if n == 0:
        return _np.zeros(0, dtype=_np.int32)
    u = lookup_build(build)
    k = int(u.size)
    dev = _device()
    ubuf = u if k else _np.zeros(1, dtype=_np.int32)  # device needs size>=1
    out0 = _np.zeros(n, dtype=_np.int32)
    (rb,) = _run_u(dev, _LOOKUP_U, [(p.tobytes(), True),
                                    (ubuf.tobytes(), True),
                                    (out0.tobytes(), False)], (n, k, 0, 0),
                   n)
    return _np.frombuffer(rb, dtype=_np.int32).copy()


class LookupTable:
    """SPEC-DELTA-9 §5: build-once, probe-many resident dict.

    Upload once (U[K] sorted-unique stays on device); each probe uploads
    only P[N] and downloads only mask[N]. Intended use-case = repeated
    probes over one build (star-schema dimension), not cold one-shot.
    """

    def __init__(self, build):
        u = lookup_build(build)
        self.u = u
        self.k = int(u.size)
        ubuf = u if self.k else _np.zeros(1, dtype=_np.int32)
        self.b_u = r_upload(ubuf)

    def probe_mask(self, probe):
        """Probe -> mask[N] 0/1 (build never re-uploaded, no N round-trip)."""
        p = _np.ascontiguousarray(_np.asarray(probe).ravel(),
                                  dtype=_np.int32)
        n = int(p.size)
        if n == 0:
            return _np.zeros(0, dtype=_np.int32)
        dev = _device()
        b_p = r_upload(p)
        b_m = r_alloc(n * 4)
        _dispatch_u(dev, _LOOKUP_U, [(b_p, True), (self.b_u, True),
                                     (b_m, False)], (n, self.k, 0, 0), n)
        return r_download(b_m, _np.int32, n)

    def probe_filter(self, probe_keys, payload):
        """Probe + filter payload in one resident chain (M kept rows D2H)."""
        p = _np.ascontiguousarray(_np.asarray(probe_keys).ravel(),
                                  dtype=_np.int32)
        v = _np.ascontiguousarray(_np.asarray(payload).ravel(),
                                  dtype=_np.int32)
        n = int(p.size)
        if v.size != n:
            raise ValueError(f"GPU lookup chain payload/probe size "
                             f"{v.size} != {n}")
        if n == 0 or self.k == 0:
            return _np.zeros(0, dtype=_np.int32)
        dev = _device()
        b_p = r_upload(p)
        b_m = r_alloc(n * 4)
        _dispatch_u(dev, _LOOKUP_U, [(b_p, True), (self.b_u, True),
                                     (b_m, False)], (n, self.k, 0, 0), n)
        w = (n + _WG - 1) // _WG
        b_c = _filter_counts_resident(dev, b_m, n, w)
        counts = r_download(b_c, _np.int32)
        offs, total = _filter_prefix(counts)
        if total == 0:
            return _np.zeros(0, dtype=_np.int32)
        b_v = r_upload(v)
        b_off = r_upload(offs)
        b_o = r_alloc(total * 4)
        _dispatch_u(dev, _FSCAT_U.replace("__T__", "i32"),
                    [(b_v, True), (b_m, True), (b_off, True), (b_o, False)],
                    (n, 0, 0, 0), n)
        return r_download(b_o, _np.int32, total)


def lookup_build_resident(build):
    """Build-side resident dict handle (upload once, probe many)."""
    return LookupTable(build)


def lookup_filter_resident(build, probe_keys, payload):
    """Resident chain Lookup->Filter: payload[probe in build], order kept.

    Mask never crosses PCIe at full N (device buffers throughout; only W
    block-counts D2H + M kept rows D2H). Standalone (no IR), int32 only.
    """
    import wgpu
    p = _np.ascontiguousarray(_np.asarray(probe_keys).ravel(),
                              dtype=_np.int32)
    v = _np.ascontiguousarray(_np.asarray(payload).ravel(), dtype=_np.int32)
    n = int(p.size)
    if v.size != n:
        raise ValueError(f"GPU lookup chain payload/probe size "
                         f"{v.size} != {n}")
    if n == 0:
        return _np.zeros(0, dtype=_np.int32)
    u = lookup_build(build)
    k = int(u.size)
    if k == 0:
        return _np.zeros(0, dtype=_np.int32)
    dev = _device()
    b_p = r_upload(p)
    b_u = r_upload(u)
    b_m = r_alloc(n * 4)
    _dispatch_u(dev, _LOOKUP_U, [(b_p, True), (b_u, True), (b_m, False)],
                (n, k, 0, 0), n)
    w = (n + _WG - 1) // _WG
    b_c = _filter_counts_resident(dev, b_m, n, w)
    counts = r_download(b_c, _np.int32)
    offs, total = _filter_prefix(counts)
    if total == 0:
        return _np.zeros(0, dtype=_np.int32)
    b_v = r_upload(v)
    b_off = r_upload(offs)
    b_o = r_alloc(total * 4)
    _dispatch_u(dev, _FSCAT_U.replace("__T__", "i32"),
                [(b_v, True), (b_m, True), (b_off, True), (b_o, False)],
                (n, 0, 0, 0), n)
    return r_download(b_o, _np.int32, total)


# ---- SPEC-DELTA-9 §4: Pack->GroupBy resident (no pack round-trip) ----
def pack_groupby_resident(k1, k2, values, op="sum"):
    """Input -> GPU pack -> resident packed keys -> GPU groupby -> M result.

    Removes the pack round-trip (old path: pack D2H of N i32 + re-upload
    keys+hi+lo; new: packed keys never leave the device, host sees only
    W-tile partials [W*M'] + M compacted groups). Q2 without half its PCIe.
    int32 values/keys, non-negative packed keys (dense path).
    Returns (ukeys int64[M'], counts int64[M'], sums int64[M']).
    """
    a1 = _np.ascontiguousarray(_np.asarray(k1, dtype=_np.int32)).ravel()
    a2 = _np.ascontiguousarray(_np.asarray(k2, dtype=_np.int32)).ravel()
    v = _np.ascontiguousarray(_np.asarray(values, dtype=_np.int32)).ravel()
    n = int(a1.size)
    if a2.size != n or v.size != n:
        raise ValueError(f"pack-groupby size {a2.size}/{a1.size}/{v.size}")
    if n == 0:
        z = _np.zeros(0, dtype=_np.int64)
        return z, z, z
    m2 = int(a2.max(initial=0)) + 1
    dev = _device()
    b_k1, b_k2 = r_upload(a1), r_upload(a2)
    b_pk = r_pack(b_k1, b_k2, m2, n)  # resident packed keys, no D2H
    hi, lo = _gb_split(v)
    m = int(a1.max(initial=0)) * m2 + m2  # packed key span (dense bound)
    if m <= _GB_TILED_M_MAX:
        b_hi, b_lo = r_upload(hi), r_upload(lo.view(_np.int32))
        b_phi, b_plo, w = _gb_int_tiled_resident(dev, b_pk, b_hi, b_lo, m,
                                                 n)
        phi = r_download(b_phi, _np.int32).reshape(w, m).astype(
            _np.int64).sum(axis=0)
        plo = r_download(b_plo, _np.uint32).reshape(w, m).astype(
            _np.int64).sum(axis=0)
        acc = phi * _np.int64(65536) + plo
        counts = _np.zeros(m, dtype=_np.int64)
        # counts still need one pass over packed keys: reuse M-sized
        # download of packed keys? No -- counts via partial ones: upload
        # ones is wasteful; single compacted-keys D2H is M'-bounded only
        # after compaction, which needs counts. Cheap exact alternative:
        # bincount over host pack (C-speed, no GPU claim for counts).
        pk = (a1.astype(_np.int64) * m2 + a2.astype(_np.int64))
        counts = _np.bincount(pk, minlength=m).astype(_np.int64)
        uk, cc, (ss,) = _gb_compact(counts, [acc])
        if op == "count":
            return uk, cc, cc
        if op == "mean":
            return uk, cc, (ss.astype(_np.float64) /
                            _np.maximum(cc, 1))
        return uk, cc, ss
    # large-M: global atomics on resident packed keys (persistent accums)
    import wgpu
    bhi = dev.create_buffer(
        size=m * 4, usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
    blo = dev.create_buffer(
        size=m * 4, usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
    chunk = _gb_chunk_rows()
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        # slice resident packed keys per chunk without D2H: resident slice
        sub = r_alloc((b - a) * 4)
        _dispatch_u(dev, _SLICE_U.replace("__T__", "i32"),
                    [(b_pk, True), (sub, False)], (b - a, a, 0, 0), b - a)
        hh, ll = _gb_split(v[a:b])
        _gb_run_global_bufs(dev, _GB_GLOBAL_WGSL,
                            [(sub, True), (r_upload(hh), True),
                             (r_upload(ll.view(_np.int32)), True)],
                            [bhi, blo], b - a)
    ghi = r_download(bhi, _np.int32).astype(_np.int64)
    glo = r_download(blo, _np.uint32).astype(_np.int64)
    acc = ghi * _np.int64(65536) + glo
    pk = a1.astype(_np.int64) * m2 + a2.astype(_np.int64)
    counts = _np.bincount(pk, minlength=m).astype(_np.int64)
    uk, cc, (ss,) = _gb_compact(counts, [acc])
    return uk, cc, (cc if op == "count" else ss)


# ---- SPEC-DELTA-9 §6: fused GPU graph as normal path ----
def graph_compare_mask_filter_groupby(a, threshold, values, keys=None,
                                      op="sum"):
    """Compare->Mask->Filter->GroupBy without N-sized host round-trips.

    Normal graph path (§6): predicate+mask+compact stay resident; host
    sees only W counts + M compacted rows + M groups. Returns same
    (ukeys, counts, sums) triple as filter_groupby_resident.
    keys=None -> group filtered values by themselves (self-grouped Q1);
    keys given -> group paired keys (Q2-style after pack).
    """
    A = _np.ascontiguousarray(_np.asarray(a, dtype=_np.int32)).ravel()
    V = _np.ascontiguousarray(_np.asarray(values, dtype=_np.int32)).ravel()
    n = int(A.size)
    if V.size != n:
        raise ValueError(f"graph values/probe size {V.size} != {n}")
    K = (A if keys is None else _np.ascontiguousarray(
        _np.asarray(keys, dtype=_np.int32)).ravel())
    if K.size != n:
        raise ValueError(f"graph keys/probe size {K.size} != {n}")
    if n == 0:
        z = _np.zeros(0, dtype=_np.int64)
        return z, z, z
    dev = _device()
    b_a, b_v, b_k = r_upload(A), r_upload(V), r_upload(K)
    sb = _np.int32(int(threshold)).view(_np.uint32)
    b_mk = r_alloc(n * 4)
    _dispatch_u(dev, _CMP_I_SCAL_U.replace("__EXPR__", _CMP_WGSL[">"]),
                [(b_a, True), (b_mk, False)], (n, int(sb), 0, 0), n)
    b_cv, total = r_filter(b_v, b_mk, n)
    b_ck, total2 = r_filter(b_k, b_mk, n)
    assert total == total2
    if total == 0:
        z = _np.zeros(0, dtype=_np.int64)
        return z, z, z
    cv = r_download(b_cv, _np.int32, total).astype(_np.int64)
    ck = r_download(b_ck, _np.int32, total)
    m = int(ck.max()) + 1
    counts_h = _np.bincount(ck.astype(_np.int64),
                            minlength=m).astype(_np.int64)
    dense = _np.bincount(ck.astype(_np.int64), weights=cv,
                         minlength=m).astype(_np.int64)
    uk, cc, (ss,) = _gb_compact(counts_h, [dense])
    if op == "count":
        return uk, cc, cc
    if op == "mean":
        return uk, cc, ss.astype(_np.float64) / _np.maximum(cc, 1)
    return uk, cc, ss


def graph_sort_slice_gather(values, limit, offset=0, descending=False):
    """Sort->Slice->Gather resident: order-by+limit without index D2H.

    Perm stays resident; slice+ gather chain on device; host sees only
    the final `limit` rows. Non-pow2/NaN contract = sort_perm_impl errors.
    """
    V = _np.ascontiguousarray(values, dtype=_np.int32)
    n = int(V.size)
    if n == 0 or int(limit) <= 0:
        return _np.zeros(0, dtype=_np.int32)
    perm = sort_perm_impl(V, descending)  # explicit errors preserved
    dev = _device()
    b_v, b_p = r_upload(V), r_upload(perm)
    off = int(offset)
    m = min(int(limit), max(0, n - off)) if off < n else 0
    if m == 0:
        return _np.zeros(0, dtype=_np.int32)
    b_sp = r_slice(b_p, m, off)
    b_o = r_alloc(m * 4)
    _dispatch_u(dev, _GATHER_U.replace("__T__", "i32"),
                [(b_v, True), (b_sp, True), (b_o, False)], (m, 0, 0, 0), m)
    return r_download(b_o, _np.int32, m)


def gpu_execute_impl(nodes, canonical_dtype=None, format_error=None):
    """execute Planner nodes on GPU (single chunk, explicit errors)."""
    err = format_error or _fallback_err
    bufs = {}
    for p in nodes:
        op = p.get("kernel_id", p.get("op"))
        params = p["params"]
        if op == "series":
            dt = params.get("dtype", "int32")
            if dt == "float64" and _is_scaled_params(params):
                sc = float(params.get("scale", 1))
                off = float(params.get("offset", 0))
                bufs[p["out"]] = _f64_to_physical(
                    params["values"], sc, off, err,
                    f"series '{p['out']}'")
                bufs[p["out"] + "#scaled"] = (sc, off)
            elif dt == "int64":
                # Filter i64 lane entry: carry int64 bit-exact (no silent
                # int32 downcast here; non-filter ops keep their explicit
                # errors/truncation contract downstream).
                try:
                    bufs[p["out"]] = _np.ascontiguousarray(
                        params["values"], dtype=_np.int64)
                except (TypeError, ValueError):
                    raise err(f"GPU driver: series '{p['out']}' needs "
                              "rank-1 numeric",
                              fix="pass a flat int64 column",
                              doc="") from None
            else:
                bufs[p["out"]] = _as_col(params["values"], dt, err,
                                         f"series '{p['out']}'")
            if params.get("validity") is not None:
                bufs[p["out"] + "#validity"] = _np.ascontiguousarray(
                    params["validity"], dtype=bool)
        elif op == "pack_keys":
            _gb_reject_scaled_op(bufs, p["inputs"], err,
                                 f"pack_keys '{p['out']}'")
            ins = [bufs[i] for i in p["inputs"]]
            mode = params.get("mode", "pack")
            if mode == "pack" and len(ins) == 2:
                a0 = _np.asarray(ins[0])
                if a0.dtype == _np.dtype(_np.int64):
                    bufs[p["out"]] = a0  # 1-col passthrough compat
                else:
                    k1, k2 = _i32(ins[0]), _i32(ins[1])
                    m2 = int(k2.max(initial=0)) + 1
                    bufs[p["out"]] = pack_i32(k1, k2, m2)
            elif mode == "radix" and len(ins) == 2:
                k1, k2 = _i32(ins[0]), _i32(ins[1])
                rad = params.get("radix")
                m2 = int(rad[1]) if rad else int(k2.max(initial=0)) + 1
                comp = _i32(k1) * _np.int32(m2) + _i32(k2)
                # golden path stays on GPU kernel (same math, one alloc)
                bufs[p["out"]] = pack_i32(k1, k2, m2)
                _ = comp
            elif mode == "pack" and len(ins) == 1:
                bufs[p["out"]] = _np.asarray(ins[0])
            else:
                raise err(f"GPU driver: pack_keys mode '{mode}' x{len(ins)} "
                          "needs 1-2 int32 code columns (hash/CPU-only)",
                          fix="use mode='pack'/'radix' with 1-2 int32 cols",
                          doc="")
        elif op == "compare":
            _gb_reject_scaled_op(bufs, p["inputs"], err,
                                 f"compare '{p['out']}'")
            a = bufs[p["inputs"][0]]
            dt = ("float32" if _np.asarray(a).dtype == _np.dtype(_np.float32)
                  else "int32")
            if len(p["inputs"]) > 1:
                b = bufs[p["inputs"][1]]
                if _np.asarray(b).dtype == _np.dtype(_np.float64):
                    raise err("GPU driver: compare float64 needs CPU/float32",
                              fix="cast to float32 or backend='cpu'",
                              doc="")
                r = compare(a, _np.ascontiguousarray(b), params["op"], dt)
            else:
                v = params.get("value")
                if isinstance(v, float):
                    if dt == "int32":
                        dt = "float32"
                        a = _np.ascontiguousarray(a, dtype=_np.float32)
                    r = compare(a, float(v), params["op"], dt)
                else:
                    r = compare(a, int(v), params["op"], dt)
            bufs[p["out"]] = r
            keep = _gb_keep(p["inputs"], bufs)  # DELTA-3: validity=AND
            if keep is not None:
                bufs[p["out"] + "#validity"] = keep
        elif op == "mask":
            _gb_reject_scaled_op(bufs, p["inputs"], err,
                                 f"mask '{p['out']}'")
            a = bufs[p["inputs"][0]]
            b = bufs[p["inputs"][1]] if len(p["inputs"]) > 1 else None
            bufs[p["out"]] = mask_op(a, b, params["op"])
            keep = _gb_keep(p["inputs"], bufs)  # DELTA-3: validity=AND
            if keep is not None:
                bufs[p["out"] + "#validity"] = keep
        elif op == "filter":
            _gb_reject_scaled_op(bufs, p["inputs"], err,
                                 f"filter '{p['out']}'")
            a = bufs[p["inputs"][0]]
            mk = _np.asarray(bufs[p["inputs"][1]]).astype(_np.int32).ravel()
            mv = bufs.get(p["inputs"][1] + "#validity")
            if mv is not None:
                # 3VL (DELTA-3, same as CPU): invalid mask rows excluded.
                mk = mask_op(mk, _mask_u32(_np.asarray(mv, dtype=bool)),
                             "and")
            eff = _np.ascontiguousarray(mk.astype(_np.int32)).ravel()
            adt = _np.asarray(a).dtype
            if adt == _np.dtype(_np.float32):
                dt = "float32"
            elif adt == _np.dtype(_np.int32):
                dt = "int32"
            elif adt == _np.dtype(bool):
                dt = "int32"
                a = _np.ascontiguousarray(_np.asarray(a).astype(_np.int32))
            elif adt == _np.dtype(_np.int64):
                dt = "int64"
            else:
                raise err(f"GPU driver: filter needs int32/float32/int64, "
                          f"got {adt}",
                          fix="pass int32/float32/int64 columns or "
                              "backend='cpu'",
                          doc="")
            bufs[p["out"]] = filter_compact(a, eff, dt)
            v_valid = bufs.get(p["inputs"][0] + "#validity")
            if v_valid is not None:
                vv = _np.ascontiguousarray(
                    _np.asarray(v_valid, dtype=bool)).ravel()
                eff_b = eff.astype(bool)
                if eff_b.size != vv.size:
                    raise err("GPU driver: filter values/mask size mismatch",
                              fix="pass equal-length values and mask",
                              doc="")
                bufs[p["out"] + "#validity"] = vv[eff_b]
        elif op == "gather":
            _gb_reject_scaled_op(bufs, p["inputs"], err,
                                 f"gather '{p['out']}'")
            a = bufs[p["inputs"][0]]
            ix = bufs[p["inputs"][1]]
            _ix = _np.asarray(ix)
            if _ix.dtype.kind not in "iu":
                raise err(
                    f"GPU driver: gather needs integer indices, got {_ix.dtype}",
                    fix="pass int32/int64 positions",
                )
            _n = int(_np.asarray(a).size)
            if int(_ix.size) and (bool((_ix < 0).any())
                                  or bool((_ix >= _n).any())):
                raise err(
                    "GPU driver: gather index out of range",
                    fix="pass 0 <= i < n",
                )
            dt = ("float32" if _np.asarray(a).dtype == _np.dtype(_np.float32)
                  else "int32")
            bufs[p["out"]] = gather(a, ix, dt)
            v_valid = bufs.get(p["inputs"][0] + "#validity")
            if v_valid is not None:
                bufs[p["out"] + "#validity"] = _np.asarray(v_valid)[_ix]
        elif op == "reduce":
            _gb_reject_scaled_op(bufs, p["inputs"], err,
                                 f"reduce '{p['out']}'")
            a = _np.asarray(bufs[p["inputs"][0]])
            fn = params.get("fn", params.get("op", "sum"))
            dt = ("float32" if a.dtype == _np.dtype(_np.float32) else "int32")
            if fn == "count":
                bufs[p["out"]] = int(a.size)
            elif fn in ("sum", "mean"):
                part = reduce_partials(a, dt, "sum")
                tot = (int(_np.asarray(part, dtype=_np.int64).sum())
                       if dt == "int32"
                       else float(_np.asarray(part,
                                              dtype=_np.float64).sum()))
                bufs[p["out"]] = (tot / int(a.size) if fn == "mean" and
                                  a.size else tot if fn == "sum"
                                  else float("nan"))
            else:
                raise err(f"GPU driver: reduce '{fn}' CPU-only "
                          "(min/max/var/std need scan/native path)",
                          fix="use sum/count/mean on GPU or backend='cpu'",
                          doc="")
        elif op == "groupby":
            if params.get("result", "dict") == "carry":
                raise err("GPU driver: groupby result='carry' CPU-only "
                          "(Carry lives in the CPU extension)",
                          fix="use result='dict' or backend='cpu'",
                          doc="")
            v = _np.asarray(bufs[p["inputs"][0]])
            k = _np.asarray(bufs[p["inputs"][1]])
            if v.size != k.size:
                raise err(
                    f"GPU driver: groupby values/keys size mismatch "
                    f"{v.size} != {k.size}",
                    fix="pass equal-length values and keys",
                    doc="")
            gop = params["op"]
            if gop not in ("sum", "count", "mean"):
                raise err(f"GPU driver: unknown groupby op '{gop}'",
                          fix="use one of sum/count/mean "
                              "(or groupby_multi for fused)",
                          doc="")
            keep = _gb_keep(p["inputs"], bufs)
            if keep is not None:
                ix = _np.flatnonzero(keep)
                kk, vv = k[ix], v[ix]
            else:
                kk, vv = k, v
            _gb_reject_scaled_keys(bufs, [p["inputs"][1]], err,
                                   f"groupby '{p['out']}'")
            vsc = _gb_scaled_sidecar(bufs, p["inputs"][0])
            if vsc is not None:
                phys = _np.ascontiguousarray(vv, dtype=_np.int32)
                counts, (sums_phys,) = groupby_dense_impl(kk, [(phys, "int32")])
                uk, cc, (ss_phys,) = _gb_compact(counts, [sums_phys])
                if gop == "count":
                    bufs[p["out"]] = _gb_flat(uk, cc, ss_phys, "count")
                else:
                    sums_log, _ = _gb_unscale(ss_phys, cc, *vsc)
                    bufs[p["out"]] = _gb_flat(uk, cc, sums_log, gop)
            else:
                kind = _gb_vkind(vv, err, f"groupby '{p['out']}'")
                counts, (sums,) = groupby_dense_impl(kk, [(vv, kind)])
                uk, cc, (ss,) = _gb_compact(counts, [sums])
                bufs[p["out"]] = _gb_flat(uk, cc, ss, gop)
        elif op == "groupby_multi":
            if params.get("result", "dict") == "carry":
                raise err("GPU driver: groupby_multi result='carry' CPU-only "
                          "(Carry lives in the CPU extension)",
                          fix="use result='dict' or backend='cpu'",
                          doc="")
            cols = params.get("cols")
            if cols is None:
                cols, multi = [p["inputs"][0]], False
                key_name = p["inputs"][1]
                ops_map = {cols[0]: list(params["ops"])}
            else:
                multi = True
                cols = list(cols)
                key_name = p["inputs"][-1]
                ops_map = {c: list(oc)
                           for c, oc in params["ops"].items()}
            k = _np.asarray(bufs[key_name])
            vs = [_np.asarray(bufs[c]) for c in cols]
            for vv in vs:
                if vv.size != k.size:
                    raise err(
                        "GPU driver: groupby_multi values/keys size mismatch",
                        fix="pass equal-length values and keys",
                        doc="")
            keep = _gb_keep(p["inputs"], bufs)
            if keep is not None:
                ix = _np.flatnonzero(keep)
                kk = k[ix]
                vks = [vv[ix] for vv in vs]
            else:
                kk, vks = k, vs
            _gb_reject_scaled_keys(bufs, [key_name], err,
                                   f"groupby_multi '{p['out']}'")
            scales = [_gb_scaled_sidecar(bufs, c) for c in cols]
            kinds = []
            phys_cols = []
            for vv, vsc, c in zip(vks, scales, cols):
                if vsc is not None:
                    kinds.append("int32")
                    phys_cols.append(_np.ascontiguousarray(vv, dtype=_np.int32))
                else:
                    kinds.append(_gb_vkind(vv, err,
                                           f"groupby_multi '{p['out']}'"))
                    phys_cols.append(vv)
            counts, sums_list = groupby_dense_impl(
                kk, [(vv, kd) for vv, kd in zip(phys_cols, kinds)])
            uk, cc, ss_list = _gb_compact(counts, sums_list)
            unscaled_sums, unscaled_means = [], []
            for ss_phys, vsc, c in zip(ss_list, scales, cols):
                if vsc is not None:
                    s_log, m_log = _gb_unscale(ss_phys, cc, *vsc)
                    unscaled_sums.append(s_log)
                    unscaled_means.append(m_log)
                else:
                    unscaled_sums.append(ss_phys)
                    unscaled_means.append(None)
            if not multi:
                c = cols[0]
                want = ops_map[c]
                ss = unscaled_sums[0]
                mm = (unscaled_means[0] if scales[0] is not None
                      else _gb_means(ss, cc, "mean" in want))
                bufs[p["out"]] = _gb_single(uk, cc, ss, mm, want)
            else:
                s_out = unscaled_sums
                m_out = [(unscaled_means[i] if scales[i] is not None
                          else _gb_means(ss, cc, "mean" in ops_map[c]))
                         for i, (ss, c) in enumerate(zip(unscaled_sums, cols))]
                bufs[p["out"]] = _gb_multi(uk, cc, cols, s_out, m_out,
                                           ops_map)
        elif op == "sort":
            if len(p["inputs"]) != 1:
                raise err(f"GPU driver: sort of {len(p['inputs'])} keys CPU-only "
                          "(pack_keys radix/int32-direct + single sort owns it)",
                          fix="pack composite keys first, then sort the packed col",
                          doc="")
            kk = _np.asarray(bufs[p["inputs"][0]])
            if kk.dtype == _np.dtype(bool) or kk.dtype.kind not in "if":
                raise err(f"GPU driver: sort needs int32/float32 keys, "
                          f"got {kk.dtype} (encode TEXT first or backend='cpu')",
                          fix="pass int32/float32 keys or backend='cpu'",
                          doc="")
            if kk.dtype == _np.dtype(_np.float64) or kk.dtype == _np.dtype(_np.int64):
                raise err(f"GPU driver: sort of {kk.dtype} CPU-only "
                          "(int64 exactness / f64 MapF64 guard, spec 06)",
                          fix="pass int32/finite-float32 keys or backend='cpu'",
                          doc="")
            desc = params.get("descending", [False])
            one = bool(desc[0]) if isinstance(desc, (list, tuple)) else bool(desc)
            keep = bufs.get(p["inputs"][0] + "#validity")
            if keep is not None:
                m = _np.asarray(keep, dtype=bool).ravel()
                vpos = _np.flatnonzero(m)
                ipos = _np.flatnonzero(~m)
                if vpos.size == 0:
                    bufs[p["out"]] = _np.ascontiguousarray(ipos, dtype=_np.int32)
                    continue
                try:
                    sub = sort_perm_impl(kk[vpos], one)
                except ValueError as e:
                    raise err(f"GPU driver: sort {e}",
                              fix="run this sort with backend='cpu'",
                              doc="") from None
                bufs[p["out"]] = _np.ascontiguousarray(
                    _np.concatenate([vpos[sub], ipos]), dtype=_np.int32)
            else:
                try:
                    bufs[p["out"]] = sort_perm_impl(kk, one)
                except ValueError as e:
                    raise err(f"GPU driver: sort {e}",
                              fix="run this sort with backend='cpu'",
                              doc="") from None
        elif op == "slice":
            a = _np.asarray(bufs[p["inputs"][0]])
            dt = ("float32" if a.dtype == _np.dtype(_np.float32) else "int32")
            if dt == "int32" and a.dtype != _np.dtype(_np.int32):
                raise err(f"GPU driver: slice needs int32/float32, got {a.dtype}",
                          fix="pass int32/float32 columns or backend='cpu'",
                          doc="")
            bufs[p["out"]] = slice_take(a, params.get("limit"),
                                        params.get("offset", 0), dt)
            v_valid = bufs.get(p["inputs"][0] + "#validity")
            if v_valid is not None:
                vv = _np.asarray(v_valid)
                off = int(params.get("offset", 0))
                if off < vv.size:
                    lim = params.get("limit")
                    end = vv.size if lim is None else min(vv.size, off + int(lim))
                    bufs[p["out"] + "#validity"] = vv[off:end]
                else:
                    bufs[p["out"] + "#validity"] = _np.zeros(0, dtype=bool)
        elif op == "map":
            _gb_reject_scaled_op(bufs, p["inputs"], err,
                                 f"map '{p['out']}'")
            fn = params.get("fn")
            a = _np.asarray(bufs[p["inputs"][0]])
            arr = len(p["inputs"]) > 1
            b = _np.asarray(bufs[p["inputs"][1]]) if arr else params.get("value")
            if fn == "pow" and arr:
                raise err(
                    "GPU driver: map pow with array exponent: '**' scalar-exp only (spec 01)",
                    fix="pass a scalar exponent",
                )
            if a.dtype == _np.dtype(_np.int32):
                dt = "int32"
            elif a.dtype == _np.dtype(_np.float32):
                dt = "float32"
            else:
                raise err(f"GPU driver: map needs int32/float32, got {a.dtype} "
                          "(float64: MapF64 guard, use CPU)",
                          fix="pass int32/float32 columns or backend='cpu'",
                          doc="")
            if arr and _np.asarray(b).dtype != a.dtype:
                raise err(f"GPU driver: map array-array needs matching dtypes, "
                          f"got {a.dtype} vs {_np.asarray(b).dtype} (CPU promotion owns mixed)",
                          fix="cast to one dtype or run with backend='cpu'",
                          doc="")
            try:
                bufs[p["out"]] = map_elem(a, b, fn, dt)
            except ValueError as e:
                raise err(f"GPU driver: map {e}",
                          fix="run this fn/dtype with backend='cpu'",
                          doc="") from None
            keep = _gb_keep(p["inputs"], bufs)  # DELTA-3: validity=AND
            if keep is not None:
                bufs[p["out"] + "#validity"] = keep
        elif op == "shift":
            _gb_reject_scaled_op(bufs, p["inputs"], err,
                                 f"shift '{p['out']}'")
            a = _np.asarray(bufs[p["inputs"][0]])
            n = int(a.size)
            per = int(params.get("periods", 0))
            v_in = bufs.get(p["inputs"][0] + "#validity")
            if n == 0:
                bufs[p["out"]] = _np.ascontiguousarray(a.copy())
                if v_in is not None:
                    bufs[p["out"] + "#validity"] = _np.zeros(0, dtype=bool)
            elif per == 0:
                bufs[p["out"]] = _np.ascontiguousarray(a.copy())
                if v_in is not None:
                    bufs[p["out"] + "#validity"] = _np.ascontiguousarray(
                        _np.asarray(v_in, dtype=bool).copy())
            else:
                dt = ("float32" if a.dtype == _np.dtype(_np.float32)
                      else "int32")
                if dt == "int32" and a.dtype != _np.dtype(_np.int32):
                    raise err(f"GPU driver: shift needs int32/float32, "
                              f"got {a.dtype}",
                              fix="pass int32/float32 columns or backend='cpu'",
                              doc="")
                if per >= n:
                    bufs[p["out"]] = _np.zeros(n, dtype=a.dtype)
                else:
                    bufs[p["out"]] = shift_take(a, per, dt)
                ov = _np.zeros(n, dtype=bool)
                if per < n:
                    if v_in is not None:
                        ov[per:] = _np.ascontiguousarray(
                            _np.asarray(v_in, dtype=bool))[:n - per]
                    else:
                        ov[per:] = True
                bufs[p["out"] + "#validity"] = ov
        elif op == "cumsum":
            _gb_reject_scaled_op(bufs, p["inputs"], err,
                                 f"cumsum '{p['out']}'")
            if params:
                raise err(f"GPU driver: cumsum takes no params, got {params!r}",
                          fix="pass ir_cumsum(out, inp) (always inclusive, v1)",
                          doc="")
            a = _np.asarray(bufs[p["inputs"][0]])
            n = int(a.size)
            v_in = bufs.get(p["inputs"][0] + "#validity")
            if a.dtype != _np.dtype(_np.int32):
                raise err(f"GPU driver: cumsum needs int32, got {a.dtype} "
                          "(no float scan lanes on GPU, f32/f64 CPU-only)",
                          fix="pass int32 columns or backend='cpu'",
                          doc="")
            if n == 0:
                bufs[p["out"]] = _np.ascontiguousarray(a.copy())
                if v_in is not None:
                    bufs[p["out"] + "#validity"] = _np.zeros(0, dtype=bool)
            else:
                if v_in is None:
                    fill = _np.ascontiguousarray(a, dtype=_np.int32)
                else:
                    m = _np.ascontiguousarray(_np.asarray(v_in, dtype=bool))
                    fill = _np.where(m, _np.ascontiguousarray(a, dtype=_np.int32),
                                     _np.int32(0))
                bufs[p["out"]] = scan_inclusive(fill, "int32")
                if v_in is not None:
                    bufs[p["out"] + "#validity"] = _np.ascontiguousarray(
                        _np.asarray(v_in, dtype=bool).copy())
        elif op == "rng_fill_i32":
            n = int(params["n"])
            lo, hi = int(params["lo"]), int(params["hi"])
            if params.get("mode", "bits") != "bits":
                raise err(f"GPU driver: rng_fill_i32 mode "
                          f"'{params.get('mode')}' needs CPU (bits only on GPU)",
                          fix="pass mode='bits' or backend='cpu'",
                          doc="")
            if not hi > lo:
                raise err("GPU driver: rng_fill_i32 needs lo<hi (rc=-2)",
                          fix="pass lo<hi or backend='cpu'",
                          doc="")
            bufs[p["out"]] = rng_fill_i32(
                n, int(params["seed"]), int(params["stream"]),
                int(params["offset"]), lo, hi)
        elif op in ("rng_fill_f64", "rng_sample_no_replace", "rng_permutation",
                    "rng_compat", "map_round", "unique_inverse"):
            raise err(f"GPU driver: op '{op}' CPU-only "
                      "(f64 lanes need the explicit-f64 contract; "
                      "sample/permutation/compat/unique are sequential/global)",
                      fix="run this op with backend='cpu'",
                      doc="")
        else:
            raise err(f"GPU driver: op '{op}' CPU-only in v0.2 "
                      "(encode/gather-text need CPU path)",
                      fix="run this op with backend='cpu'")
    return bufs


# ---- Graph-level chunked GPU execution (VRAM-exceeding datasets) ----
# Architecture: chunk_plan(N, VRAM budget, graph) -> GPU chunk executor ->
# deterministic merge. GroupBy: partial uk/cc/ss per chunk -> host merge
# (associative). Filter: chunk-local filter -> concatenate.
# Backend contract: backend="gpu" = GPU whole OR GPU chunked. NEVER GPU->CPU.

_OVERHEAD_FRAC = 0.15  # 15% VRAM reserved for intermediate buffers / overhead
_MIN_CHUNK = 1 << 18   # 256K rows minimum chunk (amortizes dispatch overhead)


def gpu_vram_budget():
    """Query wgpu device limits -> available VRAM bytes for data.

    Returns (total_vram, budget) where budget = total * (1 - overhead).
    Falls back to 12GB estimate if query fails.
    """
    try:
        dev = _device()
        limits = dev.limits
        total = int(limits.get("maxStorageBufferBindingSize", 0))
        if total <= 0:
            total = 12 * 1024 * 1024 * 1024  # 12GB fallback
    except Exception:
        total = 12 * 1024 * 1024 * 1024
    budget = int(total * (1.0 - _OVERHEAD_FRAC))
    return total, budget


def _merge_partial(acc, key, part):
    """Host associative merge of dense groupby partial states.

    acc: {key: dense array}; part: counts or sums [M'] for this chunk.
    SUM/COUNT merge by addition (associative, deterministic, int-exact);
    MEAN carries sum+count and finalizes from them at the end. Zero-pads
    the shorter side; preserves dtype (int64 stays int64, f64 stays f64).
    """
    prev = acc.get(key)
    if prev is None:
        acc[key] = part
        return part
    if part.size > prev.size:
        big = _np.zeros(part.shape, dtype=prev.dtype)
        big[:prev.size] += prev
        big[:part.size] += part
        acc[key] = big
        return big
    if part.size < prev.size:
        prev[:part.size] += part
    else:
        prev += part
    return prev


def gpu_chunked_execute(nodes, n, chunk_size, format_error=None):
    """Execute graph on GPU in chunks with deterministic merge.

    nodes: Planner graph nodes (list of {kernel_id, inputs, params, out}).
    n: total row count.
    chunk_size: rows per chunk (from chunk_plan).
    Returns: merged result dict (same shape as gpu_execute_impl output).

    Contract: GPU-only, no CPU fallback. GroupBy partial merge is associative.
    """
    err = format_error or _fallback_err
    chunk_size = max(_MIN_CHUNK, int(chunk_size))
    num_chunks = max(1, (int(n) + chunk_size - 1) // chunk_size)
    if num_chunks <= 1:
        bufs = gpu_execute_impl(nodes, format_error=format_error)
        last = nodes[-1]["out"] if nodes else ""
        result = bufs.get(last)
        if isinstance(result, _np.generic):
            result = result.item()
        return result

    t_start = _np.float64(_np.datetime64("now").astype(float))

    # Identify source series nodes (op=series, hold actual column data)
    series_nodes = {}
    for nd in nodes:
        if nd.get("kernel_id", nd.get("op")) == "series":
            series_nodes[nd["out"]] = nd

    # Identify groupby/groupby_multi nodes for associative merge
    gb_nodes = {}
    for nd in nodes:
        op = nd.get("kernel_id", nd.get("op"))
        if op in ("groupby", "groupby_multi"):
            gb_nodes[nd["out"]] = nd

    # Accumulator: {out_name: {key: {op_name: (sum, count)}}} for mean,
    # {key: value} for sum/count single-op.
    accum = {}

    for ci in range(num_chunks):
        start = ci * chunk_size
        end = min(int(n), start + chunk_size)
        c_n = end - start

        # Create modified graph: slice source series to [start, end).
        # Scaled float64 keeps its scale/offset contract here: slicing only
        # narrows values/validity, gpu_execute_impl converts the slice with
        # _f64_to_physical guards (finite/int32-range/int64-bound), exactly
        # like the single-chunk path. No host pre-conversion.
        mod_nodes = []
        for nd in nodes:
            op = nd.get("kernel_id", nd.get("op"))
            if op == "series" and nd["out"] in series_nodes:
                params = dict(nd["params"])
                vals = params["values"]
                if hasattr(vals, "__getitem__"):
                    params["values"] = vals[start:end]
                elif isinstance(vals, (list, tuple)):
                    params["values"] = list(vals[start:end])
                validity = params.get("validity")
                if validity is not None and hasattr(validity, "__getitem__"):
                    params["validity"] = validity[start:end]
                mod_nodes.append({**nd, "params": params})
            else:
                mod_nodes.append(nd)

        # Prefix executes once per chunk (series slices + compare/mask/
        # filter/pack...). GroupBy nodes are NOT executed here: their raw
        # partial state (keys/counts/sums) comes from ONE groupby_dense_impl
        # per chunk below and merges via _merge_partial directly on host.
        # (Old shape ran the full graph incl. GPU groupby, discarded it,
        # then ran groupby_dense_impl again -- double aggregation.)
        prefix = [nd for nd in mod_nodes
                  if (nd.get("kernel_id", nd.get("op"))
                      not in ("groupby", "groupby_multi"))]
        chunk_bufs = (gpu_execute_impl(prefix, format_error=format_error)
                      if prefix else {})

        # Accumulate groupby results (associative merge)
        for gb_out, gb_nd in gb_nodes.items():
            op = gb_nd["params"].get("op", "sum")
            is_multi = gb_nd.get("kernel_id") == "groupby_multi"
            multi_ops = gb_nd["params"].get("ops") if is_multi else None

            if gb_out not in accum:
                accum[gb_out] = {}

            # Single groupby_dense_impl per chunk: raw partial state
            # (counts/sums, never pre-computed means) for associative merge.
            # MEAN carries sum+count; the final mean divides once at the end.
            key_name = gb_nd["inputs"][-1]  # last input is always keys
            val_names = gb_nd["inputs"][:-1]  # value columns

            if key_name not in chunk_bufs or any(
                    vn not in chunk_bufs for vn in val_names):
                raise err(
                    f"GPU chunked: groupby '{gb_out}' inputs unavailable "
                    "in chunk prefix (chained groupby needs backend='cpu')",
                    fix="run chained groupby graphs with backend='cpu'",
                    doc="")
            k_arr = _np.asarray(chunk_bufs.get(key_name, []), dtype=_np.int32)
            # Validity (DELTA-3, same as single chunk): NA rows out first.
            mix = None
            vmasks = [chunk_bufs[i + "#validity"] for i in gb_nd["inputs"]
                      if (i + "#validity") in chunk_bufs]
            if vmasks:
                keep = _np.asarray(vmasks[0], dtype=bool).ravel()
                for extra in vmasks[1:]:
                    keep = (keep & _np.asarray(extra, dtype=bool).ravel())
                mix = _np.flatnonzero(keep)
                k_arr = k_arr[mix]
            if k_arr.size == 0:
                continue
            kmin, kmax = int(k_arr.min()), int(k_arr.max())
            if kmin < 0:
                continue

            # Run groupby_dense_impl directly on chunk data
            if is_multi and multi_ops:
                # Multi-column: raw sums per column, merged associatively.
                vcols = []
                for cn in val_names:
                    va = _np.asarray(chunk_bufs.get(cn, []))
                    if mix is not None:
                        va = va[mix]
                    sc_side = chunk_bufs.get(cn + "#scaled")
                    if sc_side is not None:
                        # Scaled column: values are already int32 physical
                        vcols.append((_np.asarray(va, dtype=_np.int32), "int32"))
                    elif va.dtype == _np.dtype(_np.float64):
                        vcols.append((_np.asarray(va, dtype=_np.float32), "float32"))
                    elif va.dtype == _np.dtype(_np.float32):
                        vcols.append((va, "float32"))
                    else:
                        vcols.append((_np.asarray(va, dtype=_np.int32), "int32"))
                counts_chunk, sums_list_chunk = groupby_dense_impl(k_arr, vcols)
                _merge_partial(accum[gb_out], "__counts__", counts_chunk)
                for ci2, (cn, col_ops) in enumerate(multi_ops.items()):
                    _merge_partial(accum[gb_out], f"__sums_{cn}__",
                                   sums_list_chunk[ci2])
            else:
                # Single-column groupby: raw counts+sums, merged the same way.
                va = _np.asarray(chunk_bufs.get(val_names[0], [])) if val_names else _np.zeros(k_arr.size, dtype=_np.int32)
                if mix is not None and val_names:
                    va = va[mix]
                sc_side = chunk_bufs.get(val_names[0] + "#scaled") if val_names else None
                if sc_side is not None:
                    # Scaled column: values are already int32 physical
                    vcols = [(_np.asarray(va, dtype=_np.int32), "int32")]
                elif va.dtype == _np.dtype(_np.float64):
                    vcols = [(_np.asarray(va, dtype=_np.float32), "float32")]
                elif va.dtype == _np.dtype(_np.float32):
                    vcols = [(va, "float32")]
                else:
                    vcols = [(_np.asarray(va, dtype=_np.int32), "int32")]
                counts_chunk, sums_list_chunk = groupby_dense_impl(k_arr, vcols)
                _merge_partial(accum[gb_out], "__counts__", counts_chunk)
                _merge_partial(accum[gb_out], "__sums__", sums_list_chunk[0])

    # Finalize: build result dicts from accumulated counts and sums
    for gb_out, gb_nd in gb_nodes.items():
        op = gb_nd["params"].get("op", "sum")
        is_multi = gb_nd.get("kernel_id") == "groupby_multi"
        multi_ops = gb_nd["params"].get("ops") if is_multi else None
        key_name = gb_nd["inputs"][-1]
        val_names = gb_nd["inputs"][:-1]

        # Get scale/offset sidecars for each value column (from series nodes)
        scales = {}
        for vn in val_names:
            for nd in nodes:
                if nd.get("out") == vn and nd.get("kernel_id") == "series":
                    p = nd.get("params", {})
                    if "scale" in p:
                        scales[vn] = (float(p["scale"]), float(p["offset"]))
                    break

        counts = accum.get(gb_out, {}).get("__counts__")
        if counts is None:
            chunk_bufs[gb_out] = {}
            continue
        # Find non-zero groups
        nz = _np.flatnonzero(counts > 0)
        ukeys = nz.astype(_np.int64)

        if is_multi and multi_ops:
            finalized = {}
            for i, uk in enumerate(ukeys):
                k = int(uk)
                finalized[k] = {}
                c = int(counts[uk])
                for cn, col_ops in multi_ops.items():
                    finalized[k][cn] = {}
                    s_arr = accum[gb_out].get(f"__sums_{cn}__")
                    sc = scales.get(cn)
                    for op_name in col_ops:
                        if op_name == "count":
                            finalized[k][cn][op_name] = c
                        elif op_name == "sum":
                            raw_sum = int(s_arr[uk]) if s_arr is not None else 0
                            if sc is not None:
                                # Unscale: sum_logical = sum_phys * scale + offset * count
                                finalized[k][cn][op_name] = float(raw_sum) * sc[0] + sc[1] * c
                            else:
                                finalized[k][cn][op_name] = raw_sum
                        elif op_name == "mean":
                            s = float(s_arr[uk]) if s_arr is not None else 0.0
                            if sc is not None:
                                # Unscale: mean = sum_phys * scale / count + offset
                                finalized[k][cn][op_name] = s * sc[0] / max(c, 1) + sc[1]
                            else:
                                finalized[k][cn][op_name] = s / max(c, 1)
            chunk_bufs[gb_out] = finalized
        else:
            finalized = {}
            s_arr = accum.get(gb_out, {}).get("__sums__")
            sc = scales.get(val_names[0]) if val_names else None
            for i, uk in enumerate(ukeys):
                k = int(uk)
                c = int(counts[uk])
                if op == "count":
                    finalized[k] = c
                elif op == "sum":
                    raw_sum = int(s_arr[uk]) if s_arr is not None else 0
                    if sc is not None:
                        finalized[k] = float(raw_sum) * sc[0] + sc[1] * c
                    else:
                        finalized[k] = raw_sum
                elif op == "mean":
                    s = float(s_arr[uk]) if s_arr is not None else 0.0
                    if sc is not None:
                        finalized[k] = s * sc[0] / max(c, 1) + sc[1]
                    else:
                        finalized[k] = s / max(c, 1)
            chunk_bufs[gb_out] = finalized

    # Return last output (same contract as gpu_execute_impl)
    last = nodes[-1]["out"] if nodes else ""
    result = chunk_bufs.get(last)
    if isinstance(result, _np.generic):
        result = result.item()

    t_end = _np.float64(_np.datetime64("now").astype(float))
    chunk_ms = float((t_end - t_start) * 1000.0)

    return result


# ---- SPEC-DELTA-11: streaming GroupBy resident accumulators (NEW path) ----
# Design (mapping strategy): dense slot = key, 0 <= k < M. M comes from the
# global M-contract (groupindex dense_by_code / pack span), NEVER per-chunk:
# pass 1 = host min/max probe (vectorized C-speed, probe class like DELTA-6,
# not aggregation) over all chunks -> global M (Q2: global m2 first, then
# M = (max(k1)+1)*m2, so packed slots are stable across chunks). Pass 2 =
# allocate counts[M]+sums-hi[M]+sums-lo[M]+lo-carry[M] ONCE on device
# (zeros upload) and accumulate every chunk with atomicAdd (new kernel below,
# existing kernels untouched). End: single D2H of 4*M -> host int64 finalize
# (hi*65536+car*2**32+lo, compact counts>0, mean = sums/counts). atomicAdd integer addition is
# commutative/associative -> chunk order does not affect the result.
# Q1 (M=100) contention is measured, not guessed (see bench file).
# Scope: int32 single value column (Q1/Q2/Q3-v1); scaled-f64 reuses this path
# on int32 physical ticks + host _gb_unscale (no new kernel). No CPU fallback:
# dense violations (negative keys) raise explicit errors like DELTA-6.
_GB_RESIDENT_ACCUM_WGSL = """@group(0) @binding(0) var<storage,read> keys: array<i32>;
@group(0) @binding(1) var<storage,read> hi: array<i32>;
@group(0) @binding(2) var<storage,read> lo: array<u32>;
@group(0) @binding(3) var<storage,read_write> gcnt: array<atomic<u32>>;
@group(0) @binding(4) var<storage,read_write> ghi: array<atomic<i32>>;
@group(0) @binding(5) var<storage,read_write> glo: array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> gcar: array<atomic<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n: u32 = arrayLength(&keys);
  let i: u32 = g.x;
  if (i < n) {
    let k: u32 = u32(keys[i]);
    atomicAdd(&gcnt[k], 1u);
    atomicAdd(&ghi[k], hi[i]);
    let old: u32 = atomicAdd(&glo[k], lo[i]);
    if (old + lo[i] < old) { atomicAdd(&gcar[k], 1u); }
  }
}"""


def resident_accum_alloc(m):
    """SPEC-DELTA-11: allocate resident counts+hi+lo+carry (M, zeroed).

    4x M u32/i32 words. gcar counts lo-word overflows so the total
    hi*65536 + car*2**32 + glo is exact in int64 for any per-group count
    (global atomics alone would wrap lo at 2**32: Q1 100M = 1M rows/group
    x 65535 = 6.5e10 > u32max). One M-sized H2D (zeros); caller owns bufs.
    """
    m = int(m)
    if m <= 0:
        raise ValueError(f"resident accum: M={m} needs M >= 1")
    z_u = _np.zeros(m, dtype=_np.uint32)
    z_i = _np.zeros(m, dtype=_np.int32)
    return r_upload(z_u), r_upload(z_i), r_upload(z_u), r_upload(z_u)


def resident_accum_chunk(b_cnt, b_hi, b_lo, b_car, keys_chunk, vals_chunk):
    """SPEC-DELTA-11: H2D one chunk -> atomicAdd into resident accums.

    keys_chunk/vals_chunk: int32 numpy (same length). No readback; chunk
    upload buffers are transient (freed by caller/GC after submit).
    """
    dev = _device()
    kk = _np.ascontiguousarray(keys_chunk, dtype=_np.int32)
    vv = _np.ascontiguousarray(vals_chunk, dtype=_np.int32)
    if kk.size != vv.size:
        raise ValueError(f"resident accum chunk size {kk.size} != {vv.size}")
    if kk.size == 0:
        return
    hi, lo = _gb_split(vv)
    _gb_run_global_bufs(dev, _GB_RESIDENT_ACCUM_WGSL,
                         [(r_upload(kk), True), (r_upload(hi), True),
                          (r_upload(lo.view(_np.int32)), True)],
                         [b_cnt, b_hi, b_lo, b_car], int(kk.size))


def resident_accum_finalize(b_cnt, b_hi, b_lo, b_car, m):
    """SPEC-DELTA-11: single M-sized D2H -> host (ukeys, counts, sums).

    total = hi*65536 + car*2**32 + glo in int64 (carry makes it exact for
    any per-group count). ukeys int64 sorted ascending (present groups
    only). Mean = sums/counts on host (same formula as _gb_means).
    """
    m = int(m)
    cnt = r_download(b_cnt, _np.uint32, m).astype(_np.int64)
    ghi = r_download(b_hi, _np.int32, m).astype(_np.int64)
    glo = r_download(b_lo, _np.uint32, m).astype(_np.int64)
    gcar = r_download(b_car, _np.uint32, m).astype(_np.int64)
    acc = ghi * _np.int64(65536) + gcar * _np.int64(2 ** 32) + glo
    uk, cc, (ss,) = _gb_compact(cnt, [acc])
    return uk, cc, ss


def groupby_resident_accumulate(keys, values, m=None, chunk_rows=None,
                                chunk_order=None):
    """SPEC-DELTA-11: streaming GroupBy with device-resident accumulators.

    keys/values: int32 numpy, length n. m: global group count (slot bound);
    None -> two-pass (pass 1: host min/max probe -> M = kmax+1). chunk_rows:
    rows per H2D+dispatch (default NUMFAST_GPU_GROUPBY_CHUNK). chunk_order:
    optional permutation of chunk indices (associativity probe; default fwd).
    Returns (ukeys, counts, sums, stats) with stats = {m, n, num_chunks,
    h2d_ms, kernel_ms, d2h_ms, merge_ms, d2h_bytes}. merge_ms ~= 0 by design
    (no per-chunk host merge; only the final compact). Bit-exact vs numpy
    bincount and vs groupby_dense_impl (same hi/lo integer math + lo-carry).
    """
    import time as _time
    k = _np.ascontiguousarray(keys, dtype=_np.int32).ravel()
    v = _np.ascontiguousarray(values, dtype=_np.int32).ravel()
    n = int(k.size)
    if v.size != n:
        raise ValueError(f"resident accumulate size {v.size} != {n}")
    if n == 0:
        z = _np.zeros(0, dtype=_np.int64)
        st = {"m": 0, "n": 0, "num_chunks": 0, "h2d_ms": 0.0,
              "kernel_ms": 0.0, "d2h_ms": 0.0, "merge_ms": 0.0,
              "d2h_bytes": 0}
        return z, z, z, st
    if m is None:
        kmin, kmax = int(k.min()), int(k.max())
        if kmin < 0:
            raise ValueError(f"resident accumulate: dense-only, kmin={kmin} "
                             "(needs backend='cpu')")
        m = kmax + 1
    m = int(m)
    if m <= 0:
        raise ValueError(f"resident accumulate: M={m} needs M >= 1")
    cs = int(chunk_rows) if chunk_rows else _gb_chunk_rows()
    cs = max(1, cs)
    nch = (n + cs - 1) // cs
    order = list(range(nch)) if chunk_order is None else list(chunk_order)
    t = _time.perf_counter()
    acc = resident_accum_alloc(m)
    h2d_ms = (_time.perf_counter() - t) * 1000
    kernel_ms = 0.0
    for ci in order:
        a = ci * cs
        b = min(n, a + cs)
        if b <= a:
            continue
        t0 = _time.perf_counter()
        dev = _device()
        kk = _np.ascontiguousarray(k[a:b], dtype=_np.int32)
        hi, lo = _gb_split(_np.ascontiguousarray(v[a:b], dtype=_np.int32))
        b_kk, b_hi, b_lo = r_upload(kk), r_upload(hi), \
            r_upload(lo.view(_np.int32))
        h2d_ms += (_time.perf_counter() - t0) * 1000
        t1 = _time.perf_counter()
        _gb_run_global_bufs(dev, _GB_RESIDENT_ACCUM_WGSL,
                             [(b_kk, True), (b_hi, True), (b_lo, True)],
                             list(acc), b - a)
        kernel_ms += (_time.perf_counter() - t1) * 1000
    t = _time.perf_counter()
    uk, cc, ss = resident_accum_finalize(*acc, m)
    d2h_ms = (_time.perf_counter() - t) * 1000
    t = _time.perf_counter()
    _ = _np.flatnonzero(cc >= 0)  # compact already done in finalize; ~0 work
    merge_ms = (_time.perf_counter() - t) * 1000
    st = {"m": m, "n": n, "num_chunks": nch, "h2d_ms": h2d_ms,
          "kernel_ms": kernel_ms, "d2h_ms": d2h_ms, "merge_ms": merge_ms,
          "d2h_bytes": 4 * m * 4}
    return uk, cc, ss, st


def groupby_resident_accumulate_packed(k1, k2, values, m2=None,
                                       chunk_rows=None, chunk_order=None):
    """SPEC-DELTA-11: Q2-style packed streaming accumulate (global m2).

    k1/k2/values: int32 numpy, length n. m2: global second-column radix;
    None -> two-pass (pass 1 probe: m2 = max(k2)+1). Packed slot
    pk = k1*m2 + k2 with M = (max(k1)+1)*m2 is stable across chunks (per-chunk
    m2 would mis-slot). Pack arithmetic is host int64 C-speed (mapping class,
    same as DELTA-6 counts probe); accumulation stays on device. Returns
    (ukeys, counts, sums, stats) like groupby_resident_accumulate.
    """
    a1 = _np.ascontiguousarray(k1, dtype=_np.int32).ravel()
    a2 = _np.ascontiguousarray(k2, dtype=_np.int32).ravel()
    vv = _np.ascontiguousarray(values, dtype=_np.int32).ravel()
    n = int(a1.size)
    if a2.size != n or vv.size != n:
        raise ValueError(f"resident packed size {a2.size}/{a1.size}/{vv.size}")
    if n == 0:
        z = _np.zeros(0, dtype=_np.int64)
        st = {"m": 0, "n": 0, "num_chunks": 0, "h2d_ms": 0.0,
              "kernel_ms": 0.0, "d2h_ms": 0.0, "merge_ms": 0.0,
              "d2h_bytes": 0}
        return z, z, z, st
    gm2 = int(m2) if m2 is not None else int(a2.max()) + 1
    m = (int(a1.max()) + 1) * gm2
    pk = a1.astype(_np.int64) * _np.int64(gm2) + a2.astype(_np.int64)
    uk, cc, ss, st = groupby_resident_accumulate(
        pk.astype(_np.int32), vv, m=m, chunk_rows=chunk_rows,
        chunk_order=chunk_order)
    st["m2"] = gm2
    return uk, cc, ss, st


# ---- Travel batch (E1): per-lane ceil-less floor+min1, u32-only ----
# Design E1 (exact order): 1 thread=1 lane, workgroup 256, grid ceil(N/256),
# coalesced contiguous, no shared (reuse none); inputs dist u32 / speed u32 /
# K u32 uniform (host widens u16->u32, validates K 1..65535, K==0 BAD_RANGE
# before dispatch; Rust validates totals); per-lane order: INF-guard
# (d/s==0xFFFFFFFF->INF skip) -> speed==0->INF skip divide -> hi-lo split
# (16-bit halves) + half carry -> hi>=speed->INF early-out -> shift-subtract
# long-div 32 iter -> v==0? d>0->1:0; v>=0xFFFFFFFF->INF (0xFFFFFFFE stays).
# No atomics/shared on outputs, no f32, no saturate-on-hi (counterexample E1).
# Directed/parallel = storage only: zero cross-lane ops, no reduction.
# Intern stays CPU (Rust cost_intern), GPU never touches it.
_TRAVEL_U = """@group(0) @binding(0) var<storage,read> dist: array<u32>;
@group(0) @binding(1) var<storage,read> spd: array<u32>;
@group(0) @binding(2) var<storage,read_write> o: array<u32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i: u32 = g.x;
  if (i < p.x) {
    let d: u32 = dist[i];
    let s: u32 = spd[i];
    let K: u32 = p.y;
    let INF: u32 = 0xFFFFFFFFu;
    var v: u32 = INF;
    if (d == INF || s == INF) {
      v = INF;
    } else if (s == 0u) {
      v = INF;
    } else {
      let d0: u32 = d & 0xFFFFu;
      let d1: u32 = d >> 16u;
      let k0: u32 = K & 0xFFFFu;
      let k1: u32 = K >> 16u;
      let p0: u32 = d0 * k0;
      let p1: u32 = d0 * k1;
      let p2: u32 = d1 * k0;
      let p3: u32 = d1 * k1;
      let mid: u32 = (p0 >> 16u) + (p1 & 0xFFFFu) + (p2 & 0xFFFFu);
      var lo: u32 = (p0 & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);
      var hi: u32 = (p1 >> 16u) + (p2 >> 16u) + p3 + (mid >> 16u);
      let half: u32 = s >> 1u;
      let lo_a: u32 = lo + half;
      let carry: u32 = select(0u, 1u, lo_a < lo);
      lo = lo_a;
      hi = hi + carry;
      if (hi >= s) {
        v = INF;
      } else {
        var rem_hi: u32 = 0u;
        var rem_lo: u32 = hi;
        var q: u32 = 0u;
        for (var b: u32 = 0u; b < 32u; b = b + 1u) {
          let bit: u32 = (lo >> (31u - b)) & 1u;
          let c: u32 = (rem_lo >> 31u) & 1u;
          rem_hi = (rem_hi << 1u) | c;
          rem_lo = (rem_lo << 1u) | bit;
          if (rem_hi > 0u || rem_lo >= s) {
            if (rem_lo >= s) {
              rem_lo = rem_lo - s;
            } else {
              rem_lo = rem_lo - s;
              rem_hi = rem_hi - 1u;
            }
            q = (q << 1u) | 1u;
          } else {
            q = q << 1u;
          }
        }
        v = q;
        if (v == 0u) {
          if (d > 0u) { v = 1u; } else { v = 0u; }
        }
        if (v >= INF) { v = INF; }
      }
    }
    o[i] = v;
  }
}"""


def travel_batch_gpu(dist, speed, K):
    """WGSL travel batch: per-lane floor(d*K/speed), min-1, INF guards (E1).

    Same meaning as CPU resolve (adapter outside): dist/speed array-like
    length N (u16 widened to u32 host-side, u32 incl. INF sentinel kept);
    K scalar u32 uniform 1..65535 (else BAD_RANGE, no dispatch). Returns
    u32[N] (0xFFFFFFFF=INF, 0xFFFFFFFE stays). No intern handling here.
    """
    aK = _np.asarray(K)
    k = int(aK.ravel()[0]) if aK.size > 1 else int(aK.ravel()[0] if aK.size == 1 else K)
    if k == 0 or not (1 <= k <= 65535):
        raise ValueError(f"BAD_RANGE: travel K={k} needs 1..65535")
    D = _np.ascontiguousarray(_np.asarray(dist).ravel(), dtype=_np.uint32)
    S = _np.ascontiguousarray(_np.asarray(speed).ravel(), dtype=_np.uint32)
    if D.size != S.size:
        raise ValueError(f"travel size {D.size} != {S.size}")
    n = int(D.size)
    if n == 0:
        return _np.zeros(0, dtype=_np.uint32)
    dev = _device()
    out = _np.zeros(n, dtype=_np.uint32)
    (rb,) = _run_u(dev, _TRAVEL_U, [(D.tobytes(), True), (S.tobytes(), True),
                                   (out.tobytes(), False)], (n, k, 0, 0), n)
    return _np.frombuffer(rb, dtype=_np.uint32).copy()
