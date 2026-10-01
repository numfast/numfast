# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Generic batch sweep over u64 bitmask lanes with predicate-swizzle probes.

Semantics (frozen, generic lanes only, no domain vocabulary):
- `masks[n]` u64 + `probes[p]` u64 (predicate-swizzle set: each probe is
  a bit-subset query) -> `hits[n*p]` u8 row-major
  (`hits[r*p+j] = 1 iff (masks[r] & probes[j]) == probes[j]`),
  `counts[p]` per-probe hit counts, `popcnt[n]` per-row popcounts.
- Caller-owned outputs (fresh arrays; inputs never mutated).

WGSL-first (u32 lo/hi split pairs: WGSL has no u64; subset test is
bit-exact on the split lanes), host-side counts/popcnt merge (same hybrid
discipline as GroupBy: GPU subset-test + host reduce). Bit-exact numpy
fallback when wgpu is missing (GPU without CPU fallback is forbidden).
"""
import os

import numpy as np

_SWEEP_WGSL = """@group(0) @binding(0) var<storage,read> m_lo: array<u32>;
@group(0) @binding(1) var<storage,read> m_hi: array<u32>;
@group(0) @binding(2) var<storage,read> q_lo: array<u32>;
@group(0) @binding(3) var<storage,read> q_hi: array<u32>;
@group(0) @binding(4) var<storage,read_write> hits: array<u32>;
@group(0) @binding(5) var<uniform> prm: vec4<u32>;
fn run(idx: u32) {
  let n: u32 = prm.x;
  let p: u32 = prm.y;
  let total: u32 = n * p;
  if (idx >= total) { return; }
  let r: u32 = idx / p;
  let j: u32 = idx % p;
  let ok_lo: u32 = (m_lo[r] & q_lo[j]) ^ q_lo[j];
  let ok_hi: u32 = (m_hi[r] & q_hi[j]) ^ q_hi[j];
  var h: u32 = 0u;
  if (ok_lo == 0u && ok_hi == 0u) { h = 1u; }
  hits[idx] = h;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  run(gid.x);
}
"""

_gpu_why = "unprobed"
_gpu_ok = None


def _gpu_probe():
    global _gpu_why, _gpu_ok
    if _gpu_why != "unprobed":
        return _gpu_ok
    if os.environ.get("NUMFAST_GPU_DISABLE") == "1":
        _gpu_why, _gpu_ok = "disabled-by-env", False
        return False
    try:
        import wgpu  # noqa: F401
        _gpu_why, _gpu_ok = "wgpu-present", True
    except Exception as e:
        _gpu_why, _gpu_ok = "unavailable:%s" % e, False
    return _gpu_ok


def bitmask_sweep_available():
    return True


def _as_u64(a, name):
    x = np.ascontiguousarray(a)
    if x.ndim != 1:
        raise ValueError(f"bitmask_sweep {name} ndim != 1")
    if x.size == 0:
        return np.zeros(0, dtype=np.uint64)
    if x.dtype == np.dtype(np.uint64):
        return x
    if x.dtype == np.dtype(np.int64):
        return x.view(np.uint64)
    raise ValueError(f"bitmask_sweep {name} dtype {x.dtype} not u64/i64")


def _fb_hits(masks, probes):
    n = int(masks.shape[0])
    p = int(probes.shape[0])
    hits = np.zeros(n * p, dtype=np.uint8)
    for r in range(n):
        m = int(masks[r])
        base = r * p
        for j in range(p):
            q = int(probes[j])
            if (m & q) == q:
                hits[base + j] = np.uint8(1)
    return hits


def _gpu_hits(m_lo, m_hi, q_lo, q_hi, n, p):
    from Drivers.GPU._lib import gpu as _gpu
    dev = _gpu._device()
    total = n * p
    out = np.zeros(total, dtype=np.uint32)
    (rb,) = _gpu._run_u(
        dev, _SWEEP_WGSL,
        [(m_lo.tobytes(), True), (m_hi.tobytes(), True),
         (q_lo.tobytes(), True), (q_hi.tobytes(), True),
         (out.tobytes(), False)],
        (n, p, 0, 0), total)
    return np.frombuffer(rb, dtype=np.uint32).astype(np.uint8)


def _popcnt_u64(masks):
    m = np.ascontiguousarray(masks, dtype=np.uint64)
    bits = np.unpackbits(m.view(np.uint8))
    return bits.reshape(int(m.shape[0]), 64).sum(axis=1).astype(np.uint64)


def bitmask_sweep(masks, probes):
    """Sweep `masks[n]` u64 against `probes[p]` u64 subset queries.

    Returns `(hits, counts, popcnt)`: `hits[n*p]` u8 row-major,
    `counts[p]` uint64 per-probe hits, `popcnt[n]` uint64 per-row popcounts.
    """
    m = _as_u64(masks, "masks")
    q = _as_u64(probes, "probes")
    n, p = int(m.shape[0]), int(q.shape[0])
    if n == 0 or p == 0:
        return (np.zeros(n * p, dtype=np.uint8),
                np.zeros(p, dtype=np.uint64),
                np.zeros(n, dtype=np.uint64))
    m_lo = (m & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    m_hi = ((m >> np.uint64(32)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    q_lo = (q & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    q_hi = ((q >> np.uint64(32)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    try:
        if _gpu_probe():
            hits = _gpu_hits(m_lo, m_hi, q_lo, q_hi, n, p)
        else:
            hits = _fb_hits(m, q)
    except Exception:
        hits = _fb_hits(m, q)
    hits = np.ascontiguousarray(hits, dtype=np.uint8)
    counts = hits.reshape(n, p).sum(axis=0).astype(np.uint64)
    popcnt = _popcnt_u64(m)
    return hits, np.ascontiguousarray(counts), np.ascontiguousarray(popcnt)
