/**
 * IndexKernel WGSL — JS port of Compute/_lib/index_kernel/wgsl.py.
 * Creation primitives: const / arange / linspace / tile / repeat.
 *
 * out[i] depends ONLY on global_invocation_id. tile/repeat pattern (mode 3/4)
 * is a plain GPU-resident input buffer pat (any k, binding 2, storage).
 *
 * dtype semantics (owner decision, int32 index):
 * - params.dtype="float32" -> array<f32> buffers, f32 arithmetic (legacy path).
 * - params.dtype="int32"   -> array<i32>, INTEGER arithmetic:
 *   mode 1 arange: v = p0 + p1*i32(i) — EXACT up to 2^31 (not limited to 2^24).
 *   i32 overflow WRAPS (two's complement) — documented contract.
 * Variant selection — wgsl(uniforms) by uniform flag int_mode (0=f32, 1=i32)
 * and pattern presence (mode 3/4 -> binding 2); without uniforms returns the
 * f32 variant without pat.
 *
 * Uniform struct holds ONLY scalars (n, mode, p0..p2, k, rep, int_mode) =
 * 32 bytes: arrays are forbidden in uniform by alignment rules.
 */

"use strict";

const _F32_HEAD = `
    var v: f32;
    if (params.mode == 0u) {
        // const(value = p0)
        v = params.p0;
    } else if (params.mode == 1u) {
        // arange(start = p0, step = p1): start + step * i
        v = params.p0 + params.p1 * f32(i);
    } else if (params.mode == 2u) {
        // linspace(a = p0, b = p1, denom = p2): a + (b - a) * i / denom
        v = params.p0 + (params.p1 - params.p0) * f32(i) / params.p2;
    }`;

const _F32_PAT = ` else if (params.mode == 3u) {
        // tile(pattern, n): out[i] = pat[i % k] — pat: GPU-resident input
        v = pat[i % params.k];
    } else {
        // repeat(pattern, repeats=r): out[i] = pat[min(i / rep, k - 1)] (u32 div)
        v = pat[min(i / params.rep, params.k - 1u)];
    }`;

const _I32_HEAD = `
    var v: i32;
    if (params.mode == 0u) {
        // const(value = p0)
        v = params.p0;
    } else if (params.mode == 1u) {
        // arange(start = p0, step = p1): EXACT i32 arithmetic up to 2^31.
        // Signed overflow WRAPS (two's complement) — documented contract.
        v = params.p0 + params.p1 * i32(i);
    } else if (params.mode == 2u) {
        // linspace integer-trunc semantics: C-style trunc division,
        // all steps in i32 domain (WRAP on overflow, documented).
        v = params.p0 + (params.p1 - params.p0) * i32(i) / params.p2;
    }`;

const _I32_PAT = ` else if (params.mode == 3u) {
        // tile(pattern, n): out[i] = pat[i % k]; pattern values truncate to i32
        v = pat[i % params.k];
    } else {
        // repeat(pattern, repeats=r): out[i] = pat[min(i / rep, k - 1)] (u32 div)
        v = pat[min(i / params.rep, params.k - 1u)];
    }`;


function _make_wgsl(int_mode, has_pat) {
  const scalar_t = int_mode ? "i32" : "f32";
  const head = int_mode ? _I32_HEAD : _F32_HEAD;
  const pat_branch = int_mode ? _I32_PAT : _F32_PAT;
  const pat_binding = has_pat
    ? `@group(0) @binding(2) var<storage, read> pat: array<${scalar_t}>;`
    : "";
  return `
struct Params {
    n: u32,
    mode: u32,
    p0: ${scalar_t},
    p1: ${scalar_t},
    p2: ${scalar_t},
    k: u32,
    rep: u32,
    int_mode: u32,
};

@group(0) @binding(0) var<storage, read_write> dst: array<${scalar_t}>;
@group(0) @binding(1) var<uniform> params: Params;
${pat_binding}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.n) { return; }
${head}${has_pat ? pat_branch : ""}
    dst[i] = v;
}
`;
}


const WGSL_FLOAT32 = _make_wgsl(false, false);
const WGSL_INT32 = _make_wgsl(true, false);

// Backward-compatible alias: default (float32) shader source.
const WGSL = WGSL_FLOAT32;


function wgsl(uniforms) {
  // Select shader variant by uniforms: int_mode flag + pattern presence.
  // mode 3/4 (tile/repeat) -> storage pat binding included; without uniforms
  // returns the f32 variant without pat.
  const int_mode = Boolean(Math.trunc(Number((uniforms || {}).int_mode || 0)));
  const mode = Math.trunc(Number((uniforms || {}).mode || 0));
  const has_pat = mode === 3 || mode === 4;
  return _make_wgsl(int_mode, has_pat);
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { WGSL, WGSL_FLOAT32, WGSL_INT32, wgsl };
}
