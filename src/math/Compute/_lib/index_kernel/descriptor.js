/**
 * IndexKernel descriptor — JS port of Compute/_lib/index_kernel/descriptor.py.
 *
 * Modes: 0=const(p0), 1=arange(start=p0, step=p1), 2=linspace(a=p0, b=p1, denom=p2),
 * 3=tile(pat, k), 4=repeat(pat, rep).
 *
 * dtype semantics (owner decision, int32 index):
 * params.dtype = "float32" (default) | "int32". int32 -> OutputSlot dtype="int32",
 * mode 1 arange values = p0 + p1*i EXACT up to 2^31 (f32 2^24 limit not applied).
 * Numeric uniform flag int_mode (0|1) selects the WGSL variant; string key "dtype"
 * is removed from params — uniforms must be numeric. Descriptor is idempotent:
 * re-describe (planner/builder) reads already-set int_mode.
 *
 * tile/repeat contract: pattern is a plain INPUT buffer (InputSlot "pat"), any k
 * (no 64 limit). In int32 mode pattern values truncate to i32 (host-cast parity).
 *
 * Uniform layout: scalars only, in WGSL Params order
 * (n, mode, p0, p1, p2, k, rep, int_mode). No arrays in uniform.
 */

"use strict";

const WORKGROUP = 64;

function _int(params, key, dflt) {
  const v = params[key];
  return Math.trunc(Number(v === undefined ? dflt : v));
}

function _float(params, key, dflt) {
  const v = params[key];
  return Number(v === undefined ? dflt : v);
}

function describe(params) {
  params = params || {};
  const n = _int(params, "n", 0);
  const mode = _int(params, "mode", 0);

  if (n < 1) {
    throw new Error(`IndexKernel: n must be >= 1, got ${n}`);
  }
  if (mode !== 0 && mode !== 1 && mode !== 2 && mode !== 3 && mode !== 4) {
    throw new Error(`IndexKernel: unknown mode ${mode}, expected 0|1|2|3|4`);
  }

  // dtype -> numeric uniform flag (idempotent: describe may run twice on the
  // same params object — compile/planner/builder re-describe mutated params).
  if (params.int_mode === undefined) {
    const dtype = params.dtype !== undefined ? params.dtype : "float32";
    if (dtype !== "float32" && dtype !== "int32") {
      throw new Error(
        `IndexKernel: dtype ${JSON.stringify(dtype)} not supported ` +
        `(allowed: float32|int32)`);
    }
    params.int_mode = dtype === "int32" ? 1 : 0;
    delete params.dtype;
  }
  const int_mode = _int(params, "int_mode", 0);

  let p0; let p1; let p2;
  if (int_mode) {
    p0 = _int(params, "p0", 0);
    p1 = _int(params, "p1", 1);
    p2 = _int(params, "p2", Math.max(n - 1, 1));
  } else {
    p0 = _float(params, "p0", 0.0);
    p1 = _float(params, "p1", 1.0);
    p2 = _float(params, "p2", Number(Math.max(n - 1, 1)));
  }
  params.p0 = p0; params.p1 = p1; params.p2 = p2;

  if (mode === 2 && p2 === 0) {
    throw new Error("IndexKernel: linspace denominator p2 must be != 0");
  }

  let k = _int(params, "k", 1);
  let rep = _int(params, "rep", 1);
  if (mode === 3 || mode === 4) {
    // Pattern — input buffer "pat" (any k), no 64 limit.
    if (k < 1) {
      throw new Error("IndexKernel: pattern must contain at least 1 element");
    }
  } else {
    k = 1;
  }
  if (mode === 4 && rep < 1) {
    throw new Error(`IndexKernel: repeat requires rep >= 1, got ${rep}`);
  }
  if (mode !== 3 && mode !== 4) {
    rep = 1;
  }

  const uniforms = {
    n: n, mode: mode, p0: p0, p1: p1, p2: p2,
    k: k, rep: rep, int_mode: int_mode,
  };

  return {
    inputs: (mode === 3 || mode === 4)
      ? [{ name: "pat", dtype: int_mode ? "int" : "float" }]
      : [],
    outputs: [{ dtype: int_mode ? "int32" : "float", template: `idx_${n}` }],
    uniforms: uniforms,
    output_size_fn: function () { return [n]; },
    dispatch: [Math.ceil(n / WORKGROUP), 1, 1],
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { describe, WORKGROUP };
}
