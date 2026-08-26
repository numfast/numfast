"use strict";

// RandomKernel CPU oracle — exact port of cpu.py (PCG-u32 canonical).
// uniform/integers — bit-in-bit with WGSL: hash arithmetic in u32 WRAP
// (Math.imul + >>>0), every float op rounded to f32 via Math.fround
// (np.float32 semantics). normal — same Box-Muller with f32 rounding;
// transcendentals are a tolerance domain.

var C1 = 747796405 >>> 0;
var C2 = 2891336453 >>> 0;
var C3 = 277803737 >>> 0;
var TWO_POW_NEG32 = 2.3283064365386963e-10;        // 2^-32 (power of two, exact)
var TWO_POW_NEG33 = 1.16415321826934814453125e-10; // 2^-33, log(0) guard
var TWO_PI = Math.fround(6.2831853071795864769);   // f32 literal like WGSL/np.float32

function pcg_hash(state) {
  // state: u32 number. 3 rounds: state = (state >> ((state>>28)+4)) ^ (state*C3)
  var shift, shifted, prod;
  for (var k = 0; k < 3; k++) {
    shift = ((state >>> 28) + 4) | 0;   // range 4..19 — safe for >>>
    shifted = state >>> shift;          // u32 logical shift
    prod = Math.imul(state, C3) >>> 0;  // u32 wrap multiply
    state = (shifted ^ prod) >>> 0;
  }
  return state;
}

function hash_u01(i, seed) {
  // state = seed ^ (i*C1 + C2), all u32 WRAP; u01 = f32(h) * 2^-32 (exact scale)
  var sum = (Math.imul(i, C1) + C2) >>> 0;
  var state = ((seed >>> 0) ^ sum) >>> 0;
  return Math.fround(pcg_hash(state)) * TWO_POW_NEG32;
}

function cpu(ctx) {
  var dst = ctx.outputs[0].view;
  var n = ctx.uniforms.n | 0;
  var mode = ctx.uniforms.mode | 0;
  var seed = ctx.uniforms.seed >>> 0;
  var p0 = Math.fround(Number(ctx.uniforms.p0));
  var p1 = Math.fround(Number(ctx.uniforms.p1));

  for (var i = 0; i < n; i++) {
    var v;
    if (mode === 0) {
      // uniform(low=p0, high=p1): p0 + (p1-p0)*u01, each op rounded to f32
      var u = hash_u01(i, seed);
      var d = Math.fround(p1 - p0);
      v = Math.fround(p0 + Math.fround(d * u));
    } else if (mode === 1) {
      // int_range [p0, p1): floor(...), clamp to [p0, max(p0, p1-1)]
      var u1 = hash_u01(i, seed);
      var d1 = Math.fround(p1 - p0);
      var raw = Math.floor(Math.fround(p0 + Math.fround(d1 * u1)));
      var hi = Math.max(p0, Math.fround(p1 - 1)); // np.maximum(p0, p1 - 1.0)
      v = Math.min(Math.max(raw, p0), hi);        // np.clip(raw, p0, hi)
    } else {
      // normal(loc=p0, scale=p1): Box-Muller from hashes of counters (i, i+n)
      var u1_raw = hash_u01(i, seed);
      var u2 = hash_u01((i + n) >>> 0, seed);
      var ug = (u1_raw <= 0) ? TWO_POW_NEG33 : u1_raw; // log(0) guard
      var r = Math.sqrt(Math.fround(-2.0 * Math.log(ug)));
      var theta = Math.fround(TWO_PI * u2);
      var z = Math.fround(r * Math.cos(theta));
      v = Math.fround(p0 + Math.fround(p1 * z));
    }
    dst.write(i, v);
  }
}

if (typeof module !== "undefined" && module.exports) { module.exports = { cpu: cpu }; }
