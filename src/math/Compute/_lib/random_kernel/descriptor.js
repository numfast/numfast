"use strict";

// RandomKernel descriptor — counter-based random creation without input buffers.
// Exact port of descriptor.py: inputs=[], outputs=[rnd_{n}],
// uniforms {n, mode, seed, p0..p1}.
// Modes: 0=uniform(low=p0, high=p1), 1=int_range[p0,p1), 2=normal(loc=p0, scale=p1).
// dispatch/workgroup mirror Python WORKGROUP=64.

var WORKGROUP = 64;

function describe(params) {
  var p = params || {};
  var n = Math.trunc(Number(p.n !== undefined ? p.n : 0));
  var mode = Math.trunc(Number(p.mode !== undefined ? p.mode : 0));
  var seed = Math.trunc(Number(p.seed !== undefined ? p.seed : 42));

  if (!(n >= 1)) {
    throw new Error("RandomKernel: n must be >= 1, got " + n);
  }
  if (mode !== 0 && mode !== 1 && mode !== 2) {
    throw new Error("RandomKernel: unknown mode " + mode + ", expected 0|1|2");
  }

  var p0 = Number(p.p0 !== undefined ? p.p0 : 0.0);
  var p1 = Number(p.p1 !== undefined ? p.p1 : 1.0);

  return {
    inputs: [],
    outputs: [{ dtype: "float", template: "rnd_" + n }],
    workspace: [],
    uniforms: {
      n: n,
      mode: mode,
      seed: seed >>> 0,
      p0: p0,
      p1: p1,
    },
  };
}

if (typeof module !== "undefined" && module.exports) { module.exports = { describe: describe }; }
