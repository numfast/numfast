/**
 * Mask descriptor — JS version.
 * cond is uint32 array. a and b can be arrays or scalars.
 */

"use strict";

function describe(params) {
  params = params || {};
  const inputs = [{ name: "cond", dtype: "uint32" }];
  if (!Number(params.use_scalar_a || 0)) {
    inputs.push({ name: "a", dtype: "float" });
  }
  if (!Number(params.use_scalar_b || 0)) {
    inputs.push({ name: "b", dtype: "float" });
  }
  return {
    inputs: inputs,
    outputs: [{ dtype: "float", template: "mask" }],
    workspace: [],
    uniforms: {
      scalar_a: Number(params.scalar_a || 0.0),
      scalar_b: Number(params.scalar_b || 0.0),
      use_scalar_a: Number(params.use_scalar_a || 0),
      use_scalar_b: Number(params.use_scalar_b || 0),
    },
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { describe };
}
