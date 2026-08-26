"use strict";
function describe(params) {
  params = params || {};
  var inputs = [];
  if (!parseFloat(params.use_scalar_a || 0)) {
    inputs.push({ name: "a", dtype: "float" });
  }
  if (!parseFloat(params.use_scalar_b || 0)) {
    inputs.push({ name: "b", dtype: "float" });
  }
  return {
    inputs: inputs,
    outputs: [{ dtype: "float", template: "map_binary_" + (params.op || "0") }],
    workspace: [],
    uniforms: {
      op: parseFloat(params.op || 0),
      scalar_a: parseFloat(params.scalar_a || 0.0),
      scalar_b: parseFloat(params.scalar_b || 0.0),
      use_scalar_a: parseFloat(params.use_scalar_a || 0),
      use_scalar_b: parseFloat(params.use_scalar_b || 0),
    },
  };
}
if (typeof module !== "undefined" && module.exports) { module.exports = { describe }; }
