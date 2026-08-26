"use strict";
function describe(params) {
  return {
    inputs: [{ name: "data", dtype: "float" }],
    outputs: [{ dtype: "float", template: "clamp" }],
    workspace: [],
    uniforms: {
      min_val: parseFloat((params || {}).min_val || 0.0),
      max_val: parseFloat((params || {}).max_val || 1.0),
    },
  };
}
if (typeof module !== "undefined" && module.exports) { module.exports = { describe }; }
