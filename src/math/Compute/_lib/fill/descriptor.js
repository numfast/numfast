"use strict";
function describe(params) {
  return {
    inputs: [{ name: "data", dtype: "float" }],
    outputs: [{ dtype: "float", template: "fill" }],
    workspace: [],
    uniforms: { scalar: parseFloat((params || {}).scalar || 0.0) },
  };
}
if (typeof module !== "undefined" && module.exports) { module.exports = { describe }; }
