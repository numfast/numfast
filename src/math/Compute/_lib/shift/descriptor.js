"use strict";
function describe(params) {
  return {
    inputs: [{ name: "data", dtype: "float" }],
    outputs: [{ dtype: "float", template: "shift_" + (params.offset || "1") }],
    workspace: [],
    uniforms: { offset: parseFloat(params.offset || 1) },
  };
}
if (typeof module !== "undefined" && module.exports) { module.exports = { describe }; }
