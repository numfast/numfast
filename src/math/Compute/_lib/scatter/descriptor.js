"use strict";
function describe(params) {
  return {
    inputs: [
      { name: "value", dtype: "float" },
      { name: "index", dtype: "float" },
    ],
    outputs: [{ dtype: "float", template: "scatter" }],
    workspace: [],
    uniforms: {},
  };
}
if (typeof module !== "undefined" && module.exports) { module.exports = { describe }; }
