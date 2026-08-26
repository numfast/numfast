"use strict";
function describe(params) {
  return {
    inputs: [{ name: "data", dtype: "float" }],
    outputs: [{ dtype: "float", template: "map_" + (params.func || "7") }],
    workspace: [],
    uniforms: params || {},
  };
}
if (typeof module !== "undefined" && module.exports) { module.exports = { describe }; }
