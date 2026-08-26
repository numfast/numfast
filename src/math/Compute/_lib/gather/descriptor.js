"use strict";
function describe(params) {
  return {
    inputs: [
      { name: "src", dtype: "float" },
      { name: "index", dtype: "float" },
    ],
    outputs: [{ dtype: "float", template: "gather" }],
    workspace: [],
    uniforms: {},
  };
}
if (typeof module !== "undefined" && module.exports) { module.exports = { describe }; }
