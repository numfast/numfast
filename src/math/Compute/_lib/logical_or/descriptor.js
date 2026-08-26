/**
 * LogicalOr descriptor — JS version.
 */

"use strict";

function describe(params) {
  return {
    inputs: [
      { name: "a", dtype: "uint32" },
      { name: "b", dtype: "uint32" },
    ],
    outputs: [{ dtype: "uint32", template: "logical_or" }],
    workspace: [],
    uniforms: {},
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { describe };
}
