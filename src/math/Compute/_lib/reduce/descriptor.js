"use strict";
var OP_NAMES = { 0: "sum", 1: "min", 2: "max" };
var OP_CODES = { sum: 0.0, min: 1.0, max: 2.0 };
function describe(params) {
  params = params || {};
  var op = params.op === undefined ? "sum" : params.op;
  var name;
  var code;
  if (typeof op === "number") {
    name = OP_NAMES[op] || "sum";
    code = op;
  } else {
    name = OP_CODES[op] !== undefined ? op : "sum";
    code = OP_CODES[name] !== undefined ? OP_CODES[name] : 0.0;
  }
  return {
    inputs: [{ name: "data", dtype: "float" }],
    outputs: [{ dtype: "float", template: name }],
    workspace: [{ dtype: "float", elements: 0 }],
    uniforms: { op: code },
    output_size_fn: function (sizes) { return [1]; },
  };
}
if (typeof module !== "undefined" && module.exports) { module.exports = { describe }; }
