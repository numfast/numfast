"use strict";
function describe(params) {
  params = params || {};
  var numInputs = parseInt(params.num_inputs || 2, 10);
  var inputs = [];
  for (var i = 0; i < numInputs; i++) {
    inputs.push({ name: "in" + i, dtype: "float" });
  }
  return {
    inputs: inputs,
    outputs: [{ dtype: "float", template: "combine_" + numInputs }],
    workspace: [],
    uniforms: {
      num_inputs: parseFloat(numInputs),
      w0: parseFloat(params.w0 || 1.0),
      w1: parseFloat(params.w1 || 0.0),
      w2: parseFloat(params.w2 || 0.0),
      w3: parseFloat(params.w3 || 0.0),
      bias: parseFloat(params.bias || 0.0),
    },
  };
}
if (typeof module !== "undefined" && module.exports) { module.exports = { describe }; }
