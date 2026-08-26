"use strict";
function cpu(ctx) {
  var nInputs = ctx.inputs.length;
  var views = [];
  for (var k = 0; k < nInputs; k++) {
    views.push(ctx.inputs[k].view);
  }
  var dst = ctx.outputs[0].view;
  var weights = [];
  for (var w = 0; w < 4; w++) {
    weights.push(ctx.uniforms["w" + w] || 0.0);
  }
  var bias = ctx.uniforms.bias || 0.0;
  var length = views[0].length();
  for (var i = 0; i < length; i++) {
    var acc = bias;
    for (var j = 0; j < nInputs; j++) {
      acc += weights[j] * views[j].read(i);
    }
    dst.write(i, acc);
  }
}
if (typeof module !== "undefined" && module.exports) { module.exports = { cpu }; }
