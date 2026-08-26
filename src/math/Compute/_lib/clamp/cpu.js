"use strict";
function cpu(ctx) {
  var src = ctx.inputs[0].view;
  var dst = ctx.outputs[0].view;
  var minVal = ctx.uniforms.min_val || 0.0;
  var maxVal = ctx.uniforms.max_val || 1.0;
  var n = src.length();
  for (var i = 0; i < n; i++) {
    var x = src.read(i);
    if (x < minVal) {
      dst.write(i, minVal);
    } else if (x > maxVal) {
      dst.write(i, maxVal);
    } else {
      dst.write(i, x);
    }
  }
}
if (typeof module !== "undefined" && module.exports) { module.exports = { cpu }; }
