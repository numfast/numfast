"use strict";
function cpu(ctx) {
  var src = ctx.inputs[0].view;
  var dst = ctx.outputs[0].view;
  var offset = parseInt(ctx.uniforms.offset || 1, 10);
  var n = src.length();
  for (var i = 0; i < n; i++) {
    var j = i - offset;
    if (j >= 0 && j < n) {
      dst.write(i, src.read(j));
    } else {
      dst.write(i, 0.0);
    }
  }
}
if (typeof module !== "undefined" && module.exports) { module.exports = { cpu }; }
