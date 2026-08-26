"use strict";
function cpu(ctx) {
  var dst = ctx.outputs[0].view;
  var scalar = ctx.uniforms.scalar || 0.0;
  var n = dst.length();
  for (var i = 0; i < n; i++) {
    dst.write(i, scalar);
  }
}
if (typeof module !== "undefined" && module.exports) { module.exports = { cpu }; }
