"use strict";
function cpu(ctx) {
  var src = ctx.inputs[0].view;
  var idx = ctx.inputs[1].view;
  var dst = ctx.outputs[0].view;
  var n = dst.length();
  var srcLen = src.length();
  for (var i = 0; i < n; i++) {
    var ix = Math.floor(idx.read(i));
    if (ix >= 0 && ix < srcLen) {
      dst.write(i, src.read(ix));
    } else {
      dst.write(i, 0.0);
    }
  }
}
if (typeof module !== "undefined" && module.exports) { module.exports = { cpu }; }
