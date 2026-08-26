"use strict";
function cpu(ctx) {
  var value = ctx.inputs[0].view;
  var idx = ctx.inputs[1].view;
  var dst = ctx.outputs[0].view;
  var n = value.length();
  var dstN = dst.length();
  for (var i = 0; i < n; i++) {
    var ix = Math.floor(idx.read(i));
    if (ix >= 0 && ix < dstN) {
      dst.write(ix, value.read(i));
    }
  }
}
if (typeof module !== "undefined" && module.exports) { module.exports = { cpu }; }
