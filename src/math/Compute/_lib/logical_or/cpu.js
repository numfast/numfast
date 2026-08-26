/**
 * LogicalOr CPU — JS reference: out[i] = a[i] | b[i] (uint32).
 */

"use strict";

function cpu(ctx) {
  const a = ctx.inputs[0].view;
  const b = ctx.inputs[1].view;
  const dst = ctx.outputs[0].view;
  const n = dst.length();
  for (let i = 0; i < n; i++) {
    dst.write(i, a.read(i) | b.read(i));
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { cpu };
}
