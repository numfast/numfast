"use strict";
var OPS = {
  sum: function (a, b) { return a + b; },
  min: function (a, b) { return Math.min(a, b); },
  max: function (a, b) { return Math.max(a, b); },
};
var CODES = { 0: "sum", 1: "min", 2: "max" };
function cpu(ctx) {
  var src = ctx.inputs[0].view;
  var dst = ctx.outputs[0].view;
  var n = src.length();
  if (n === 0) {
    dst.write(0, 0.0);
    return;
  }
  var op = ctx.uniforms.op === undefined ? "sum" : ctx.uniforms.op;
  if (typeof op !== "string") {
    op = CODES[op];
    if (op === undefined) { throw new Error("Unknown Reduce op code: " + ctx.uniforms.op); }
  }
  var func = OPS[op];
  if (!func) { throw new Error("Unknown Reduce op: " + op); }
  var total = src.read(0);
  for (var i = 1; i < n; i++) {
    total = func(total, src.read(i));
  }
  dst.write(0, total);
}
if (typeof module !== "undefined" && module.exports) { module.exports = { cpu }; }
