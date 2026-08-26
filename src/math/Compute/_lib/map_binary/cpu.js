"use strict";
var _OPS = [
  function (a, b) { return a + b; },
  function (a, b) { return a - b; },
  function (a, b) { return a * b; },
  function (a, b) { return b !== 0 ? a / b : 0.0; },
  function (a, b) { return Math.max(a, b); },
  function (a, b) { return Math.min(a, b); },
];
function cpu(ctx) {
  var dst = ctx.outputs[0].view;
  var useSa = parseInt(ctx.uniforms.use_scalar_a || 0, 10);
  var useSb = parseInt(ctx.uniforms.use_scalar_b || 0, 10);
  var scalarA = ctx.uniforms.scalar_a || 0.0;
  var scalarB = ctx.uniforms.scalar_b || 0.0;
  var opCode = parseInt(ctx.uniforms.op || 0, 10);
  var inputIdx = 0;
  var aView = null;
  if (!useSa) { aView = ctx.inputs[inputIdx].view; inputIdx++; }
  var bView = null;
  if (!useSb) { bView = ctx.inputs[inputIdx].view; }
  var func = _OPS[opCode];
  if (!func) { throw new Error("Unknown MapBinary op code: " + opCode); }
  var n = dst.length();
  for (var i = 0; i < n; i++) {
    var aVal = useSa ? scalarA : aView.read(i);
    var bVal = useSb ? scalarB : bView.read(bView.length() > 1 ? i : 0);
    dst.write(i, func(aVal, bVal));
  }
}
if (typeof module !== "undefined" && module.exports) { module.exports = { cpu }; }
