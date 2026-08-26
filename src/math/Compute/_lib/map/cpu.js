"use strict";
var _FUNCS = [
  Math.sin,
  Math.cos,
  Math.exp,
  Math.sqrt,
  Math.log,
  Math.abs,
  function (x) { return -x; },
  function (x) { return x * x; },
];
function cpu(ctx) {
  var src = ctx.inputs[0].view;
  var dst = ctx.outputs[0].view;
  // Parity with Python `.get("func", 7)`: default only when the key is
  // ABSENT. `|| 7` hijacked code 0 (sin) -> 7 (square).
  var funcCode = parseInt(ctx.uniforms.func === undefined ? 7 : ctx.uniforms.func, 10);
  var func = _FUNCS[funcCode];
  if (!func) { throw new Error("Unknown Map function code: " + funcCode); }
  var n = dst.length();
  for (var i = 0; i < n; i++) {
    dst.write(i, func(src.read(i)));
  }
}
if (typeof module !== "undefined" && module.exports) { module.exports = { cpu }; }
