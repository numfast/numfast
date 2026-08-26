"use strict";
function wgsl(uniforms) {
  uniforms = uniforms || {};
  var numInputsRaw = parseInt(uniforms.num_inputs || 2, 10);
  var numInputs = Math.max(1, Math.min(4, numInputsRaw));
  var inputBindings = [];
  var accumLines = [];
  for (var i = 0; i < numInputs; i++) {
    inputBindings.push("@group(0) @binding(" + i + ") var<storage, read> in" + i + ": array<f32>;");
    if (i === 0) {
      accumLines.push("    result = result + params.w0 * in0[i];");
    } else {
      accumLines.push("    if (ni >= " + (i + 1) + ") { result = result + params.w" + i + " * in" + i + "[i]; }");
    }
  }
  var bindingOut = numInputs;
  var bindingUniform = numInputs + 1;
  var accumCode = "    var result = params.bias;\n" + accumLines.join("\n");
  return [
    "struct Params {",
    "    num_inputs: f32,",
    "    w0: f32,",
    "    w1: f32,",
    "    w2: f32,",
    "    w3: f32,",
    "    bias: f32,",
    "};",
    "",
    inputBindings.join("\n"),
    "@group(0) @binding(" + bindingOut + ") var<storage, read_write> dst: array<f32>;",
    "@group(0) @binding(" + bindingUniform + ") var<uniform> params: Params;",
    "",
    "@compute @workgroup_size(64)",
    "fn main(@builtin(global_invocation_id) id: vec3<u32>) {",
    "    let i = id.x;",
    "    let n = arrayLength(&in0);",
    "    if (i >= n) { return; }",
    "    let ni = i32(params.num_inputs);",
    "    " + accumCode.replace(/\n/g, "\n    "),
    "    dst[i] = result;",
    "}",
  ].join("\n");
}
if (typeof module !== "undefined" && module.exports) { module.exports = { wgsl }; }
