/**
 * Compare WGSL — GPU: out[i] = select(0u, 1u, a[i] op b[i]).
 * Numeric op code in uniform "op": 0=gt, 1=ge, 2=lt, 3=le, 4=eq, 5=ne.
 * Dynamic bindings by scalar mode (use_scalar_a/b), like map_binary.
 */

"use strict";

function makeWgsl(numInputs) {
  const bindings = [];
  let body = "";
  const n = "arrayLength(&dst)";

  if (numInputs === 0) {
    bindings.push("    @group(0) @binding(0) var<storage, read_write> dst: array<u32>;");
    bindings.push("    @group(0) @binding(1) var<uniform> params: Params;");
    body = `    let i = id.x;
    let n = ${n};
    if (i >= n) { return; }
    let a_val = params.scalar_a;
    let b_val = params.scalar_b;
`;
  } else if (numInputs === 1) {
    bindings.push("    @group(0) @binding(0) var<storage, read> a_arr: array<f32>;");
    bindings.push("    @group(0) @binding(1) var<storage, read_write> dst: array<u32>;");
    bindings.push("    @group(0) @binding(2) var<uniform> params: Params;");
    body = `    let i = id.x;
    let n = ${n};
    if (i >= n) { return; }
    let use_sa = params.use_scalar_a != 0.0;
    let use_sb = params.use_scalar_b != 0.0;
    let a_val = select(a_arr[i], params.scalar_a, use_sa);
    let b_val = select(a_arr[i], params.scalar_b, use_sb);
`;
  } else {
    bindings.push("    @group(0) @binding(0) var<storage, read> a_arr: array<f32>;");
    bindings.push("    @group(0) @binding(1) var<storage, read> b_arr: array<f32>;");
    bindings.push("    @group(0) @binding(2) var<storage, read_write> dst: array<u32>;");
    bindings.push("    @group(0) @binding(3) var<uniform> params: Params;");
    body = `    let i = id.x;
    let n = ${n};
    if (i >= n) { return; }
    let use_sa = params.use_scalar_a != 0.0;
    let use_sb = params.use_scalar_b != 0.0;
    let a_val = select(a_arr[i], params.scalar_a, use_sa);
    let b_val = select(b_arr[i], params.scalar_b, use_sb);
`;
  }

  return `
struct Params {
    op: f32,
    scalar_a: f32,
    scalar_b: f32,
    use_scalar_a: f32,
    use_scalar_b: f32,
};

${bindings.join("\n")}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
${body}
    var r: u32;
    if (params.op == 0.0) {
        r = select(0u, 1u, a_val > b_val);
    } else if (params.op == 1.0) {
        r = select(0u, 1u, a_val >= b_val);
    } else if (params.op == 2.0) {
        r = select(0u, 1u, a_val < b_val);
    } else if (params.op == 3.0) {
        r = select(0u, 1u, a_val <= b_val);
    } else if (params.op == 4.0) {
        r = select(0u, 1u, a_val == b_val);
    } else {
        r = select(0u, 1u, a_val != b_val);
    }
    dst[i] = r;
}
`;
}

function wgslGenerator(uniforms) {
  uniforms = uniforms || {};
  const use_sa = Number(uniforms.use_scalar_a || 0);
  const use_sb = Number(uniforms.use_scalar_b || 0);
  const numInputs = (use_sa ? 0 : 1) + (use_sb ? 0 : 1);
  return makeWgsl(numInputs);
}

const WGSL = makeWgsl(2);

if (typeof module !== "undefined" && module.exports) {
  module.exports = { WGSL, makeWgsl, wgslGenerator, wgsl: wgslGenerator };
}
