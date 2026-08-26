/**
 * Mask WGSL — GPU: out[i] = select(b, a, cond[i] != 0u).
 * Dynamic bindings by scalar mode (use_scalar_a/b), like map_binary.
 */

"use strict";

function wgsl(uniforms) {
  uniforms = uniforms || {};
  const use_sa = Number(uniforms.use_scalar_a || 0);
  const use_sb = Number(uniforms.use_scalar_b || 0);
  const num_inputs = 1 + (use_sa ? 0 : 1) + (use_sb ? 0 : 1);

  const bindings = ["    @group(0) @binding(0) var<storage, read> cond: array<u32>;"];
  if (!use_sa) {
    bindings.push("    @group(0) @binding(1) var<storage, read> a_arr: array<f32>;");
  }
  if (!use_sb) {
    bindings.push(`    @group(0) @binding(${num_inputs - 1}) var<storage, read> b_arr: array<f32>;`);
  }
  bindings.push(`    @group(0) @binding(${num_inputs}) var<storage, read_write> dst: array<f32>;`);
  bindings.push(`    @group(0) @binding(${num_inputs + 1}) var<uniform> params: Params;`);

  const aExpr = use_sa ? "0.0" : "a_arr[i]";
  const bExpr = use_sb ? "0.0" : "b_arr[i]";

  return `
struct Params {
    scalar_a: f32,
    scalar_b: f32,
    use_scalar_a: f32,
    use_scalar_b: f32,
};

${bindings.join("\n")}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&dst);
    if (i >= n) { return; }
    let use_sa = params.use_scalar_a != 0.0;
    let use_sb = params.use_scalar_b != 0.0;
    let a_val = select(${aExpr}, params.scalar_a, use_sa);
    let b_val = select(${bExpr}, params.scalar_b, use_sb);
    dst[i] = select(b_val, a_val, cond[i] != 0u);
}
`;
}

const WGSL = wgsl({});

if (typeof module !== "undefined" && module.exports) {
  module.exports = { WGSL, wgsl };
}
