"use strict";
var WGSL = `
struct Params {
    op: f32,
    scalar_a: f32,
    scalar_b: f32,
    use_scalar_a: f32,
    use_scalar_b: f32,
};

@group(0) @binding(0) var<storage, read> a_arr: array<f32>;
@group(0) @binding(1) var<storage, read> b_arr: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&dst);
    if (i >= n) { return; }
    
    let use_sa = params.use_scalar_a != 0.0;
    let use_sb = params.use_scalar_b != 0.0;
    
    let a_val = select(a_arr[i], params.scalar_a, use_sa);
    let b_n = arrayLength(&b_arr);
    let b_i = select(i, 0u, b_n <= 1u);
    let b_val = select(b_arr[b_i], params.scalar_b, use_sb);
    
    var result: f32;
    if (params.op == 0.0) {
        result = a_val + b_val;
    } else if (params.op == 1.0) {
        result = a_val - b_val;
    } else if (params.op == 2.0) {
        result = a_val * b_val;
    } else if (params.op == 3.0) {
        if (b_val != 0.0) { result = a_val / b_val; } else { result = 0.0; }
    } else if (params.op == 4.0) {
        result = max(a_val, b_val);
    } else if (params.op == 5.0) {
        result = min(a_val, b_val);
    } else {
        result = a_val + b_val;
    }
    
    dst[i] = result;
}
`;
if (typeof module !== "undefined" && module.exports) { module.exports = { WGSL }; }
