"use strict";
var WGSL = `
struct Params {
    min_val: f32,
    max_val: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&src);
    if (i >= n) { return; }
    
    let x = src[i];
    if (x < params.min_val) {
        dst[i] = params.min_val;
    } else if (x > params.max_val) {
        dst[i] = params.max_val;
    } else {
        dst[i] = x;
    }
}
`;
if (typeof module !== "undefined" && module.exports) { module.exports = { WGSL }; }
