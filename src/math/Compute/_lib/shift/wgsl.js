"use strict";
var WGSL = `
struct Params {
    offset: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = i32(id.x);
    let n = i32(arrayLength(&src));
    let offset = i32(params.offset);
    if (i >= n) { return; }
    
    if (i >= offset) {
        dst[i] = src[i - offset];
    } else {
        dst[i] = 0.0;
    }
}
`;
if (typeof module !== "undefined" && module.exports) { module.exports = { WGSL }; }
