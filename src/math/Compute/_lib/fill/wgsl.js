"use strict";
var WGSL = `
struct Params {
    scalar: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&src);
    if (i >= n) { return; }
    dst[i] = params.scalar;
}
`;
if (typeof module !== "undefined" && module.exports) { module.exports = { WGSL }; }
