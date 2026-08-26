"use strict";
var WGSL = `
@group(0) @binding(0) var<storage, read> value: array<f32>;
@group(0) @binding(1) var<storage, read> index: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&value);
    if (i >= n) { return; }
    
    let ix = i32(index[i]);
    let dst_len = i32(arrayLength(&dst));
    if (ix >= 0 && ix < dst_len) {
        dst[ix] = value[i];
    }
}
`;
if (typeof module !== "undefined" && module.exports) { module.exports = { WGSL }; }
