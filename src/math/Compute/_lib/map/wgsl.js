"use strict";
var WGSL = `
struct Params {
    func: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&src)) { return; }
    
    let x = src[i];
    var result: f32;
    
    // 0=sin, 1=cos, 2=exp, 3=sqrt, 4=log, 5=abs, 6=neg, 7=square
    if (params.func == 0) {
        result = sin(x);
    } else if (params.func == 1) {
        result = cos(x);
    } else if (params.func == 2) {
        result = exp(x);
    } else if (params.func == 3) {
        result = sqrt(x);
    } else if (params.func == 4) {
        result = log(x);
    } else if (params.func == 5) {
        result = abs(x);
    } else if (params.func == 6) {
        result = -x;
    } else {
        result = x * x;  // square (default)
    }
    
    dst[i] = result;
}
`;
if (typeof module !== "undefined" && module.exports) { module.exports = { WGSL }; }
