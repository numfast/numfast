"use strict";

const WGSL = `
struct Params {
    period: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = i32(arrayLength(&src));
    let period = i32(params.period);

    var sum: f32 = 0.0;
    for (var i: i32 = 0; i < n; i = i + 1) {
        sum = sum + src[i];
        if (i >= period) {
            sum = sum - src[i - period];
        }
        if (i >= period - 1) {
            dst[i] = sum;
        } else {
            dst[i] = 0.0;
        }
    }
}
`;

module.exports = { WGSL };
