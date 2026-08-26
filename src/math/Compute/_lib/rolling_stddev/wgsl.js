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
    let pf = f32(period);
    
    var sum: f32 = 0.0;
    var sum_sq: f32 = 0.0;
    for (var i: i32 = 0; i < n; i = i + 1) {
        let x = src[i];
        sum = sum + x;
        sum_sq = sum_sq + x * x;
        if (i >= period) {
            let old = src[i - period];
            sum = sum - old;
            sum_sq = sum_sq - old * old;
        }
        if (i >= period - 1) {
            let mean = sum / pf;
            let variance = sum_sq / pf - mean * mean;
            if (variance > 0.0) {
                dst[i] = sqrt(variance);
            } else {
                dst[i] = 0.0;
            }
        } else {
            dst[i] = 0.0;
        }
    }
}
`;

module.exports = { WGSL };
