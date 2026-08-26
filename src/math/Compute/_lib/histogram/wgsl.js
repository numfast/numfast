"use strict";

const WGSL = `
struct Uniforms {
    min_val: f32,
    max_val: f32,
    num_bins: f32,
    _pad: f32,
};

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> uniforms : Uniforms;

var<workgroup> shmem : array<atomic<u32>, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(workgroup_id) wgid : vec3<u32>) {
    let max_bins = u32(uniforms.num_bins);

    if (wgid.x > 0u) { return; }

    if (lid.x < max_bins) {
        atomicStore(&shmem[lid.x], 0u);
    }
    workgroupBarrier();

    let n = arrayLength(&input);
    let stride = 64u;
    var i = lid.x;
    while (i < n) {
        let val = input[i];
        let clamped = clamp(val, uniforms.min_val, uniforms.max_val);
        let range = uniforms.max_val - uniforms.min_val;
        var bin = 0u;
        if (range > 0.0) {
            let norm = (clamped - uniforms.min_val) / range;
            bin = min(u32(norm * f32(max_bins)), max_bins - 1u);
        }
        atomicAdd(&shmem[bin], 1u);
        i = i + stride;
    }
    workgroupBarrier();

    if (lid.x < max_bins) {
        let count = atomicLoad(&shmem[lid.x]);
        if (count > 0u) {
            output[lid.x] = f32(count);
        }
    }
}
`;

module.exports = { WGSL };
