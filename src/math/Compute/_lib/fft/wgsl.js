"use strict";

const WGSL_STAGE = `
struct Uniforms {
    stage: f32,
    N: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> uniforms : Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let stage = u32(uniforms.stage);
    let N = u32(uniforms.N);

    let stride = 1u << (stage + 1u);
    let half = 1u << stage;
    let total_butterflies = N / 2u;

    let bf = gid.x;
    if (bf >= total_butterflies) { return; }

    let group = bf / half;
    let k_in_group = bf % half;
    let k = group * stride + k_in_group;
    let j = k + half;

    // Twiddle factor: exp(-2πi * k_in_group / stride)
    let angle = -6.283185307 * f32(k_in_group) / f32(stride);
    let w_re = cos(angle);
    let w_im = sin(angle);

    // Read complex pair (k)
    let a_re = input[2u * k];
    let a_im = input[2u * k + 1u];
    // Read complex pair (j)
    let b_re = input[2u * j];
    let b_im = input[2u * j + 1u];

    // Butterfly: a + w*b, a - w*b
    let wb_re = w_re * b_re - w_im * b_im;
    let wb_im = w_re * b_im + w_im * b_re;

    output[2u * k] = a_re + wb_re;
    output[2u * k + 1u] = a_im + wb_im;
    output[2u * j] = a_re - wb_re;
    output[2u * j + 1u] = a_im - wb_im;
}
`;

const WGSL_BITREVERSE = `
struct Uniforms {
    N: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> uniforms : Uniforms;

// Bit-reversal function
fn bit_rev(x: u32, bits: u32) -> u32 {
    var y = 0u;
    var xv = x;
    for (var b = 0u; b < bits; b = b + 1u) {
        y = (y << 1u) | (xv & 1u);
        xv = xv >> 1u;
    }
    return y;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let N = u32(uniforms.N);
    let bits = u32(log2(f32(N)));
    let i = gid.x;

    if (i >= N) { return; }
    let j = bit_rev(i, bits);

    if (j == i) {
        // Self-reversal: copy as-is
        output[2u * i] = input[2u * i];
        output[2u * i + 1u] = input[2u * i + 1u];
    } else if (j > i) {
        // Swap complex pair (i) with (j)
        let ri = input[2u * i];
        let ii = input[2u * i + 1u];
        let rj = input[2u * j];
        let ij = input[2u * j + 1u];

        output[2u * i] = rj;
        output[2u * i + 1u] = ij;
        output[2u * j] = ri;
        output[2u * j + 1u] = ii;
    }
    // if j < i: do nothing (pair already handled by thread j)
}
`;

const wgsl_stage = WGSL_STAGE;
const wgsl_bitreverse = WGSL_BITREVERSE;

module.exports = { WGSL_STAGE, WGSL_BITREVERSE, wgsl_stage, wgsl_bitreverse };
