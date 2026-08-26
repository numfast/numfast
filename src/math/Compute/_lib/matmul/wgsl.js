"use strict";

const WGSL = `
struct Uniforms {
    M: f32,
    N: f32,
    K: f32,
    _pad: f32,
};

@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;
@group(0) @binding(3) var<uniform> uniforms : Uniforms;

const TILE = 16u;
var<workgroup> shA : array<array<f32, TILE>, TILE>;
var<workgroup> shB : array<array<f32, TILE>, TILE>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(workgroup_id) wgid : vec3<u32>) {
    let M = u32(uniforms.M);
    let N = u32(uniforms.N);
    let K = u32(uniforms.K);
    let tx = lid.x;
    let ty = lid.y;

    // Output tile position in C
    let row = wgid.y * TILE + ty;
    let col = wgid.x * TILE + tx;

    var acc = 0.0;

    // Loop over K in tile steps
    let num_tiles = (K + TILE - 1u) / TILE;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of A into shared memory
        let a_row = row;
        let a_col = t * TILE + tx;
        if (a_row < M && a_col < K) {
            shA[ty][tx] = A[a_row * K + a_col];
        } else {
            shA[ty][tx] = 0.0;
        }

        // Load tile of B into shared memory
        let b_row = t * TILE + ty;
        let b_col = col;
        if (b_row < K && b_col < N) {
            shB[ty][tx] = B[b_row * N + b_col];
        } else {
            shB[ty][tx] = 0.0;
        }

        workgroupBarrier();

        // Compute partial product
        for (var k = 0u; k < TILE; k = k + 1u) {
            acc = acc + shA[ty][k] * shB[k][tx];
        }

        workgroupBarrier();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
`;

const wgsl_matmul = WGSL;

module.exports = { WGSL, wgsl_matmul };
