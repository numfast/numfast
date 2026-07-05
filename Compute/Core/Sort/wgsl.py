"""Bitonic Sort WGSL — one substage pass per dispatch."""

WGSL = """
struct Uniforms {
    stage: i32,
    substage: i32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> uniforms : Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&input);

    if (i >= n) { return; }

    let substage = u32(uniforms.substage);
    let stage = u32(uniforms.stage);
    let dist = 1u << (substage - 1u);
    let mask = 1u << stage;

    let j = i ^ dist;

    if (j > i && j < n) {
        let a = input[i];
        let b = input[j];
        let ascending = (i & mask) == 0u;

        if (ascending) {
            if (a > b) {
                output[i] = b;
                output[j] = a;
            } else {
                output[i] = a;
                output[j] = b;
            }
        } else {
            if (a < b) {
                output[i] = b;
                output[j] = a;
            } else {
                output[i] = a;
                output[j] = b;
            }
        }
    }
}
"""

wgsl_sort = WGSL
