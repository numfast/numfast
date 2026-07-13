"""Recurrent WGSL — GPU: out[i] = a*in[i] + b*out[i-1].

Mode 0 (single): one input, one workspace, one output
Mode 1 (dual):   two inputs, two workspaces, two outputs

Binding order: inputs -> workspace -> outputs -> uniform.
Workgroup_size(1) because recurrence is inherently sequential.
"""

WGSL = """
struct Params {
    mode: f32,
    a: f32,
    b: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> state: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let mode = i32(params.mode);
    let a = params.a;
    let b = params.b;
    let n = i32(arrayLength(&src));

    if (mode == 0) {
        // Single: out[i] = a*in[i] + b*out[i-1]
        var prev: f32 = 0.0;
        for (var i: i32 = 0; i < n; i = i + 1) {
            let val = a * src[i] + b * prev;
            dst[i] = val;
            state[i] = val;
            prev = val;
        }
    } else if (mode == 1) {
        // Dual: two parallel recurrences, handled in WGSL_DUAL
        var prev: f32 = 0.0;
        for (var i: i32 = 0; i < n; i = i + 1) {
            let val = a * src[i] + b * prev;
            dst[i] = val;
            state[i] = val;
            prev = val;
        }
    }
}
"""

# Dual mode variant: 2 inputs, 2 workspaces, 2 outputs
WGSL_DUAL = """
struct Params {
    mode: f32,
    a: f32,
    b: f32,
};

@group(0) @binding(0) var<storage, read> in0: array<f32>;
@group(0) @binding(1) var<storage, read> in1: array<f32>;
@group(0) @binding(2) var<storage, read_write> st0: array<f32>;
@group(0) @binding(3) var<storage, read_write> st1: array<f32>;
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let a = params.a;
    let b = params.b;
    let n = i32(arrayLength(&in0));

    var prev0: f32 = 0.0;
    var prev1: f32 = 0.0;
    for (var i: i32 = 0; i < n; i = i + 1) {
        let val0 = a * in0[i] + b * prev0;
        let val1 = a * in1[i] + b * prev1;
        out0[i] = val0;
        out1[i] = val1;
        st0[i] = val0;
        st1[i] = val1;
        prev0 = val0;
        prev1 = val1;
    }
}
"""

__all__ = ["WGSL", "WGSL_DUAL"]
