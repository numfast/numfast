"""StateKernel WGSL — GPU implementations for all modes.

Binding order: inputs -> workspace -> outputs -> uniform.
All modes use workgroup_size(1) due to sequential nature.
"""

WGSL_SINGLE = """
struct Params {
    mode: f32,
    a: f32,
    b: f32,
    _pad: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> state: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = i32(arrayLength(&src));
    let a = params.a;
    let b = params.b;
    var prev: f32 = 0.0;
    for (var i: i32 = 0; i < n; i = i + 1) {
        let val = a * src[i] + b * prev;
        dst[i] = val;
        state[i] = val;
        prev = val;
    }
}
"""

WGSL_DUAL = """
struct Params {
    mode: f32,
    a: f32,
    b: f32,
    _pad: f32,
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
    let n = i32(arrayLength(&in0));
    let a = params.a;
    let b = params.b;
    var p0: f32 = 0.0;
    var p1: f32 = 0.0;
    for (var i: i32 = 0; i < n; i = i + 1) {
        let v0 = a * in0[i] + b * p0;
        let v1 = a * in1[i] + b * p1;
        out0[i] = v0;
        out1[i] = v1;
        st0[i] = v0;
        st1[i] = v1;
        p0 = v0;
        p1 = v1;
    }
}
"""

WGSL_SUPERTREND = """
struct Params {
    mode: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> close: array<f32>;
@group(0) @binding(1) var<storage, read> upper_band: array<f32>;
@group(0) @binding(2) var<storage, read> lower_band: array<f32>;
@group(0) @binding(3) var<storage, read_write> st_upper: array<f32>;
@group(0) @binding(4) var<storage, read_write> st_lower: array<f32>;
@group(0) @binding(5) var<storage, read_write> st_dir: array<f32>;
@group(0) @binding(6) var<storage, read_write> dst: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = i32(arrayLength(&close));

    var pu: f32 = 0.0;  // prev_upper
    var pl: f32 = 0.0;  // prev_lower
    var pd: f32 = 1.0;  // prev_dir

    for (var i: i32 = 0; i < n; i = i + 1) {
        var direction: f32;
        var final_upper: f32;
        var final_lower: f32;

        if (i == 0) {
            direction = 1.0;
            final_upper = upper_band[i];
            final_lower = lower_band[i];
        } else {
            if (pd == 1.0) {
                if (close[i] < pl) {
                    direction = -1.0;
                } else {
                    direction = 1.0;
                }
            } else {
                if (close[i] > pu) {
                    direction = 1.0;
                } else {
                    direction = -1.0;
                }
            }

            if (direction == 1.0) {
                final_upper = max(upper_band[i], pu);
                final_lower = lower_band[i];
            } else {
                final_upper = upper_band[i];
                final_lower = min(lower_band[i], pl);
            }
        }

        dst[i] = direction;
        st_upper[i] = final_upper;
        st_lower[i] = final_lower;
        st_dir[i] = direction;

        pu = final_upper;
        pl = final_lower;
        pd = direction;
    }
}
"""

__all__ = ["WGSL_SINGLE", "WGSL_DUAL", "WGSL_SUPERTREND"]
