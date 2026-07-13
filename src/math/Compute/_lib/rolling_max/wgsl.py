"""Rolling Max WGSL -- parallel O(period) per element.

Each thread computes one output element.
"""

WGSL = """
struct Params {
    period: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = i32(id.x);
    let n = i32(arrayLength(&src));
    let period = i32(params.period);

    if (i >= n) {
        return;
    }

    if (i < period - 1) {
        dst[i] = 0.0;
        return;
    }

    var max_val = src[i - period + 1];
    for (var j: i32 = i - period + 2; j <= i; j = j + 1) {
        let v = src[j];
        if (v > max_val) {
            max_val = v;
        }
    }
    dst[i] = max_val;
}
"""

__all__ = ["WGSL"]
