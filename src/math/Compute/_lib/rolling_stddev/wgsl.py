"""Rolling StdDev WGSL -- parallel O(period) per element.

Each thread computes one output element with a window sum
and sum of squares.
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
    let pf = f32(period);

    if (i >= n) {
        return;
    }

    if (i < period - 1) {
        dst[i] = 0.0;
        return;
    }

    var sum: f32 = 0.0;
    var sum_sq: f32 = 0.0;
    for (var j: i32 = i - period + 1; j <= i; j = j + 1) {
        let x = src[j];
        sum = sum + x;
        sum_sq = sum_sq + x * x;
    }
    let mean = sum / pf;
    let variance = sum_sq / pf - mean * mean;
    if (variance > 0.0) {
        dst[i] = sqrt(variance);
    } else {
        dst[i] = 0.0;
    }
}
"""

__all__ = ["WGSL"]
