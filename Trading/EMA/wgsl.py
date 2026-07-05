"""EMA WGSL — GPU implementation.

Serial approach: workgroup_size(1), one thread computes all elements.
EMA is inherently sequential (each element depends on previous).
Future optimization: parallel segmented scan for large arrays.

Multiplier is computed from period inside the shader,
so only period needs to be passed as a uniform.
"""

WGSL = """
struct Params {
    period: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = arrayLength(&src);
    let multiplier = 2.0 / (params.period + 1.0);
    dst[0] = src[0];
    var i = 1u;
    while (i < n) {
        let prev = dst[i - 1u];
        dst[i] = (src[i] - prev) * multiplier + prev;
        i = i + 1u;
    }
}
"""

__all__ = ["WGSL"]
