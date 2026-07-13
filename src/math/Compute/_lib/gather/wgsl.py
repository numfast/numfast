"""Gather WGSL — GPU: out[i] = src[index[i]]."""

WGSL = """
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> index: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&dst);
    if (i >= n) { return; }
    
    let ix = i32(index[i]);
    let src_len = i32(arrayLength(&src));
    if (ix >= 0 && ix < src_len) {
        dst[i] = src[ix];
    } else {
        dst[i] = 0.0;
    }
}
"""

__all__ = ["WGSL"]