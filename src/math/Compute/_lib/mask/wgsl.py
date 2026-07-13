"""Mask WGSL — GPU: out[i] = select(b[i], a[i], cond[i] != 0u)."""

WGSL = """
@group(0) @binding(0) var<storage, read> cond: array<u32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&dst);
    if (i >= n) { return; }
    
    // select(false_case, true_case, condition)
    dst[i] = select(b[i], a[i], cond[i] != 0u);
}
"""

__all__ = ["WGSL"]