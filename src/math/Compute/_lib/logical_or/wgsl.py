"""LogicalOr WGSL -- GPU: out[i] = a[i] | b[i] (uint32)."""

WGSL = """
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&dst);
    if (i >= n) { return; }
    dst[i] = a[i] | b[i];
}
"""

__all__ = ["WGSL"]
