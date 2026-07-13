"""MapScalar WGSL — GPU: out[i] = data[i] op scalar.

op codes: 0=add, 1=sub, 2=mul, 3=div, 4=max, 5=min
"""

WGSL = """
struct Params {
    op: f32,
    scalar: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&src);
    if (i >= n) { return; }
    
    let x = src[i];
    let s = params.scalar;
    var result: f32;
    
    // 0=add, 1=sub, 2=mul, 3=div, 4=max, 5=min
    if (params.op == 0.0) {
        result = x + s;
    } else if (params.op == 1.0) {
        result = x - s;
    } else if (params.op == 2.0) {
        result = x * s;
    } else if (params.op == 3.0) {
        if (s != 0.0) { result = x / s; } else { result = 0.0; }
    } else if (params.op == 4.0) {
        result = max(x, s);
    } else if (params.op == 5.0) {
        result = min(x, s);
    } else {
        result = x;
    }
    
    dst[i] = result;
}
"""

__all__ = ["WGSL"]
