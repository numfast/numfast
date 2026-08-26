"""Map WGSL — element-wise functions on GPU.

Selects built-in function via uniform parameter.
"""

WGSL = """
struct Params {
    func: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&src)) { return; }
    
    let x = src[i];
    // S39: NaN trap (defense-in-depth). The func code is validated in
    // descriptor.describe() (single point, both backends, before dispatch),
    // so the else->square default is unreachable. If a garbage func somehow
    // reaches the shader, the result is NaN, NOT square. The trap must be a
    // runtime expression (x * 0.0 / 0.0): a const-expression NaN literal is
    // rejected by WGSL validation ("Float literal is NaN").
    var result: f32 = x * 0.0 / 0.0;
    
    // 0=sin, 1=cos, 2=exp, 3=sqrt, 4=log, 5=abs, 6=neg, 7=square
    if (params.func == 0) {
        result = sin(x);
    } else if (params.func == 1) {
        result = cos(x);
    } else if (params.func == 2) {
        result = exp(x);
    } else if (params.func == 3) {
        result = sqrt(x);
    } else if (params.func == 4) {
        result = log(x);
    } else if (params.func == 5) {
        result = abs(x);
    } else if (params.func == 6) {
        result = -x;
    } else if (params.func == 7) {
        result = x * x;
    }
    
    dst[i] = result;
}
"""

__all__ = ["WGSL"]
