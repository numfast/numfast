"""Reduce WGSL — сумма элементов на GPU.

Каждый поток суммирует свой chunk, затем первый поток сводит partials.
"""

WGSL = """
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>,
         @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = arrayLength(&src);
    
    var sum = 0.0;
    var i = id.x;
    while (i < n) {
        sum = sum + src[i];
        i = i + 256u;
    }
    
    partials[id.x] = sum;
    
    if (id.x == 0u) {
        var total = 0.0;
        for (var j = 0u; j < 256u; j = j + 1u) {
            total = total + partials[j];
        }
        dst[0] = total;
    }
}
"""

__all__ = ["WGSL"]
