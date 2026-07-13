"""ArgMin/ArgMax WGSL — GPU: single-pass workgroup reduction.

For each output element, scans a chunk of input and finds
the min/max value and its index.

Thread i reads input[i], compares with workgroup-shared memory.
"""

WGSL = """
struct Params {
    _pad: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_val: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_idx: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> wg_val: array<f32, 64>;
var<workgroup> wg_idx: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(workgroup_id) wgid: vec3<u32>) {
    let out_n = arrayLength(&out_val);
    if (wgid.x >= out_n) { return; }
    
    let src_n = arrayLength(&src);
    let chunk = (src_n + out_n - 1u) / out_n;
    let start = wgid.x * chunk;
    let end = min(start + chunk, src_n);
    
    // Each thread loads one element
    let i = start + lid;
    if (i < end) {
        wg_val[lid] = src[i];
        wg_idx[lid] = f32(i);
    } else {
        wg_val[lid] = 1e30;  // neutral for min
        wg_idx[lid] = f32(i);
    }
"""

# Op-specific WGSL
OPS = {
    "min": """
    // Workgroup reduction (min)
    var active = 64u;
    var step = 32u;
    while (step > 0u) {
        workgroupBarrier();
        if (lid < step) {
            if (wg_val[lid + step] < wg_val[lid]) {
                wg_val[lid] = wg_val[lid + step];
                wg_idx[lid] = wg_idx[lid + step];
            }
        }
        step = step / 2u;
    }
    workgroupBarrier();
    if (lid == 0u) {
        out_val[wgid.x] = wg_val[0];
        out_idx[wgid.x] = wg_idx[0];
    }
""",
    "max": """
    // Workgroup reduction (max)
    var active = 64u;
    var step = 32u;
    while (step > 0u) {
        workgroupBarrier();
        if (lid < step) {
            if (wg_val[lid + step] > wg_val[lid]) {
                wg_val[lid] = wg_val[lid + step];
                wg_idx[lid] = wg_idx[lid + step];
            }
        }
        step = step / 2u;
    }
    workgroupBarrier();
    if (lid == 0u) {
        out_val[wgid.x] = wg_val[0];
        out_idx[wgid.x] = wg_idx[0];
    }
""",
}


def wgsl_generator(uniforms: dict) -> str:
    op = uniforms.get("op", "min")
    body = OPS.get(op, OPS["min"])
    return WGSL + body + "\n}"


__all__ = ["WGSL", "wgsl_generator"]
