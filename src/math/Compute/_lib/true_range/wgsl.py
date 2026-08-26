"""TrueRange WGSL -- GPU, parallel.

TR = max(high - low, |high - prev_close|, |low - prev_close|)
Each thread computes one output element.
"""

WGSL = """
@group(0) @binding(0) var<storage, read> high: array<f32>;
@group(0) @binding(1) var<storage, read> low: array<f32>;
@group(0) @binding(2) var<storage, read> close: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = i32(id.x);
    let n = i32(arrayLength(&high));

    if (i >= n) {
        return;
    }

    let prev_close = close[max(0, i - 1)];
    let hl = high[i] - low[i];
    let hc = abs(high[i] - prev_close);
    let lc = abs(low[i] - prev_close);

    var tr = hl;
    if (hc > tr) { tr = hc; }
    if (lc > tr) { tr = lc; }

    dst[i] = tr;
}
"""

__all__ = ["WGSL"]