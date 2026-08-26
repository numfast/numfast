"use strict";

const WGSL = `
@group(0) @binding(0) var<storage, read> high: array<f32>;
@group(0) @binding(1) var<storage, read> low: array<f32>;
@group(0) @binding(2) var<storage, read> close: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = i32(arrayLength(&high));

    var prev_close = close[0];
    for (var i: i32 = 0; i < n; i = i + 1) {
        let hl = high[i] - low[i];
        let hc = abs(high[i] - prev_close);
        let lc = abs(low[i] - prev_close);

        var tr = hl;
        if (hc > tr) { tr = hc; }
        if (lc > tr) { tr = lc; }

        dst[i] = tr;
        prev_close = close[i];
    }
}
`;

module.exports = { WGSL };
