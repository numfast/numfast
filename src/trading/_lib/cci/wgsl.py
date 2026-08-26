"""CCI WGSL — GPU implementation.

TP = (High + Low + Close) / 3
CCI = (TP - SMA(TP, period)) / (0.015 * MeanDeviation)

Single-pass: workgroup_size(1), one thread computes all elements.
"""

WGSL = """
struct Params {
    period: f32,
};

@group(0) @binding(0) var<storage, read> high: array<f32>;
@group(0) @binding(1) var<storage, read> low: array<f32>;
@group(0) @binding(2) var<storage, read> close: array<f32>;
@group(0) @binding(3) var<storage, read_write> tp_buf: array<f32>;
@group(0) @binding(4) var<storage, read_write> sma_buf: array<f32>;
@group(0) @binding(5) var<storage, read_write> cci: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = i32(arrayLength(&high));
    let period = i32(params.period);

    for (var i: i32 = 0; i < n; i = i + 1) {
        let tp = (high[i] + low[i] + close[i]) / 3.0;
        tp_buf[i] = tp;
    }

    for (var i: i32 = 0; i < n; i = i + 1) {
        if (i < period - 1) {
            cci[i] = 0.0;
            continue;
        }

        // Compute SMA of TP over window [i-period+1, i]
        var sma_sum: f32 = 0.0;
        for (var j: i32 = 0; j < period; j = j + 1) {
            sma_sum = sma_sum + tp_buf[i - j];
        }
        let sma = sma_sum / f32(period);

        // Compute Mean Deviation
        var md_sum: f32 = 0.0;
        for (var j: i32 = 0; j < period; j = j + 1) {
            md_sum = md_sum + abs(tp_buf[i - j] - sma);
        }
        let md = md_sum / f32(period);

        if (md > 0.0) {
            cci[i] = (tp_buf[i] - sma) / (0.015 * md);
        } else {
            cci[i] = 0.0;
        }
    }
}
"""

__all__ = ["WGSL"]