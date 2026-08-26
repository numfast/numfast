"use strict";

function cpu(ctx) {
    const src = ctx.inputs[0].view;
    const dst = ctx.outputs[0].view;
    const period = ctx.uniforms.period;
    const n = src.length();

    let running_sum = 0.0;
    let running_sum_sq = 0.0;
    for (let i = 0; i < n; i++) {
        const x = src.read(i);
        running_sum += x;
        running_sum_sq += x * x;
        if (i >= period) {
            const old = src.read(i - period);
            running_sum -= old;
            running_sum_sq -= old * old;
        }
        if (i >= period - 1) {
            const mean = running_sum / period;
            const variance = running_sum_sq / period - mean * mean;
            if (variance > 0) {
                dst.write(i, Math.sqrt(variance));
            } else {
                dst.write(i, 0.0);
            }
        } else {
            dst.write(i, 0.0);
        }
    }
}

module.exports = { cpu };
