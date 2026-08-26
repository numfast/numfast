"use strict";

function cpu(ctx) {
    const src = ctx.inputs[0].view;
    const dst = ctx.outputs[0].view;
    const period = ctx.uniforms.period;
    const n = src.length();

    let running_sum = 0.0;
    for (let i = 0; i < n; i++) {
        running_sum += src.read(i);
        if (i >= period) {
            running_sum -= src.read(i - period);
        }
        if (i >= period - 1) {
            dst.write(i, running_sum);
        } else {
            dst.write(i, 0.0);
        }
    }
}

module.exports = { cpu };
