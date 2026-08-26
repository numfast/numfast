"use strict";

function cpu(ctx) {
    const src = ctx.inputs[0].view;
    const dst = ctx.outputs[0].view;
    const period = ctx.uniforms.period;
    const n = src.length();

    const limit = Math.min(period - 1, n);
    for (let i = 0; i < limit; i++) {
        dst.write(i, 0.0);
    }

    if (n < period) return;

    const dq = [];
    let head = 0;
    for (let i = 0; i < n; i++) {
        while (head < dq.length && dq[head] <= i - period) {
            head++;
        }

        const val = src.read(i);

        while (dq.length > head && src.read(dq[dq.length - 1]) >= val) {
            dq.pop();
        }

        dq.push(i);

        if (i >= period - 1) {
            dst.write(i, src.read(dq[head]));
        }
    }
}

module.exports = { cpu };
