"use strict";

function cpu(ctx) {
    const high = ctx.inputs[0].view;
    const low = ctx.inputs[1].view;
    const close = ctx.inputs[2].view;
    const dst = ctx.outputs[0].view;
    const n = high.length();

    let prev_close = close.read(0);
    for (let i = 0; i < n; i++) {
        if (i === 0) {
            dst.write(i, 0.0);
        } else {
            const hl = high.read(i) - low.read(i);
            const hc = Math.abs(high.read(i) - prev_close);
            const lc = Math.abs(low.read(i) - prev_close);
            const tr = Math.max(hl, hc, lc);
            dst.write(i, tr);
        }
        prev_close = close.read(i);
    }
}

module.exports = { cpu };
