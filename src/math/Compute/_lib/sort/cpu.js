"use strict";

function cpu_sort(ctx) {
    const src = ctx.inputs[0].view;
    const dst = ctx.outputs[0].view;
    const n = src.length();
    const stage = ctx.uniforms.stage !== undefined ? ctx.uniforms.stage : 1;
    const substage = ctx.uniforms.substage !== undefined ? ctx.uniforms.substage : 1;

    for (let i = 0; i < n; i++) {
        dst.write(i, src.read(i));
    }

    const dist = 1 << (substage - 1);
    const mask = 1 << stage;

    for (let i = 0; i < n; i++) {
        const j = i ^ dist;
        if (j > i && j < n) {
            const a = dst.read(i);
            const b = dst.read(j);
            const ascending = (i & mask) === 0;
            if (ascending) {
                if (a > b) {
                    dst.write(i, b);
                    dst.write(j, a);
                }
            } else {
                if (a < b) {
                    dst.write(i, b);
                    dst.write(j, a);
                }
            }
        }
    }
}

module.exports = { cpu_sort };
