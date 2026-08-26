"use strict";

function cpu(ctx) {
    const op = ctx.uniforms.op || "min";
    const src = ctx.inputs[0].view;
    const val_out = ctx.outputs[0].view;
    const idx_out = ctx.outputs[1].view;
    const n = src.length();
    const out_n = val_out.length();

    if (n === 0 || out_n === 0) { return; }

    const chunk = Math.max(1, Math.floor(n / out_n));
    for (let j = 0; j < out_n; j++) {
        const start = j * chunk;
        const end = Math.min(start + chunk, n);
        if (start >= n) {
            val_out.write(j, 0.0);
            idx_out.write(j, 0.0);
            continue;
        }

        let best_val = src.read(start);
        let best_idx = start;

        for (let i = start + 1; i < end; i++) {
            const v = src.read(i);
            if ((op === "min" && v < best_val) || (op === "max" && v > best_val)) {
                best_val = v;
                best_idx = i;
            }
        }

        val_out.write(j, best_val);
        idx_out.write(j, best_idx);
    }
}

module.exports = { cpu };
