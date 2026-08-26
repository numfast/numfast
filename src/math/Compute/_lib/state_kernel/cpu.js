"use strict";

function cpu(ctx) {
    const mode = parseInt(ctx.uniforms.mode || 0, 10);

    if (mode === 0) {
        const src = ctx.inputs[0].view;
        const dst = ctx.outputs[0].view;
        const state = ctx.workspace[0].view;
        const a = ctx.uniforms.a || 1.0;
        const b = ctx.uniforms.b || 0.0;
        const n = src.length();
        if (n === 0) { return; }
        let prev = src.read(0);
        dst.write(0, prev);
        state.write(0, prev);
        for (let i = 1; i < n; i++) {
            const val = a * src.read(i) + b * prev;
            dst.write(i, val);
            state.write(i, val);
            prev = val;
        }
    } else if (mode === 1) {
        const in0 = ctx.inputs[0].view;
        const in1 = ctx.inputs[1].view;
        const out0 = ctx.outputs[0].view;
        const out1 = ctx.outputs[1].view;
        const st0 = ctx.workspace[0].view;
        const st1 = ctx.workspace[1].view;
        const a = ctx.uniforms.a || 1.0;
        const b = ctx.uniforms.b || 0.0;
        const n = in0.length();
        let p0 = 0.0;
        let p1 = 0.0;
        for (let i = 0; i < n; i++) {
            const v0 = a * in0.read(i) + b * p0;
            const v1 = a * in1.read(i) + b * p1;
            out0.write(i, v0);
            out1.write(i, v1);
            st0.write(i, v0);
            st1.write(i, v1);
            p0 = v0;
            p1 = v1;
        }
    } else if (mode === 2) {
        const close = ctx.inputs[0].view;
        const upper_band = ctx.inputs[1].view;
        const lower_band = ctx.inputs[2].view;
        const dst = ctx.outputs[0].view;
        const st_upper = ctx.workspace[0].view;
        const st_lower = ctx.workspace[1].view;
        const st_dir = ctx.workspace[2].view;
        const n = close.length();

        let prev_upper = 0.0;
        let prev_lower = 0.0;
        let prev_dir = 1.0;

        for (let i = 0; i < n; i++) {
            let direction;
            let final_upper;
            let final_lower;

            if (i === 0) {
                direction = 1.0;
                final_upper = upper_band.read(i);
                final_lower = lower_band.read(i);
            } else {
                if (prev_dir === 1.0) {
                    if (close.read(i) < prev_lower) {
                        direction = -1.0;
                    } else {
                        direction = 1.0;
                    }
                } else {
                    if (close.read(i) > prev_upper) {
                        direction = 1.0;
                    } else {
                        direction = -1.0;
                    }
                }

                if (direction === 1.0) {
                    final_upper = Math.max(upper_band.read(i), prev_upper);
                    final_lower = lower_band.read(i);
                } else {
                    final_upper = upper_band.read(i);
                    final_lower = Math.min(lower_band.read(i), prev_lower);
                }
            }

            dst.write(i, direction);
            st_upper.write(i, final_upper);
            st_lower.write(i, final_lower);
            st_dir.write(i, direction);
            prev_upper = final_upper;
            prev_lower = final_lower;
            prev_dir = direction;
        }
    }
}

module.exports = { cpu };
