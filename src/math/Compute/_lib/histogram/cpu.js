"use strict";

function cpu_histogram(ctx) {
    const src = ctx.inputs[0].view;
    const dst = ctx.outputs[0].view;
    const n = src.length();
    const min_val = ctx.uniforms.min_val !== undefined ? ctx.uniforms.min_val : 0.0;
    const max_val = ctx.uniforms.max_val !== undefined ? ctx.uniforms.max_val : 1.0;
    const num_bins = ctx.uniforms.num_bins !== undefined ? ctx.uniforms.num_bins : 10;

    for (let b = 0; b < num_bins; b++) {
        dst.write(b, 0.0);
    }

    for (let i = 0; i < n; i++) {
        const val = src.read(i);
        const clamped = Math.max(min_val, Math.min(val, max_val));
        let bin_idx;
        if (max_val > min_val) {
            const norm = (clamped - min_val) / (max_val - min_val);
            bin_idx = Math.floor(norm * num_bins);
            bin_idx = Math.min(bin_idx, num_bins - 1);
        } else {
            bin_idx = 0;
        }
        dst.write(bin_idx, dst.read(bin_idx) + 1.0);
    }
}

module.exports = { cpu_histogram };
