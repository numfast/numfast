"use strict";

function describe(params) {
    const mode = parseInt(params.mode || 0, 10);
    const params_out = Object.assign({}, params);
    params_out.mode = mode;

    if (mode === 0) {
        return {
            inputs: [{ name: "data", dtype: "float" }],
            outputs: [{ dtype: "float", template: "ema" }],
            workspace: [{ dtype: "float", elements: 0 }],
            uniforms: params_out,
        };
    } else if (mode === 1) {
        return {
            inputs: [
                { name: "gain", dtype: "float" },
                { name: "loss", dtype: "float" },
            ],
            outputs: [
                { dtype: "float", template: "ema_gain" },
                { dtype: "float", template: "ema_loss" },
            ],
            workspace: [
                { dtype: "float", elements: 0 },
                { dtype: "float", elements: 0 },
            ],
            uniforms: params_out,
        };
    } else if (mode === 2) {
        return {
            inputs: [
                { name: "close", dtype: "float" },
                { name: "upper_band", dtype: "float" },
                { name: "lower_band", dtype: "float" },
            ],
            outputs: [{ dtype: "float", template: "supertrend" }],
            workspace: [
                { dtype: "float", elements: 0 },
                { dtype: "float", elements: 0 },
                { dtype: "float", elements: 0 },
            ],
            uniforms: params_out,
        };
    } else {
        throw new Error("Unknown StateKernel mode: " + mode);
    }
}

module.exports = { describe };
