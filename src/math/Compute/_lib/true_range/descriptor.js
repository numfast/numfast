"use strict";

function describe(params) {
    return {
        inputs: [
            { name: "high", dtype: "float" },
            { name: "low", dtype: "float" },
            { name: "close", dtype: "float" },
        ],
        outputs: [{ dtype: "float", template: "true_range" }],
        workspace: [],
        uniforms: {},
    };
}

module.exports = { describe };
