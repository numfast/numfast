"use strict";

function describe(params) {
    const op = params.op || "min";
    return {
        inputs: [
            { name: "src", dtype: "float" },
        ],
        outputs: [
            { dtype: "float", template: "arg_val" },
            { dtype: "float", template: "arg_idx" },
        ],
        workspace: [],
        uniforms: { op: op },
    };
}

module.exports = { describe };
