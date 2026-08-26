"use strict";

function describe(params) {
    return {
        inputs: [{ name: "data", dtype: "float" }],
        outputs: [{ dtype: "float", template: "rolling_max_" + params.period }],
        workspace: [],
        uniforms: { period: params.period },
    };
}

module.exports = { describe };
