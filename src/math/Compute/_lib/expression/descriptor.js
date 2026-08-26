"use strict";

function describe(params) {
    const formula = params.formula || "A";
    const vars = [...new Set(formula.split("").filter(c => c >= "A" && c <= "Z"))].sort();
    const num_vars = vars.length;
    const inputs = vars.map(v => ({ name: v, dtype: "float" }));
    return {
        inputs: inputs,
        outputs: [{ dtype: "float", template: "expr" }],
        workspace: [],
        uniforms: {
            formula: formula,
            num_vars: num_vars,
        },
    };
}

module.exports = { describe };
