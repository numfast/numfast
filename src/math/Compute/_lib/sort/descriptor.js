"use strict";

function describe_sort(params) {
    const stage = params.stage !== undefined ? params.stage : 1;
    const substage = params.substage !== undefined ? params.substage : 1;

    function sizes(input_sizes) {
        return [input_sizes[0]];
    }

    return {
        inputs: [{ name: "data", dtype: "float" }],
        outputs: [{ dtype: "float", template: "data" }],
        uniforms: {
            stage: stage,
            substage: substage,
        },
        output_size_fn: sizes,
    };
}

module.exports = { describe_sort };
