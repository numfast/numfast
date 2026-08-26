"use strict";

function describe(params) {
    const num_bins = params.num_bins !== undefined ? params.num_bins : 10;
    const min_val = params.min_val !== undefined ? params.min_val : 0.0;
    const max_val = params.max_val !== undefined ? params.max_val : 1.0;

    function sizes(input_sizes) {
        return [num_bins];
    }

    return {
        inputs: [{ name: "data", dtype: "float" }],
        outputs: [{ dtype: "float", template: "bins" }],
        uniforms: {
            min_val: min_val,
            max_val: max_val,
            num_bins: num_bins,
        },
        output_size_fn: sizes,
    };
}

module.exports = { describe };
