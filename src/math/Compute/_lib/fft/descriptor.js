"use strict";

function describe_stage(params) {
    const N = params.N || 1024;

    function _sizes(input_sizes) {
        return [2 * N];
    }

    return {
        inputs: [{ name: "complex", dtype: "float" }],
        outputs: [{ dtype: "float", template: "complex" }],
        uniforms: { stage: params.stage || 0, N: N },
        output_size_fn: _sizes,
    };
}

function describe_bitreverse(params) {
    const N = params.N || 1024;

    function _sizes(input_sizes) {
        return [2 * N];
    }

    return {
        inputs: [{ name: "complex", dtype: "float" }],
        outputs: [{ dtype: "float", template: "complex" }],
        uniforms: { N: N },
        output_size_fn: _sizes,
    };
}

module.exports = { describe_stage, describe_bitreverse };
