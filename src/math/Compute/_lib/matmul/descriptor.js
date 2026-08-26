"use strict";

const TILE = 16;

function describe_matmul(params) {
    const M = params.M || 64;
    const N = params.N || 64;
    const K = params.K || 64;

    function _sizes(input_sizes) {
        return [M * N];
    }

    const dx = Math.ceil(N / TILE);
    const dy = Math.ceil(M / TILE);

    return {
        inputs: [
            { name: "A", dtype: "float" },
            { name: "B", dtype: "float" },
        ],
        outputs: [{ dtype: "float", template: "C" }],
        uniforms: { M: M, N: N, K: K },
        output_size_fn: _sizes,
        dispatch: [dx, dy, 1],
    };
}

module.exports = { describe_matmul };
