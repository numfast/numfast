"use strict";

function cpu_matmul(ctx) {
    const A = ctx.inputs[0].view;
    const B = ctx.inputs[1].view;
    const C = ctx.outputs[0].view;
    const M = ctx.uniforms.M || 1;
    const N = ctx.uniforms.N || 1;
    const K = ctx.uniforms.K || 1;

    for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
            let acc = 0.0;
            for (let k = 0; k < K; k++) {
                acc += A.read(i * K + k) * B.read(k * N + j);
            }
            C.write(i * N + j, acc);
        }
    }
}

module.exports = { cpu_matmul };
