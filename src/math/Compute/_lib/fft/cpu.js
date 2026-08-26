"use strict";

function _bit_rev(x, bits) {
    let y = 0;
    for (let i = 0; i < bits; i++) {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    return y;
}

function cpu_bitreverse(ctx) {
    const src = ctx.inputs[0].view;
    const dst = ctx.outputs[0].view;
    const N = ctx.uniforms.N || 1;
    const bits = Math.round(Math.log2(N));

    for (let i = 0; i < N; i++) {
        const j = _bit_rev(i, bits);
        if (j > i) {
            const ri = src.read(2 * i);
            const ii = src.read(2 * i + 1);
            const rj = src.read(2 * j);
            const ij = src.read(2 * j + 1);
            dst.write(2 * i, rj);
            dst.write(2 * i + 1, ij);
            dst.write(2 * j, ri);
            dst.write(2 * j + 1, ii);
        } else if (j === i) {
            dst.write(2 * i, src.read(2 * i));
            dst.write(2 * i + 1, src.read(2 * i + 1));
        }
    }
}

function cpu_stage(ctx) {
    const src = ctx.inputs[0].view;
    const dst = ctx.outputs[0].view;
    const N = ctx.uniforms.N || 1;
    const stage = ctx.uniforms.stage || 0;

    const stride = 1 << (stage + 1);
    const half = 1 << stage;
    const total_butterflies = N / 2;

    for (let bf = 0; bf < total_butterflies; bf++) {
        const group = Math.floor(bf / half);
        const k_in_group = bf % half;
        const k = group * stride + k_in_group;
        const j = k + half;

        const angle = -2.0 * Math.PI * k_in_group / stride;
        const w_re = Math.cos(angle);
        const w_im = Math.sin(angle);

        const a_re = src.read(2 * k);
        const a_im = src.read(2 * k + 1);
        const b_re = src.read(2 * j);
        const b_im = src.read(2 * j + 1);

        const wb_re = w_re * b_re - w_im * b_im;
        const wb_im = w_re * b_im + w_im * b_re;

        dst.write(2 * k, a_re + wb_re);
        dst.write(2 * k + 1, a_im + wb_im);
        dst.write(2 * j, a_re - wb_re);
        dst.write(2 * j + 1, a_im - wb_im);
    }
}

module.exports = { cpu_stage, cpu_bitreverse };
