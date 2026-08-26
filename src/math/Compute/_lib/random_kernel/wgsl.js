"use strict";

// RandomKernel WGSL — byte-identical copy of wgsl.py WGSL string.
// SHA256 py-vs-js parity is checked by parity_test.js (CHECK 3).

const WGSL = `
struct Params {
    n: u32,
    mode: u32,
    seed: u32,
    p0: f32,
    p1: f32,
};

@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

const C1: u32 = 747796405u;
const C2: u32 = 2891336453u;
const C3: u32 = 277803737u;
const TWO_POW_NEG32: f32 = 2.3283064365386963e-10;       // 2^-32
const TWO_POW_NEG33: f32 = 1.16415321826934814453125e-10; // 2^-33, log(0) guard
const TWO_PI: f32 = 6.2831853071795864769;

fn pcg_hash(state_in: u32) -> u32 {
    var state: u32 = state_in;
    state = (state >> ((state >> 28u) + 4u)) ^ (state * C3);
    state = (state >> ((state >> 28u) + 4u)) ^ (state * C3);
    state = (state >> ((state >> 28u) + 4u)) ^ (state * C3);
    return state;
}

fn hash_u01(i: u32, seed: u32) -> f32 {
    let state: u32 = seed ^ (i * C1 + C2);
    let h: u32 = pcg_hash(state);
    return f32(h) * TWO_POW_NEG32;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.n) { return; }

    var v: f32;
    if (params.mode == 0u) {
        // uniform(low = p0, high = p1)
        let u = hash_u01(i, params.seed);
        v = params.p0 + (params.p1 - params.p0) * u;
    } else if (params.mode == 1u) {
        // int_range [p0, p1): floor(p0 + (p1-p0)*u01), clamp в диапазон
        let u = hash_u01(i, params.seed);
        let raw = floor(params.p0 + (params.p1 - params.p0) * u);
        v = clamp(raw, params.p0, max(params.p0, params.p1 - 1.0));
    } else {
        // normal(loc = p0, scale = p1): Box-Muller из двух хешей (i, i+n)
        let u1_raw = hash_u01(i, params.seed);
        let u2 = hash_u01(i + params.n, params.seed);
        let u1 = select(u1_raw, TWO_POW_NEG33, u1_raw <= 0.0); // log(0) guard
        let r = sqrt(-2.0 * log(u1));
        let theta = TWO_PI * u2;
        let z = r * cos(theta);
        v = params.p0 + params.p1 * z;
    }
    dst[i] = v;
}
`;

if (typeof module !== "undefined" && module.exports) { module.exports = { WGSL: WGSL }; }
