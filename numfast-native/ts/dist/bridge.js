// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// TS-bridge over the raw cdylib ABI of numfast-native (no wasm-bindgen).
// ESM-only, zero-copy TypedArray views (subarray, never slice).
// Runs in Node >=22.18 (native type-stripping) and browsers (via tsc build).
// Decision: raw ABI — wasm-bindgen rejected (zero deps, usize->i32 ABI stable).
export const STACK_TOP = 0x100000;
export const GUARD = 0x1000;
export const CANARY = 0x9e3779b9;
export const BASE = STACK_TOP + GUARD;
const PAGE = 65536;
const INF = 0xffffffff;
const alignUp = (off, a) => (off + a - 1) & ~(a - 1);
export async function loadBridge(wasmBytes) {
    const bytes = wasmBytes instanceof Uint8Array ? wasmBytes : new Uint8Array(wasmBytes);
    const { instance } = await WebAssembly.instantiate(bytes, {});
    return new Bridge(instance.exports);
}
function checkRc(rc, op) {
    if (rc !== 0)
        throw new Error(`${op}: rc=${rc}`);
}
export class Bridge {
    ex;
    cur;
    constructor(ex) {
        this.ex = ex;
        this.cur = BASE;
        this.fillGuard();
    }
    get mem() { return this.ex.memory; }
    get u8() { return new Uint8Array(this.mem.buffer); }
    fillGuard() {
        this.ensure(this.cur);
        new Uint32Array(this.mem.buffer, STACK_TOP, GUARD / 4).fill(CANARY);
    }
    assertGuard() {
        const g = new Uint32Array(this.mem.buffer, STACK_TOP, GUARD / 4);
        for (let i = 0; i < g.length; i++) {
            if (g[i] !== CANARY)
                throw new Error("bridge: wasm stack drifted into buffers");
        }
    }
    ensure(need) {
        const have = this.mem.buffer.byteLength;
        if (need > have)
            this.mem.grow(Math.ceil((need - have) / PAGE));
        return new Uint8Array(this.mem.buffer);
    }
    alloc(bytes, align = 8) {
        const off = alignUp(this.cur, align);
        this.cur = off + bytes;
        return off;
    }
    reset() { this.cur = BASE; }
    // Zero-copy views (subarray) — invalidated by grow; re-fetch via u8 after ensure.
    u32(off, n) { return new Uint32Array(this.mem.buffer, off, n); }
    i32(off, n) { return new Int32Array(this.mem.buffer, off, n); }
    u16(off, n) { return new Uint16Array(this.mem.buffer, off, n); }
    f32(off, n) { return new Float32Array(this.mem.buffer, off, n); }
    u8v(off, n) { return new Uint8Array(this.mem.buffer, off, n); }
    put(v) {
        const off = this.alloc(v.byteLength, 8);
        const u = this.ensure(this.cur);
        u.set(new Uint8Array(v.buffer, v.byteOffset, v.byteLength), off);
        return off;
    }
    // ---- sssp: CSR + weights + source -> dist (u32, INF=unreachable) ----
    ssspCsr(indptr, indices, weights, source) {
        const v = indptr.length - 1;
        this.reset();
        const pInd = this.put(indptr), pIx = this.put(indices), pW = this.put(weights);
        const pDist = this.alloc(v * 4, 8);
        this.ensure(this.cur);
        checkRc(this.ex.nf_sssp_csr(pInd, indptr.length, pIx, pW, indices.length, source, pDist), "ssspCsr");
        this.assertGuard();
        return this.u32(pDist, v).slice();
    }
    // ---- batch: sources[K] -> out[K*V] row-major (wasm: nthreads forced 1) ----
    ssspBatch(indptr, indices, weights, sources) {
        const v = indptr.length - 1, k = sources.length;
        this.reset();
        const pInd = this.put(indptr), pIx = this.put(indices), pW = this.put(weights);
        const pSrc = this.put(sources);
        const pOut = this.alloc(k * v * 4, 8);
        this.ensure(this.cur);
        checkRc(this.ex.nf_sssp_batch(pInd, indptr.length, pIx, pW, indices.length, pSrc, k, pOut, 1), "ssspBatch");
        this.assertGuard();
        return this.u32(pOut, k * v).slice();
    }
    // ---- Q-lookup: out[i] = (dist[i]*k[i] + speed[i]/2)/speed[i], INF-guarded ----
    costTravel(dist, speed, k) {
        const n = dist.length;
        this.reset();
        const pD = this.put(dist), pS = this.put(speed), pK = this.put(k);
        const pOut = this.alloc(n * 4, 8);
        this.ensure(this.cur);
        checkRc(this.ex.nf_cost_travel_batch(pD, pS, pK, n, pOut), "costTravel");
        this.assertGuard();
        return this.u32(pOut, n).slice();
    }
    costTravelInto(dOff, sOff, kOff, n, outOff) {
        return this.ex.nf_cost_travel_batch(dOff, sOff, kOff, n, outOff);
    }
    // ---- cost interning: row-major vecs[n*width] -> ids + uniq table (returns ng) ----
    costIntern(vecs, n, width) {
        this.reset();
        const pV = this.put(vecs);
        const pIds = this.alloc(n * 4, 8);
        const pUniq = this.alloc(n * width * 4, 8);
        this.ensure(this.cur);
        const ng = Number(this.ex.nf_cost_intern(pV, n, width, pIds, pUniq, n * width));
        if (ng < 0)
            throw new Error(`costIntern: ng=${ng}`);
        this.assertGuard();
        return { ng, ids: this.u32(pIds, n).slice(), uniq: this.u32(pUniq, ng * width).slice() };
    }
    // ---- K-way: t lanes Int32 selectors + d lanes F32 payload -> best/min/argmin ----
    kway(tLanes, dLanes) {
        const k = tLanes.length, n = tLanes[0].length;
        if (k < 1 || k > 256 || dLanes.length !== k)
            throw new Error(`kway: k=${k}`);
        this.reset();
        const tOffs = tLanes.map((t) => this.put(t));
        const dOffs = dLanes.map((d) => this.put(d));
        const pT = this.alloc(k * 4, 8), pD = this.alloc(k * 4, 8);
        const pTB = this.alloc(n * 4, 8), pDB = this.alloc(n * 4, 8), pMB = this.alloc(n, 8);
        this.ensure(this.cur);
        const u = this.u8;
        const tw = new Uint32Array(u.buffer, pT, k), dw = new Uint32Array(u.buffer, pD, k);
        for (let i = 0; i < k; i++) {
            tw[i] = tOffs[i];
            dw[i] = dOffs[i];
        }
        checkRc(this.ex.nf_rowwise_kway_time_argmin_gather(pT, pD, k, n, pTB, pDB, pMB), "kway");
        this.assertGuard();
        return { tBest: this.i32(pTB, n).slice(), dBest: this.f32(pDB, n).slice(), mBest: this.u8v(pMB, n).slice() };
    }
    // ---- adjacency: query[K] -> [begins,ends) slices, then flat gather ----
    adjacencySlice(indptr, indices, query) {
        const k = query.length;
        this.reset();
        const pInd = this.put(indptr), pIx = this.put(indices), pQ = this.put(query);
        const pB = this.alloc(k * 4, 8), pE = this.alloc(k * 4, 8);
        this.ensure(this.cur);
        checkRc(this.ex.nf_adjacency_slice(pInd, indptr.length, pIx, indices.length, pQ, k, pB, pE), "adjacencySlice");
        this.assertGuard();
        return { begins: this.u32(pB, k).slice(), ends: this.u32(pE, k).slice() };
    }
    adjacencyGather(indices, begins, ends, total) {
        const k = begins.length;
        this.reset();
        const pIx = this.put(indices), pB = this.put(begins), pE = this.put(ends);
        const pOut = this.alloc(total * 4, 8);
        this.ensure(this.cur);
        checkRc(this.ex.nf_adjacency_gather(pIx, indices.length, pB, pE, k, pOut, total), "adjacencyGather");
        this.assertGuard();
        return this.u32(pOut, total).slice();
    }
}
export const Q_INF = INF;
