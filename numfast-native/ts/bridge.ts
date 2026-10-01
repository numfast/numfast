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

const alignUp = (off: number, a: number): number => (off + a - 1) & ~(a - 1);

export type WasmExports = WebAssembly.Exports & {
  memory: WebAssembly.Memory;
  nf_sssp_csr: (indptr: number, np: number, indices: number, weights: number, e: number, source: number, dist: number) => number;
  nf_sssp_csr_pred: (indptr: number, np: number, indices: number, weights: number, e: number, source: number, dist: number, pred: number) => number;
  nf_sssp_batch: (indptr: number, np: number, indices: number, weights: number, e: number, sources: number, k: number, out: number, nthreads: number) => number;
  nf_cost_travel_batch: (dist: number, speed: number, k: number, n: number, out: number) => number;
  nf_cost_intern: (vecs: number, n: number, width: number, ids: number, uniq: number, total: number) => bigint;
  nf_rowwise_kway_time_argmin_gather: (tPtrs: number, dPtrs: number, k: number, n: number, tBest: number, dBest: number, mBest: number) => number;
  nf_adjacency_slice: (indptr: number, np: number, indices: number, e: number, query: number, k: number, begins: number, ends: number) => number;
  nf_adjacency_gather: (indices: number, e: number, begins: number, ends: number, k: number, out: number, total: number) => number;
};

export async function loadBridge(wasmBytes: ArrayBuffer | Uint8Array): Promise<Bridge> {
  const bytes = wasmBytes instanceof Uint8Array ? wasmBytes : new Uint8Array(wasmBytes);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return new Bridge(instance.exports as WasmExports);
}

function checkRc(rc: number, op: string): void {
  if (rc !== 0) throw new Error(`${op}: rc=${rc}`);
}

export class Bridge {
  readonly ex: WasmExports;
  private cur: number;

  constructor(ex: WasmExports) {
    this.ex = ex;
    this.cur = BASE;
    this.fillGuard();
  }

  get mem(): WebAssembly.Memory { return this.ex.memory; }
  get u8(): Uint8Array { return new Uint8Array(this.mem.buffer); }

  private fillGuard(): void {
    this.ensure(this.cur);
    new Uint32Array(this.mem.buffer, STACK_TOP, GUARD / 4).fill(CANARY);
  }

  assertGuard(): void {
    const g = new Uint32Array(this.mem.buffer, STACK_TOP, GUARD / 4);
    for (let i = 0; i < g.length; i++) {
      if (g[i] !== CANARY) throw new Error("bridge: wasm stack drifted into buffers");
    }
  }

  ensure(need: number): Uint8Array {
    const have = this.mem.buffer.byteLength;
    if (need > have) this.mem.grow(Math.ceil((need - have) / PAGE));
    return new Uint8Array(this.mem.buffer);
  }

  alloc(bytes: number, align = 8): number {
    const off = alignUp(this.cur, align);
    this.cur = off + bytes;
    return off;
  }

  reset(): void { this.cur = BASE; }

  // Zero-copy views (subarray) — invalidated by grow; re-fetch via u8 after ensure.
  u32(off: number, n: number): Uint32Array { return new Uint32Array(this.mem.buffer, off, n); }
  i32(off: number, n: number): Int32Array { return new Int32Array(this.mem.buffer, off, n); }
  u16(off: number, n: number): Uint16Array { return new Uint16Array(this.mem.buffer, off, n); }
  f32(off: number, n: number): Float32Array { return new Float32Array(this.mem.buffer, off, n); }
  u8v(off: number, n: number): Uint8Array { return new Uint8Array(this.mem.buffer, off, n); }

  put<T extends Uint32Array | Int32Array | Uint16Array | Float32Array | Uint8Array>(v: T): number {
    const off = this.alloc(v.byteLength, 8);
    const u = this.ensure(this.cur);
    u.set(new Uint8Array(v.buffer, v.byteOffset, v.byteLength), off);
    return off;
  }

  // ---- sssp: CSR + weights + source -> dist (u32, INF=unreachable) ----
  ssspCsr(indptr: Uint32Array, indices: Uint32Array, weights: Uint32Array, source: number): Uint32Array {
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
  ssspBatch(indptr: Uint32Array, indices: Uint32Array, weights: Uint32Array, sources: Uint32Array): Uint32Array {
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
  costTravel(dist: Uint32Array, speed: Uint32Array, k: Uint16Array): Uint32Array {
    const n = dist.length;
    this.reset();
    const pD = this.put(dist), pS = this.put(speed), pK = this.put(k);
    const pOut = this.alloc(n * 4, 8);
    this.ensure(this.cur);
    checkRc(this.ex.nf_cost_travel_batch(pD, pS, pK, n, pOut), "costTravel");
    this.assertGuard();
    return this.u32(pOut, n).slice();
  }

  costTravelInto(dOff: number, sOff: number, kOff: number, n: number, outOff: number): number {
    return this.ex.nf_cost_travel_batch(dOff, sOff, kOff, n, outOff);
  }

  // ---- cost interning: row-major vecs[n*width] -> ids + uniq table (returns ng) ----
  costIntern(vecs: Uint32Array, n: number, width: number): { ng: number; ids: Uint32Array; uniq: Uint32Array } {
    this.reset();
    const pV = this.put(vecs);
    const pIds = this.alloc(n * 4, 8);
    const pUniq = this.alloc(n * width * 4, 8);
    this.ensure(this.cur);
    const ng = Number(this.ex.nf_cost_intern(pV, n, width, pIds, pUniq, n * width));
    if (ng < 0) throw new Error(`costIntern: ng=${ng}`);
    this.assertGuard();
    return { ng, ids: this.u32(pIds, n).slice(), uniq: this.u32(pUniq, ng * width).slice() };
  }

  // ---- K-way: t lanes Int32 selectors + d lanes F32 payload -> best/min/argmin ----
  kway(tLanes: Int32Array[], dLanes: Float32Array[]): { tBest: Int32Array; dBest: Float32Array; mBest: Uint8Array } {
    const k = tLanes.length, n = tLanes[0].length;
    if (k < 1 || k > 256 || dLanes.length !== k) throw new Error(`kway: k=${k}`);
    this.reset();
    const tOffs = tLanes.map((t) => this.put(t));
    const dOffs = dLanes.map((d) => this.put(d));
    const pT = this.alloc(k * 4, 8), pD = this.alloc(k * 4, 8);
    const pTB = this.alloc(n * 4, 8), pDB = this.alloc(n * 4, 8), pMB = this.alloc(n, 8);
    this.ensure(this.cur);
    const u = this.u8;
    const tw = new Uint32Array(u.buffer, pT, k), dw = new Uint32Array(u.buffer, pD, k);
    for (let i = 0; i < k; i++) { tw[i] = tOffs[i]; dw[i] = dOffs[i]; }
    checkRc(this.ex.nf_rowwise_kway_time_argmin_gather(pT, pD, k, n, pTB, pDB, pMB), "kway");
    this.assertGuard();
    return { tBest: this.i32(pTB, n).slice(), dBest: this.f32(pDB, n).slice(), mBest: this.u8v(pMB, n).slice() };
  }

  // ---- adjacency: query[K] -> [begins,ends) slices, then flat gather ----
  adjacencySlice(indptr: Uint32Array, indices: Uint32Array, query: Uint32Array): { begins: Uint32Array; ends: Uint32Array } {
    const k = query.length;
    this.reset();
    const pInd = this.put(indptr), pIx = this.put(indices), pQ = this.put(query);
    const pB = this.alloc(k * 4, 8), pE = this.alloc(k * 4, 8);
    this.ensure(this.cur);
    checkRc(this.ex.nf_adjacency_slice(pInd, indptr.length, pIx, indices.length, pQ, k, pB, pE), "adjacencySlice");
    this.assertGuard();
    return { begins: this.u32(pB, k).slice(), ends: this.u32(pE, k).slice() };
  }

  adjacencyGather(indices: Uint32Array, begins: Uint32Array, ends: Uint32Array, total: number): Uint32Array {
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
