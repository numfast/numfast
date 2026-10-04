// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// TS bridge over the raw cdylib ABI of numfast-native (no wasm-bindgen).
// ESM-only, zero-copy TypedArray views (subarray, never slice).
// Runs in Node >=22.18 (native type-stripping) and browsers (via tsc build).
// Decision: raw ABI — wasm-bindgen rejected (zero deps, usize->i32 ABI stable).
//
// BUFFER PLACEMENT — a measured defect this file now avoids.
//
// The previous revision started caller buffers at 0x101000, on the reasoning
// that the wasm shadow stack top is 0x100000 (it is: the module's single
// mutable i32 global initialises to 0x100000). That reasoning is wrong.
// Measured on the current artefact, 2026-10-04, by diffing linear memory
// before and after instantiation:
//
//   the module's own initialised data occupies [0x100000, 0x103203)
//   -- 8764 non-zero bytes, and NOTHING below 0x100000.
//
// The float constants that `powf` reads live inside that region. A caller
// buffer that overlaps it silently overwrites them and `powf` returns
// plausible, wrong numbers with rc=0: at a 0x101000 base, any input longer
// than 8632 bytes did it, while the same input at a 0x20000 base was correct.
// Native (ctypes) was correct at every size. That is the whole of the
// reported "`map pow` diverges at n=100000" defect, and it is a caller-side
// placement bug, not a kernel bug and not a WASM/native numerical difference.
//
// So the base is no longer a constant: it is probed from the instantiated
// memory and placed above whatever this build put there.

export const GUARD = 0x1000;
export const CANARY = 0x9e3779b9;
const PAGE = 65536;
const INF = 0xffffffff;
/** Room between the top of the module's data and the first caller buffer. */
const DATA_MARGIN = 0x1000;

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
  // elementwise map -- see series/map.rs for the op codes
  nf_map_i32: (a: number, b: number, n: number, op: number, out: number) => number;
  nf_map_scalar_i32: (a: number, n: number, s: number, op: number, out: number) => number;
  nf_map_fscalar_i32: (a: number, n: number, s: number, op: number, out: number) => number;
  nf_map_f32: (a: number, b: number, n: number, op: number, out: number) => number;
  nf_map_f32_divpow: (a: number, b: number, n: number, op: number, out: number) => number;
  nf_map_scalar_f32: (a: number, n: number, s: number, op: number, out: number) => number;
  nf_map_scalar_f32_divpow: (a: number, n: number, s: number, op: number, out: number) => number;
  nf_map_f64: (a: number, b: number, n: number, op: number, out: number) => number;
  nf_map_scalar_f64: (a: number, n: number, s: number, op: number, out: number) => number;
};

export async function loadBridge(wasmBytes: ArrayBuffer | Uint8Array): Promise<Bridge> {
  const bytes = wasmBytes instanceof Uint8Array ? wasmBytes : new Uint8Array(wasmBytes);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return new Bridge(instance.exports as WasmExports);
}

export class Bridge {
  readonly ex: WasmExports;
  /** First byte a caller buffer may occupy. Above the module's own data. */
  readonly base: number;
  /** End of the module's own initialised data, measured not assumed. */
  readonly staticDataEnd: number;
  private cur: number;

  constructor(ex: WasmExports) {
    this.ex = ex;
    this.staticDataEnd = probeStaticDataEnd(ex.memory);
    this.base = alignUp(this.staticDataEnd + DATA_MARGIN, PAGE);
    this.cur = this.base;
    this.fillGuard();
  }

  get mem(): WebAssembly.Memory { return this.ex.memory; }
  get u8(): Uint8Array { return new Uint8Array(this.mem.buffer); }
  /** The guard page immediately below the first caller buffer. */
  get guardBase(): number { return this.base - GUARD; }

  private fillGuard(): void {
    this.ensure(this.cur);
    new Uint32Array(this.mem.buffer, this.guardBase, GUARD / 4).fill(CANARY);
  }

  assertGuard(): void {
    const g = new Uint32Array(this.mem.buffer, this.guardBase, GUARD / 4);
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
    if (off < this.staticDataEnd) {
      // Unreachable by construction; stated rather than assumed.
      throw new Error(`bridge: allocation at ${off} would overlap module data ending at ${this.staticDataEnd}`);
    }
    this.cur = off + bytes;
    return off;
  }

  reset(): void { this.cur = this.base; }

  // Zero-copy views (subarray) — invalidated by grow; re-fetch via u8 after ensure.
  u32(off: number, n: number): Uint32Array { return new Uint32Array(this.mem.buffer, off, n); }
  i32(off: number, n: number): Int32Array { return new Int32Array(this.mem.buffer, off, n); }
  u16(off: number, n: number): Uint16Array { return new Uint16Array(this.mem.buffer, off, n); }
  f32(off: number, n: number): Float32Array { return new Float32Array(this.mem.buffer, off, n); }
  f64(off: number, n: number): Float64Array { return new Float64Array(this.mem.buffer, off, n); }
  u8v(off: number, n: number): Uint8Array { return new Uint8Array(this.mem.buffer, off, n); }

  put<T extends Uint32Array | Int32Array | Uint16Array | Float32Array | Float64Array | Uint8Array>(v: T): number {
    const off = this.alloc(v.byteLength, 8);
    this.ensure(this.cur);
    this.u8.set(new Uint8Array(v.buffer as ArrayBuffer, v.byteOffset, v.byteLength), off);
    return off;
  }

  /** Raw-offset call of `nf_cost_travel_batch`, returning the rc untranslated.
   *
   *  Kept for the two benchmark drivers (`qlookup.ts`, `demo.html`) that own
   *  their own layout and measure the kernel alone. It is deliberately NOT
   *  the library entry point: `kernels.costTravelBatch` validates, resolves
   *  the return code, and converts traps. Use that one. */
  costTravelInto(dOff: number, sOff: number, kOff: number, n: number, outOff: number): number {
    return this.ex.nf_cost_travel_batch(dOff, sOff, kOff, n, outOff);
  }
}

/**
 * End of the module's own initialised data, found by scanning the exported
 * memory of a freshly instantiated module.
 *
 * wasm-ld merges `.rodata` and `.data` into segments whose declaration in the
 * data section does NOT reveal the true extent (the shipped .wasm declares
 * 2 data segments ending at 0x10000B while its real initialised image runs to
 * 0x103203). The image itself is authoritative: everything the module did not
 * initialise is zero, and this build leaves nothing below its first page, so
 * the highest non-zero byte is the top of its data.
 *
 * A future build that pre-faults memory above its data would make this
 * over-estimate, which costs a few kilobytes and is safe. A build that
 * initialises nothing would make it return 0 and place buffers over live
 * data — so `Bridge.alloc` re-checks every offset against the result.
 */
export function probeStaticDataEnd(mem: WebAssembly.Memory): number {
  const u8 = new Uint8Array(mem.buffer);
  let hi = 0;
  for (let i = u8.length - 1; i >= 0; i--) {
    if (u8[i] !== 0) { hi = i + 1; break; }
  }
  return hi;
}