/**
 * series.js — Series class (JS_PARITY_SPEC §2.3).
 *
 * Thin wrapper over TypedArray. Operators run through existing cpu kernels:
 *   add/sub/mul/div -> MapBinary op 0..3 (scalar broadcast via use_scalar_b)
 *   square/neg      -> Map func 7 / 6
 *   lt/le/gt/ge/eq/ne -> Compare ops (mask: Uint8Array wrapped as Series)
 * mod (C-trunc fmod, b==0 -> 0) and pow have NO op-codes in the JS MapBinary
 * registry (JS has ops 0..5 only; Python adds 6=fmod, no pow) — both are
 * computed inline with kernel-matching semantics.
 *
 * filter(mask): positions where mask!=0 -> Gather(src=data, idx).
 * topk(k): Sort bitonic network (substages DESCENDING — canonical order) on
 * -Inf padded pow2 buffer, Gather of the last k indices reversed.
 *
 * v1 note: every operator executes eagerly; compute() returns the
 * materialized result (single-op expressions are already executed).
 *
 * GPU path ({gpu:true} trailing option): add/sub/mul/div, square/neg and
 * lt/le/gt/ge/eq/ne execute through src/api/webgpu_driver.js (existing
 * Map/MapBinary/Compare WGSL). With {gpu:true} the method returns
 * Promise<Series>; when WebGPU is unavailable or a launch fails it falls
 * back to the CPU kernel path. mod/pow/filter/topk stay CPU-only in v1
 * (no MapBinary op-codes; Sort not wired to the driver yet).
 */

"use strict";

const { runKernel } = require("./_runner");
const { ensureReady, runMap, runMapBinary, runCompare } = require("./webgpu_driver");

const _MB_OP = { add: 0, sub: 1, mul: 2, div: 3 };
const _CMP_OP = { lt: "lt", le: "le", gt: "gt", ge: "ge", eq: "eq", ne: "ne" };
// Compare WGSL numeric op codes: 0=gt 1=ge 2=lt 3=le 4=eq 5=ne.
const _CMP_NUM = { gt: 0, ge: 1, lt: 2, le: 3, eq: 4, ne: 5 };

function _isOperand(x) {
  return x instanceof Series || ArrayBuffer.isView(x) || Array.isArray(x);
}

/** Run gpuThunk through the shared driver; any failure -> cpuThunk fallback. */
async function _gpuOrCpu(cpuThunk, gpuThunk) {
  try {
    if (await ensureReady()) return await gpuThunk();
  } catch (_) { /* WebGPU unavailable or launch failed -> CPU backend */ }
  return cpuThunk();
}

async function _gpuMapBinary(opCode, self, other) {
  const a = self.data;
  if (_isOperand(other)) {
    const b = Float32Array.from(_raw(other));
    if (b.length !== 1 && b.length !== a.length) {
      throw new Error(
        `Series: operand length mismatch (${a.length} vs ${b.length}); ` +
        `only scalar or length-1 broadcast`);
    }
    const out = await runMapBinary(opCode, a, b);
    return new Series(Float32Array.from(out));
  }
  const out = await runMapBinary(opCode, a, Number(other));
  return new Series(Float32Array.from(out));
}

async function _gpuMap(funcCode, self) {
  const out = await runMap(funcCode, self.data);
  return new Series(Float32Array.from(out));
}

function _maskSeries(out) {
  const mask = new Uint8Array(out.length);
  for (let i = 0; i < mask.length; i++) mask[i] = out[i] ? 1 : 0;
  const s = new Series(mask);
  s.isMask = true;
  return s;
}

async function _gpuCompare(numOp, self, other) {
  const a = self.data;
  const out = _isOperand(other)
    ? await runCompare(numOp, a, Float32Array.from(_raw(other)), 0)
    : await runCompare(numOp, a, Number(other), 1);
  return _maskSeries(out);
}

function _raw(x) {
  if (x instanceof Series) return x.data;
  if (x instanceof Float32Array || x instanceof Int32Array ||
      x instanceof Uint8Array || x instanceof Uint32Array ||
      x instanceof Float64Array) return x;
  if (Array.isArray(x)) return Float32Array.from(x);
  throw new Error(`Series: unsupported operand ${typeof x}`);
}

class Series {
  constructor(data) {
    const d = _raw(data);
    if (!(d && d.length !== undefined && typeof d.length === "number" &&
          ArrayBuffer.isView(d))) {
      throw new Error("Series: data must be a TypedArray");
    }
    this.data = d;
  }

  static from(arr) {
    return new Series(Float32Array.from(_raw(arr)));
  }

  get length() { return this.data.length; }
  len() { return this.data.length; }

  /** Materialize (v1: operators are eager, single-op already executed). */
  compute() { return this; }

  toArray() { return Array.from(this.data); }

  // ---- binary arithmetic (MapBinary op 0..3, scalar broadcast) ----
  /** opts.gpu -> Promise<Series> via WebGPU (CPU fallback). */
  _maybeGpu(opts, cpuThunk, gpuThunk) {
    if (!(opts && opts.gpu)) return cpuThunk();
    return _gpuOrCpu(cpuThunk, gpuThunk);
  }

  _mapBinary(opCode, other) {
    const a = this.data;
    if (_isOperand(other)) {
      const b = Float32Array.from(_raw(other));
      if (b.length !== 1 && b.length !== a.length) {
        throw new Error(
          `Series: operand length mismatch (${a.length} vs ${b.length}); ` +
          `only scalar or length-1 broadcast`);
      }
      const out = runKernel("MapBinary", { op: opCode }, [a, b], a.length);
      return new Series(Float32Array.from(out));
    }
    const out = runKernel("MapBinary",
      { op: opCode, use_scalar_b: 1, scalar_b: Number(other) }, [a], a.length);
    return new Series(Float32Array.from(out));
  }

  add(b, opts) {
    return this._maybeGpu(opts,
      () => this._mapBinary(_MB_OP.add, b),
      () => _gpuMapBinary(_MB_OP.add, this, b));
  }
  sub(b, opts) {
    return this._maybeGpu(opts,
      () => this._mapBinary(_MB_OP.sub, b),
      () => _gpuMapBinary(_MB_OP.sub, this, b));
  }
  mul(b, opts) {
    return this._maybeGpu(opts,
      () => this._mapBinary(_MB_OP.mul, b),
      () => _gpuMapBinary(_MB_OP.mul, this, b));
  }
  div(b, opts) {
    return this._maybeGpu(opts,
      () => this._mapBinary(_MB_OP.div, b),
      () => _gpuMapBinary(_MB_OP.div, this, b));
  }

  /** C-trunc modulo (MapBinary op-6 semantics): b==0 -> 0 */
  mod(b) {
    const a = this.data;
    const arr = b instanceof Series || ArrayBuffer.isView(b) || Array.isArray(b)
      ? Float32Array.from(_raw(b)) : null;
    const out = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) {
      const bv = arr
        ? (arr.length > 1 ? arr[i] : arr[0])
        : Number(b);
      out[i] = bv === 0 ? 0 : a[i] - Math.trunc(a[i] / bv) * bv;
    }
    return new Series(Float32Array.from(out));
  }

  /** pow: no MapBinary op-code in JS registry — inline Math.pow loop */
  pow(b) {
    const a = this.data;
    const arr = b instanceof Series || ArrayBuffer.isView(b) || Array.isArray(b)
      ? Float32Array.from(_raw(b)) : null;
    const out = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) {
      const bv = arr ? (arr.length > 1 ? arr[i] : arr[0]) : Number(b);
      out[i] = Math.pow(a[i], bv);
    }
    return new Series(Float32Array.from(out));
  }

  // ---- unary (Map func codes) ----
  _map(funcCode) {
    const out = runKernel("Map", { func: funcCode }, [this.data], this.data.length);
    return new Series(Float32Array.from(out));
  }
  square(opts) {
    return this._maybeGpu(opts,
      () => this._map(7),
      () => _gpuMap(7, this));
  }
  neg(opts) {
    return this._maybeGpu(opts,
      () => this._map(6),
      () => _gpuMap(6, this));
  }

  // ---- comparisons -> mask Series over Uint8Array ----
  _compare(opName, other) {
    const a = this.data;
    let params;
    let inputs;
    if (_isOperand(other)) {
      const b = Float32Array.from(_raw(other));
      params = { op: _CMP_OP[opName] };
      inputs = [a, b];
    } else {
      params = { op: _CMP_OP[opName], use_scalar_b: 1, scalar_b: Number(other) };
      inputs = [a];
    }
    const out = runKernel("Compare", params, inputs, a.length);
    return _maskSeries(out);
  }

  lt(b, opts) {
    return this._maybeGpu(opts,
      () => this._compare("lt", b),
      () => _gpuCompare(_CMP_NUM.lt, this, b));
  }
  le(b, opts) {
    return this._maybeGpu(opts,
      () => this._compare("le", b),
      () => _gpuCompare(_CMP_NUM.le, this, b));
  }
  gt(b, opts) {
    return this._maybeGpu(opts,
      () => this._compare("gt", b),
      () => _gpuCompare(_CMP_NUM.gt, this, b));
  }
  ge(b, opts) {
    return this._maybeGpu(opts,
      () => this._compare("ge", b),
      () => _gpuCompare(_CMP_NUM.ge, this, b));
  }
  eq(b, opts) {
    return this._maybeGpu(opts,
      () => this._compare("eq", b),
      () => _gpuCompare(_CMP_NUM.eq, this, b));
  }
  ne(b, opts) {
    return this._maybeGpu(opts,
      () => this._compare("ne", b),
      () => _gpuCompare(_CMP_NUM.ne, this, b));
  }

  // ---- compositions ----

  /** filter(mask): values at positions where mask != 0 (Compare->Gather glue). */
  filter(mask) {
    const m = _raw(mask);
    if (m.length !== this.data.length) {
      throw new Error(
        `filter: mask length ${m.length} != series length ${this.data.length}`);
    }
    let count = 0;
    for (let i = 0; i < m.length; i++) if (m[i] !== 0) count++;
    const idx = new Float64Array(count);
    let j = 0;
    for (let i = 0; i < m.length; i++) if (m[i] !== 0) idx[j++] = i;
    if (count === 0) return new Series(new Float32Array(0));
    const out = runKernel("Gather", {}, [this.data, idx], count);
    return new Series(Float32Array.from(out));
  }

  /** topk(k): k largest values, descending (Sort + Gather composition). */
  topk(k) {
    const n = this.data.length;
    const kk = Math.min(Math.max(0, Math.trunc(Number(k))), n);
    if (kk === 0) return new Series(new Float32Array(0));
    if (n === 0) return new Series(new Float32Array(0));

    // pad to next power of two with -Infinity (sorts to front in ascending)
    let m = 1;
    while (m < n) m *= 2;
    const buf = new Float32Array(m);
    buf.fill(-Infinity);
    buf.set(this.data);

    // canonical bitonic network: substages DESCENDING inside each stage
    let cur = buf;
    const stages = Math.round(Math.log2(m));
    for (let stage = 1; stage <= stages; stage++) {
      for (let substage = stage; substage >= 1; substage--) {
        cur = runKernel("Sort",
          { stage, substage }, [cur], m);
      }
    }

    // gather last k sorted slots in reverse -> descending top-k
    const idx = new Float64Array(kk);
    for (let i = 0; i < kk; i++) idx[i] = m - 1 - i;
    const out = runKernel("Gather", {}, [cur, idx], kk);
    return new Series(Float32Array.from(out));
  }
}

/** nf.topk(s, k) namespace-level form. */
function topk(s, k) {
  if (s instanceof Series) return s.topk(k);
  return Series.from(s).topk(k);
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { Series, topk };
}
