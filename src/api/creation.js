/**
 * creation.js — Creation API (JS_PARITY_SPEC §2.1).
 *
 * zeros ones full arange linspace index tile repeat.
 * Every function validates n (0 -> empty; > MAX_ELEMENTS -> Error,
 * chunking planned for v1), runs IndexKernel/IndexKernelI32 through the CPU
 * execution path and returns { data: Float32Array|Int32Array, length, dtype }.
 *
 * Kernel modes used (see Compute/_lib/index_kernel): 0=const, 1=arange,
 * 2=linspace, 3=tile(pattern), 4=repeat(pattern).
 *
 * GPU path ({gpu:true} trailing option): every function executes through
 * src/api/webgpu_driver.js (IndexKernel/IndexKernelI32 WGSL) and returns a
 * Promise of the same {data, length, dtype} shape; when WebGPU is unavailable
 * or the launch fails it falls back to the CPU kernel path. Without the
 * option behaviour is synchronous CPU, unchanged.
 *
 * random namespace: RandomKernel (uniform/integers/normal), same {gpu:true}
 * contract. Seed defaults to 42 (reproducibility rule).
 */

"use strict";

const { runKernel } = require("./_runner");
const { ensureReady, runIndexKernel, runRandomKernel } = require("./webgpu_driver");

const MAX_ELEMENTS = 4194240; // chunking planned for v1

function _n(shape) {
  const n = Array.isArray(shape) ? shape[0] : shape;
  const v = Math.trunc(Number(n));
  if (Number.isNaN(v)) throw new Error(`creation: invalid shape ${JSON.stringify(shape)}`);
  return v;
}

function _checkN(n, fn) {
  if (n < 0) throw new Error(`${fn}: n must be >= 0, got ${n}`);
  if (n > MAX_ELEMENTS) {
    throw new Error(
      `${fn}: n=${n} exceeds ${MAX_ELEMENTS} — chunking is planned for v1`);
  }
}

function _empty(dtype) {
  const data = dtype === "int32" ? new Int32Array(0) : new Float32Array(0);
  return { data, length: 0, dtype };
}

function _finish(raw, dtype) {
  const data = dtype === "int32"
    ? (raw instanceof Int32Array ? raw : Int32Array.from(raw))
    : Float32Array.from(raw); // f32 storage at the API boundary
  return { data, length: data.length, dtype: dtype === "int32" ? "int32" : "float32" };
}

/** GPU-first execution wrapper: WebGPU launch, CPU fallback on any failure. */
async function _gpuOrCpu(cpuThunk, gpuThunk) {
  try {
    if (await ensureReady()) return await gpuThunk();
  } catch (_) { /* WebGPU unavailable or launch failed -> CPU backend */ }
  return cpuThunk();
}

/** zeros(shape, dtype="float32", opts) */
function zeros(shape, dtype, opts) {
  return _const(shape, 0.0, dtype, opts);
}

/** ones(shape, dtype="float32", opts) */
function ones(shape, dtype, opts) {
  return _const(shape, 1.0, dtype, opts);
}

/** full(shape, value, dtype="float32", opts) */
function full(shape, value, dtype, opts) {
  if (value === undefined || value === null) {
    throw new Error("full: value is required");
  }
  return _const(shape, Number(value), dtype, opts);
}

function _const(shape, value, dtype, opts) {
  const dt = dtype === undefined ? "float32" : dtype;
  const n = _n(shape);
  _checkN(n, "creation");
  if (n === 0) return _empty(dt);
  if (dt !== "float32" && dt !== "int32") {
    throw new Error(`creation: dtype ${dt} not supported (float32|int32)`);
  }
  const params = { n, mode: 0 };
  if (dt === "int32") {
    params.p0 = Math.trunc(value);
    params.dtype = "int32";
    if (opts && opts.gpu) {
      return _gpuOrCpu(
        () => zeros(shape, dtype),
        async () => _finish(await runIndexKernel(params), "int32"));
    }
    const raw = runKernel("IndexKernelI32", params, [], n);
    return _finish(raw, "int32");
  }
  params.p0 = value;
  if (opts && opts.gpu) {
    return _gpuOrCpu(
      () => zeros(shape, dtype),
      async () => _finish(await runIndexKernel(params), "float32"));
  }
  const raw = runKernel("IndexKernel", params, [], n);
  return _finish(raw, "float32");
}

/** arange(start=0, stop, step=1, opts) — float32 */
function arange(start, stop, step, opts) {
  if (stop === undefined) { stop = start; start = 0; }
  start = start === undefined ? 0 : Number(start);
  step = step === undefined ? 1 : Number(step);
  stop = Number(stop);
  if (step === 0) throw new Error("arange: step must be != 0");
  const rawN = (stop - start) / step;
  const n = Math.max(0, Math.ceil(rawN));
  _checkN(n, "arange");
  if (n === 0) return _empty("float32");
  const params = { n, mode: 1, p0: start, p1: step };
  if (opts && opts.gpu) {
    return _gpuOrCpu(
      () => arange(start, stop, step),
      async () => _finish(await runIndexKernel(params), "float32"));
  }
  const raw = runKernel("IndexKernel", params, [], n);
  return _finish(raw, "float32");
}

/** linspace(start, stop, num, endpoint=true, opts) — float32 */
function linspace(start, stop, num, endpoint, opts) {
  // opts may be passed in the endpoint slot: linspace(a, b, n, {gpu:true})
  let ep;
  if (endpoint !== undefined && typeof endpoint === "object" &&
      !Array.isArray(endpoint)) {
    opts = endpoint;
    ep = true;
  } else {
    ep = endpoint === undefined ? true : Boolean(endpoint);
  }
  start = Number(start); stop = Number(stop);
  const n = Math.trunc(Number(num));
  _checkN(n, "linspace");
  if (n === 0) return _empty("float32");
  const params = { n, mode: 2, p0: start, p1: stop };
  // endpoint=true -> descriptor default denom max(n-1,1) (numpy fast path,
  // last element forced to stop); endpoint=false -> denom n (stop excluded).
  if (!ep) params.p2 = n;
  if (opts && opts.gpu) {
    return _gpuOrCpu(
      () => linspace(start, stop, num, ep),
      async () => _finish(await runIndexKernel(params), "float32"));
  }
  const raw = runKernel("IndexKernel", params, [], n);
  return _finish(raw, "float32");
}

/** index(n, dtype="int32", opts) — arange 0..n-1, exact in the requested domain */
function index(n, dtype, opts) {
  const dt = dtype === undefined || (dtype && typeof dtype === "object")
    ? "int32" : dtype;
  if (dtype && typeof dtype === "object") opts = dtype;
  const len = Math.trunc(Number(n));
  _checkN(len, "index");
  if (len === 0) return _empty(dt);
  const params = { n: len, mode: 1, p0: 0, p1: 1 };
  if (dt === "int32") {
    params.dtype = "int32";
    if (opts && opts.gpu) {
      return _gpuOrCpu(
        () => index(len, "int32"),
        async () => _finish(await runIndexKernel(params), "int32"));
    }
    const raw = runKernel("IndexKernelI32", params, [], len);
    return _finish(raw, "int32");
  }
  if (dt === "float32") {
    if (opts && opts.gpu) {
      return _gpuOrCpu(
        () => index(len, "float32"),
        async () => _finish(await runIndexKernel(params), "float32"));
    }
    const raw = runKernel("IndexKernel", params, [], len);
    return _finish(raw, "float32");
  }
  throw new Error(`index: dtype ${dt} not supported (float32|int32)`);
}

function _pattern(pattern, fn) {
  const pat = Float32Array.from(pattern);
  if (pat.length < 1) throw new Error(`${fn}: pattern must contain at least 1 element`);
  return pat;
}

/** tile(pattern, n, opts) — out[i] = pattern[i % k] */
function tile(pattern, n, opts) {
  const pat = _pattern(pattern, "tile");
  const len = Math.trunc(Number(n));
  _checkN(len, "tile");
  if (len === 0) return _empty("float32");
  const params = { n: len, mode: 3, k: pat.length };
  if (opts && opts.gpu) {
    return _gpuOrCpu(
      () => tile(pattern, len),
      async () => _finish(await runIndexKernel(params, pat), "float32"));
  }
  const raw = runKernel("IndexKernel", params, [pat], len);
  return _finish(raw, "float32");
}

/** repeat(pattern, repeats, n=null, opts) — out[i] = pattern[floor(i / repeats)] */
function repeat(pattern, repeats, n, opts) {
  if (n && typeof n === "object") { opts = n; n = null; }
  const pat = _pattern(pattern, "repeat");
  const rep = Math.trunc(Number(repeats));
  if (rep < 1) throw new Error(`repeat: repeats must be >= 1, got ${rep}`);
  const len = n === undefined || n === null
    ? pat.length * rep
    : Math.trunc(Number(n));
  _checkN(len, "repeat");
  if (len === 0) return _empty("float32");
  const params = { n: len, mode: 4, k: pat.length, rep };
  if (opts && opts.gpu) {
    return _gpuOrCpu(
      () => repeat(pattern, rep, len),
      async () => _finish(await runIndexKernel(params, pat), "float32"));
  }
  const raw = runKernel("IndexKernel", params, [pat], len);
  return _finish(raw, "float32");
}

// ---- random namespace (RandomKernel: uniform / integers / normal) ----------

function _randomCpu(n, mode, seed, p0, p1) {
  const raw = runKernel("RandomKernel",
    { n, mode, seed: seed >>> 0, p0, p1 }, [], n);
  return _finish(raw, "float32");
}

function _randomCore(mode, n, p0, p1, seed, opts) {
  const count = Math.trunc(Number(n));
  _checkN(count, "random");
  if (!(count >= 1)) {
    throw new Error(`RandomKernel: n must be >= 1, got ${count}`);
  }
  const sd = Math.trunc(Number(seed === undefined ? 42 : seed));
  const params = { n: count, mode, seed: sd >>> 0, p0: Number(p0), p1: Number(p1) };
  if (opts && opts.gpu) {
    return _gpuOrCpu(
      () => _randomCpu(count, mode, params.seed, p0, p1),
      async () => _finish(await runRandomKernel(params), "float32"));
  }
  return _randomCpu(count, mode, params.seed, p0, p1);
}

/** random.uniform(n, low=0, high=1, seed=42, opts) */
function uniform(n, low, high, seed, opts) {
  return _randomCore(0, n,
    low === undefined ? 0.0 : low,
    high === undefined ? 1.0 : high, seed, opts);
}

/** random.integers(n, low=0, high=100, seed=42, opts) — int_range [low, high) */
function integers(n, low, high, seed, opts) {
  return _randomCore(1, n,
    low === undefined ? 0.0 : low,
    high === undefined ? 100.0 : high, seed, opts);
}

/** random.normal(n, loc=0, scale=1, seed=42, opts) */
function normal(n, loc, scale, seed, opts) {
  return _randomCore(2, n,
    loc === undefined ? 0.0 : loc,
    scale === undefined ? 1.0 : scale, seed, opts);
}

const random = { uniform, integers, normal };

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    zeros, ones, full, arange, linspace, index, tile, repeat,
    random,
    MAX_ELEMENTS,
  };
}
