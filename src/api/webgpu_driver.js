/**
 * webgpu_driver.js — WebGPU compute driver (JS parity part C, STEP 1).
 *
 * Real GPU execution for the five element-wise/creation primitives whose WGSL
 * already ships in Compute/_lib/<kernel>/wgsl.js:
 *   IndexKernel / IndexKernelI32, RandomKernel, Map, MapBinary, Compare.
 *
 * Convention contract:
 * - Default bind layout ("unified"): binding 0 = output storage,
 *   binding 1..N = inputs read-only-storage, LAST binding = uniform.
 * - Existing kernel WGSL predates the unified convention, so executeKernel()
 *   accepts an explicit bindingPlan (array of "out"|"in"|"uniform", one entry
 *   per binding slot, LAST must be "uniform"). Adapters below supply the plan
 *   matching each shipped shader; no WGSL is duplicated here.
 *
 * Lifecycle: init() once (navigator.gpu.requestAdapter -> requestDevice),
 * then executeKernel() per launch: shaderModule cache -> pipeline -> buffers
 * (queue.writeBuffer upload) -> bind group -> dispatchWorkgroups ->
 * staging MAP_READ readback. Mirrors develop/numfast_js reference driver.
 *
 * Fallback rule: navigator.gpu missing or requestAdapter() fails ->
 * init()/ensureReady() rejects/resolves false; public API falls back to CPU.
 */

"use strict";

const indexWgsl = require("../math/Compute/_lib/index_kernel/wgsl");
const randomWgsl = require("../math/Compute/_lib/random_kernel/wgsl");
const mapWgsl = require("../math/Compute/_lib/map/wgsl");
const mapBinaryWgsl = require("../math/Compute/_lib/map_binary/wgsl");
const compareWgsl = require("../math/Compute/_lib/compare/wgsl");

const WORKGROUP = 64;          // matches @workgroup_size(64) in shipped WGSL
const MAX_WORKGROUPS = 65535;  // WebGPU per-dimension dispatch limit
// creation.MAX_ELEMENTS == MAX_WORKGROUPS * WORKGROUP -> 1D dispatch fits.

const _DTYPES = { float32: Float32Array, int32: Int32Array, uint32: Uint32Array };

function gpuAvailable() {
  return typeof navigator !== "undefined" &&
    typeof navigator.gpu !== "undefined" && navigator.gpu !== null;
}

/** Pack scalars into a little-endian uniform block (16-byte padded). */
function packUniforms(fields) {
  let size = 0;
  for (let i = 0; i < fields.length; i++) size += 4;
  const bytes = new Uint8Array(Math.max(16, Math.ceil(size / 16) * 16));
  const view = new DataView(bytes.buffer);
  let off = 0;
  for (const [value, type] of fields) {
    if (type === "u32") view.setUint32(off, value >>> 0, true);
    else if (type === "i32") view.setInt32(off, value | 0, true);
    else view.setFloat32(off, Number(value), true);
    off += 4;
  }
  return bytes;
}

class WebGPUDriver {
  constructor() {
    this.adapter = null;
    this.device = null;
    this._shaders = new Map();
  }

  get ready() { return !!this.device; }

  /** requestAdapter -> requestDevice. Throws when WebGPU unavailable. */
  async init() {
    if (!gpuAvailable()) {
      throw new Error("WebGPU unavailable, use CPU backend");
    }
    let adapter = null;
    try { adapter = await navigator.gpu.requestAdapter(); } catch (_) { adapter = null; }
    if (!adapter) throw new Error("WebGPU unavailable: no adapter, use CPU backend");
    const requiredLimits = {};
    if (adapter.limits) {
      if (adapter.limits.maxStorageBufferBindingSize !== undefined) {
        requiredLimits.maxStorageBufferBindingSize =
          adapter.limits.maxStorageBufferBindingSize;
      }
      if (adapter.limits.maxBufferSize !== undefined) {
        requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
      }
    }
    this.device = await adapter.requestDevice(
      Object.keys(requiredLimits).length ? { requiredLimits } : undefined);
    this.adapter = adapter;
    return this;
  }

  /** adapterInfo?.vendor with guards for partial implementations. */
  adapterVendor() {
    try {
      const info = this.adapter && this.adapter.info;
      return (info && info.vendor) || "";
    } catch (_) { return ""; }
  }

  _module(code) {
    let m = this._shaders.get(code);
    if (!m) {
      m = this.device.createShaderModule({ code });
      this._shaders.set(code, m);
    }
    return m;
  }

  /**
   * One compute launch + readback.
   * wgslString     — WGSL source (entryPoint "main").
   * buffersIn      — TypedArrays uploaded to read-only storage slots.
   * buffersOut     — [{length, dtype="float32"|"int32"|"uint32"}].
   * uniformsBytes  — packed uniform block (see packUniforms) or byte view.
   * dispatchDims   — workgroups X (number) or [X, Y].
   * bindingPlan    — optional ["out"|"in"|"uniform", ...]; default unified.
   * Returns array of decoded TypedArrays, one per buffersOut entry.
   */
  async executeKernel(wgslString, buffersIn, buffersOut, uniformsBytes,
                      dispatchDims, bindingPlan) {
    if (!this.ready) {
      throw new Error("WebGPU driver not initialized — call init() first");
    }
    const device = this.device;
    const module = this._module(wgslString);

    const plan = bindingPlan ||
      ["out"].concat(buffersIn.map(() => "in"), ["uniform"]);
    if (plan[plan.length - 1] !== "uniform") {
      throw new Error("executeKernel: LAST binding must be uniform");
    }

    const layout = device.createBindGroupLayout({
      entries: plan.map((kind, i) => ({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: kind === "uniform"
          ? { type: "uniform" }
          : (kind === "out"
            ? { type: "storage" }
            : { type: "read-only-storage" }),
      })),
    });
    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: "main" },
    });

    let inIdx = 0;
    let outIdx = 0;
    const gpuBuffers = [];
    const outBuffers = [];
    for (const kind of plan) {
      if (kind === "uniform") {
        const bytes = uniformsBytes instanceof Uint8Array
          ? uniformsBytes : new Uint8Array(uniformsBytes);
        const buf = device.createBuffer({
          size: bytes.byteLength,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buf, 0, bytes);
        gpuBuffers.push(buf);
      } else if (kind === "in") {
        const src = buffersIn[inIdx++];
        const buf = device.createBuffer({
          size: src.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buf, 0, src);
        gpuBuffers.push(buf);
      } else {
        const spec = buffersOut[outIdx++];
        const buf = device.createBuffer({
          size: Math.max(1, spec.length) * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        outBuffers.push(buf);
        gpuBuffers.push(buf);
      }
    }

    const bindGroup = device.createBindGroup({
      layout,
      entries: gpuBuffers.map((buf, i) => ({ binding: i, resource: { buffer: buf } })),
    });

    let wgX; let wgY = 1;
    if (Array.isArray(dispatchDims)) { wgX = dispatchDims[0] | 0; wgY = (dispatchDims[1] || 1) | 0; }
    else wgX = dispatchDims | 0;
    if (wgX < 1) wgX = 1;
    if (wgX > MAX_WORKGROUPS) {
      throw new Error(`executeKernel: ${wgX} workgroups exceeds ${MAX_WORKGROUPS}`);
    }

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wgX, wgY, 1);
    pass.end();

    const stagings = buffersOut.map((spec, i) => {
      const size = Math.max(1, spec.length) * 4;
      const staging = device.createBuffer({
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      encoder.copyBufferToBuffer(outBuffers[i], 0, staging, 0, size);
      return staging;
    });

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    const results = [];
    try {
      for (let i = 0; i < stagings.length; i++) {
        await stagings[i].mapAsync(GPUMapMode.READ);
        const Src = _DTYPES[buffersOut[i].dtype || "float32"];
        results.push(new Src(stagings[i].getMappedRange()).slice());
        stagings[i].unmap();
      }
    } finally {
      for (const b of gpuBuffers) b.destroy();
      for (const s of stagings) s.destroy();
    }
    return results;
  }
}

// ---- shared singleton ------------------------------------------------------

const _shared = new WebGPUDriver();
let _initPromise = null;

function sharedDriver() { return _shared; }

/** Resolve true when WebGPU is initialized; false -> use CPU backend. */
async function ensureReady() {
  if (_shared.ready) return true;
  if (!_initPromise) {
    _initPromise = _shared.init().catch(() => { _initPromise = null; return null; });
  }
  await _initPromise;
  return _shared.ready;
}

function status() {
  return {
    available: gpuAvailable(),
    ready: _shared.ready,
    adapterVendor: _shared.adapterVendor(),
  };
}

// ---- kernel adapters (existing WGSL, no duplication) -----------------------

/** IndexKernel/IndexKernelI32. params: {n, mode, p0..p2, k, rep, dtype}. */
async function runIndexKernel(params, pat) {
  const p = params || {};
  const n = Math.trunc(Number(p.n));
  const mode = Math.trunc(Number(p.mode));
  const intMode = Boolean(Math.trunc(Number(
    p.int_mode !== undefined ? p.int_mode : (p.dtype === "int32" ? 1 : 0))));
  const hasPat = mode === 3 || mode === 4;
  if (hasPat && intMode) {
    throw new Error("webgpu_driver: int32 tile/repeat pattern not supported");
  }
  const T = intMode ? "i32" : "f32";
  const p0 = p.p0 !== undefined ? p.p0 : 0;
  const p1 = p.p1 !== undefined ? p.p1 : 1;
  const p2 = p.p2 !== undefined ? p.p2 : Math.max(n - 1, 1);
  const k = hasPat ? pat.length : 1;
  const rep = p.rep !== undefined ? Math.trunc(Number(p.rep)) : 1;
  const uniforms = packUniforms([
    [n, "u32"], [mode, "u32"], [p0, T], [p1, T], [p2, T],
    [k, "u32"], [rep, "u32"], [intMode ? 1 : 0, "u32"],
  ]);
  const code = indexWgsl.wgsl({ int_mode: intMode ? 1 : 0, mode });
  const [out] = await _shared.executeKernel(
    hasPat ? code : code,
    hasPat ? [pat] : [],
    [{ length: n, dtype: intMode ? "int32" : "float32" }],
    uniforms, Math.ceil(n / WORKGROUP),
    hasPat ? ["out", "uniform", "in"] : ["out", "uniform"]);
  return out;
}

/** RandomKernel. params: {n, mode, seed, p0, p1}. */
async function runRandomKernel(params) {
  const p = params || {};
  const n = Math.trunc(Number(p.n));
  const uniforms = packUniforms([
    [n, "u32"],
    [Math.trunc(Number(p.mode)), "u32"],
    [Math.trunc(Number(p.seed === undefined ? 42 : p.seed)), "u32"],
    [Number(p.p0 === undefined ? 0 : p.p0), "f32"],
    [Number(p.p1 === undefined ? 1 : p.p1), "f32"],
  ]);
  const [out] = await _shared.executeKernel(
    randomWgsl.WGSL, [],
    [{ length: n, dtype: "float32" }],
    uniforms, Math.ceil(n / WORKGROUP), ["out", "uniform"]);
  return out;
}

/** Map func-codes 0=sin 1=cos 2=exp 3=sqrt 4=log 5=abs 6=neg 7=square. */
async function runMap(funcCode, src) {
  const n = src.length;
  const uniforms = packUniforms([
    [Number(funcCode), "f32"], [0, "f32"], [0, "f32"], [0, "f32"],
  ]);
  const [out] = await _shared.executeKernel(
    mapWgsl.WGSL, [src],
    [{ length: n, dtype: "float32" }],
    uniforms, Math.ceil(n / WORKGROUP), ["in", "out", "uniform"]);
  return out;
}

/** MapBinary ops 0=add 1=sub 2=mul 3=div. b: TypedArray or scalar broadcast. */
async function runMapBinary(opCode, a, b) {
  const bArr = b instanceof Float32Array ? b : Float32Array.of(Number(b));
  const uniforms = packUniforms([
    [Number(opCode), "f32"], [0, "f32"], [0, "f32"], [0, "f32"], [0, "f32"],
  ]);
  const [out] = await _shared.executeKernel(
    mapBinaryWgsl.WGSL, [a, bArr],
    [{ length: a.length, dtype: "float32" }],
    uniforms, Math.ceil(a.length / WORKGROUP), ["in", "in", "out", "uniform"]);
  return out;
}

/** Compare numeric ops 0=gt 1=ge 2=lt 3=le 4=eq 5=ne. Uint32 0/1 mask out. */
async function runCompare(numOp, a, b, useScalarB) {
  let code; let plan; let uniforms;
  if (useScalarB) {
    code = compareWgsl.wgslGenerator({ use_scalar_a: 0, use_scalar_b: 1 });
    plan = ["in", "out", "uniform"];
    uniforms = packUniforms([
      [Number(numOp), "f32"], [0, "f32"],
      [Number(b), "f32"], [0, "f32"], [1, "f32"],
    ]);
  } else {
    code = compareWgsl.wgslGenerator({ use_scalar_a: 0, use_scalar_b: 0 });
    plan = ["in", "in", "out", "uniform"];
    uniforms = packUniforms([
      [Number(numOp), "f32"], [0, "f32"], [0, "f32"], [0, "f32"], [0, "f32"],
    ]);
  }
  const ins = useScalarB ? [a] : [a, b];
  const [out] = await _shared.executeKernel(
    code, ins,
    [{ length: a.length, dtype: "uint32" }],
    uniforms, Math.ceil(a.length / WORKGROUP), plan);
  return out;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    WebGPUDriver, sharedDriver, ensureReady, status, packUniforms,
    gpuAvailable, runIndexKernel, runRandomKernel, runMap, runMapBinary,
    runCompare, WORKGROUP,
  };
}
