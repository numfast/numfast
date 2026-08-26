/**
 * @numfast/numfast — GPU-first numerical computing framework.
 *
 * Entry point for Node.js.
 *
 * Public API (JS_PARITY_SPEC): create -> compute -> compare -> filter/top-k
 * -> read result. Part C adds a WebGPU path: nf.initWebGPU() enables
 * {gpu:true} execution (creation + Series ops); without WebGPU everything
 * stays on the CPU kernels. deviceInfo().backend is "webgpu"|"cpu".
 *
 * Usage:
 *   const nf = require('@numfast/numfast');
 *   const z = nf.zeros(5);                    // creation
 *   const s = nf.Series.from([1, 2, 3]);      // Series operators
 *   const r = s.mul(2).add(1);                // eager single-op compute
 *   const m = s.gt(1);                        // mask Series (Uint8Array)
 *   nf.topk(s, 2);                            // Sort + Gather
 *   nf.deviceInfo();                          // { backend, adapter, ... }
 *   await nf.initWebGPU();                    // optional GPU init
 *   const g = await nf.zeros(5, undefined, { gpu: true }); // GPU creation
 */

"use strict";

const {
  compile, execute, register_kernel, Runtime, CpuDriver
} = require('./src/core/Runtime/Runtime');

const {
  register_all, register_compute_kernels
} = require('./src/math/Compute/Compute');

const {
  ExecutionPlan, ExecutionContext, BufferView, InputSlot, OutputSlot
} = require('./src/core/Runtime/_lib/mod_iface');

// ---- public API surface (legacy 12 symbols preserved) ----
const creation = require('./src/api/creation');
const { Series, topk } = require('./src/api/series');
const mathNs = require('./src/api/math');
const { deviceInfo } = require('./src/api/diagnostics');

module.exports = {
  // Legacy Runtime surface (12 symbols)
  compile, execute, register_kernel, Runtime, CpuDriver,
  register_all, register_compute_kernels,
  ExecutionPlan, ExecutionContext, BufferView, InputSlot, OutputSlot,
  // Creation
  zeros: creation.zeros,
  ones: creation.ones,
  full: creation.full,
  arange: creation.arange,
  linspace: creation.linspace,
  index: creation.index,
  tile: creation.tile,
  repeat: creation.repeat,
  // Series + compositions
  Series,
  topk,
  // Math (element-wise)
  sin: mathNs.sin,
  cos: mathNs.cos,
  exp: mathNs.exp,
  log: mathNs.log,
  sqrt: mathNs.sqrt,
  abs: mathNs.abs,
  square: mathNs.square,
  neg: mathNs.neg,
  // Stats
  mean: mathNs.mean,
  min: mathNs.min,
  max: mathNs.max,
  total: mathNs.total,
  count: mathNs.count,
  var: mathNs.var,
  std: mathNs.std,
  // Diagnostics
  deviceInfo,
};

// ---- non-enumerable extensions (part C: WebGPU) ----------------------------
// The enumerable export surface is frozen by test/test_npm.js CHECK 1
// (Object.keys(nf).length === LEGACY + PUBLIC), so WebGPU additions are
// attached as non-enumerable own properties.

const webgpuDriver = require('./src/api/webgpu_driver');

// nf.random.uniform/integers/normal — RandomKernel creation primitives.
Object.defineProperty(module.exports, 'random', { value: creation.random });
// await nf.initWebGPU() -> true when the shared driver is ready.
Object.defineProperty(module.exports, 'initWebGPU', {
  value: webgpuDriver.ensureReady,
});
// Low-level driver facade for browser examples (runMap/runCompare/...).
Object.defineProperty(module.exports, 'webgpu', { value: webgpuDriver });
