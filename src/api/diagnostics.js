/**
 * diagnostics.js — deviceInfo (JS_PARITY_SPEC §2: diagnostics row).
 *
 * backend reflects the shared WebGPU driver state:
 *   "webgpu" after a successful init (see webgpu_driver.ensureReady),
 *   "cpu" otherwise (Node.js, no navigator.gpu, or init failure).
 */

"use strict";

const VERSION = "1.0.0-alpha.2";
const KERNEL_COUNT = 33;
const { status } = require("./webgpu_driver");

function deviceInfo() {
  const s = status();
  return {
    backend: s.ready ? "webgpu" : "cpu",
    adapter: s.adapterVendor || "",
    version: VERSION,
    kernels: KERNEL_COUNT,
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { deviceInfo };
}
