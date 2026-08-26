/**
 * Compute Extension — JS version.
 * Registers all GPU primitives as kernels.
 * 
 * Registry format:
 *   [alias, descPath, cpuPath, wgslPath, descKey?, cpuKey?, wgslKey?]
 *   descKey defaults to 'describe', cpuKey defaults to 'cpu'
 *   When wgslKey is provided, use it directly.
 *   When wgslKey is absent, use fallback chain: WGSL || wgslGenerator || wgsl
 */

"use strict";

const kernels = [];

function _load() {
  if (kernels.length > 0) return kernels;
  
  const registry = [
    // L0: Memory
    ["Gather",      "./_lib/gather/descriptor",        "./_lib/gather/cpu",         "./_lib/gather/wgsl"],
    ["Scatter",     "./_lib/scatter/descriptor",       "./_lib/scatter/cpu",        "./_lib/scatter/wgsl"],
    ["Shift",       "./_lib/shift/descriptor",         "./_lib/shift/cpu",          "./_lib/shift/wgsl"],
    ["Fill",        "./_lib/fill/descriptor",          "./_lib/fill/cpu",           "./_lib/fill/wgsl"],
    // L1: Logic
    ["Compare",     "./_lib/compare/descriptor",       "./_lib/compare/cpu",        "./_lib/compare/wgsl"],
    ["Mask",        "./_lib/mask/descriptor",          "./_lib/mask/cpu",           "./_lib/mask/wgsl"],
    ["Clamp",       "./_lib/clamp/descriptor",         "./_lib/clamp/cpu",          "./_lib/clamp/wgsl"],
    // L2: Element-wise
    ["Map",         "./_lib/map/descriptor",           "./_lib/map/cpu",            "./_lib/map/wgsl"],
    ["MapBinary",   "./_lib/map_binary/descriptor",    "./_lib/map_binary/cpu",     "./_lib/map_binary/wgsl"],
    // L3: Reduction
    ["Reduce",      "./_lib/reduce/descriptor",        "./_lib/reduce/cpu",         "./_lib/reduce/wgsl"],
    ["ScanLocal",   "./_lib/scan/descriptor",          "./_lib/scan/cpu",           "./_lib/scan/wgsl",    "describe_local",  "cpu_local",  "WGSL_LOCAL"],
    ["ScanTotals",  "./_lib/scan/descriptor",          "./_lib/scan/cpu",           "./_lib/scan/wgsl",    "describe_totals", "cpu_totals", "WGSL_TOTALS"],
    ["ScanFinal",   "./_lib/scan/descriptor",          "./_lib/scan/cpu",           "./_lib/scan/wgsl",    "describe_final",  "cpu_final",  "WGSL_FINAL"],
    ["ArgMinMax",   "./_lib/argminmax/descriptor",     "./_lib/argminmax/cpu",      "./_lib/argminmax/wgsl"],
    // L4: Window
    ["RollingSum",    "./_lib/rolling_sum/descriptor",    "./_lib/rolling_sum/cpu",    "./_lib/rolling_sum/wgsl"],
    ["RollingMin",    "./_lib/rolling_min/descriptor",    "./_lib/rolling_min/cpu",    "./_lib/rolling_min/wgsl"],
    ["RollingMax",    "./_lib/rolling_max/descriptor",    "./_lib/rolling_max/cpu",    "./_lib/rolling_max/wgsl"],
    ["RollingStdDev", "./_lib/rolling_stddev/descriptor", "./_lib/rolling_stddev/cpu", "./_lib/rolling_stddev/wgsl"],
    // L5: Stateful
    ["StateKernel_Single", "./_lib/state_kernel/descriptor", "./_lib/state_kernel/cpu", "./_lib/state_kernel/wgsl", "describe", "cpu", "WGSL_SINGLE"],
    ["StateKernel_Dual",   "./_lib/state_kernel/descriptor", "./_lib/state_kernel/cpu", "./_lib/state_kernel/wgsl", "describe", "cpu", "WGSL_DUAL"],
    ["StateKernel_ST",     "./_lib/state_kernel/descriptor", "./_lib/state_kernel/cpu", "./_lib/state_kernel/wgsl", "describe", "cpu", "WGSL_SUPERTREND"],
    ["TrueRange",  "./_lib/true_range/descriptor",  "./_lib/true_range/cpu",  "./_lib/true_range/wgsl"],
    // L6: Structural
    ["Sort",       "./_lib/sort/descriptor",        "./_lib/sort/cpu",         "./_lib/sort/wgsl",        "describe_sort", "cpu_sort", "WGSL"],
    ["Histogram",  "./_lib/histogram/descriptor",   "./_lib/histogram/cpu",    "./_lib/histogram/wgsl",   "describe",      "cpu_histogram", "WGSL"],
    ["Combine",    "./_lib/combine/descriptor",     "./_lib/combine/cpu",      "./_lib/combine/wgsl"],
    // L7: Numerical
    ["BitReverse", "./_lib/fft/descriptor",         "./_lib/fft/cpu",          "./_lib/fft/wgsl",         "describe_bitreverse", "cpu_bitreverse", "WGSL_BITREVERSE"],
    ["FftStage",   "./_lib/fft/descriptor",         "./_lib/fft/cpu",          "./_lib/fft/wgsl",         "describe_stage", "cpu_stage", "WGSL_STAGE"],
    ["MatMul",     "./_lib/matmul/descriptor",      "./_lib/matmul/cpu",       "./_lib/matmul/wgsl",      "describe_matmul", "cpu_matmul", "WGSL"],
    // Logical
    ["LogicalAnd", "./_lib/logical_and/descriptor", "./_lib/logical_and/cpu",  "./_lib/logical_and/wgsl"],
    ["LogicalOr",  "./_lib/logical_or/descriptor",  "./_lib/logical_or/cpu",   "./_lib/logical_or/wgsl"],
    // Expression
    ["Expression", "./_lib/expression/descriptor",  "./_lib/expression/cpu",   "./_lib/expression/wgsl",  "describe", "cpu", "wgsl_generator"],
    // Creation
    ["RandomKernel", "./_lib/random_kernel/descriptor", "./_lib/random_kernel/cpu", "./_lib/random_kernel/wgsl"],
    // IndexKernel + I32: same describe/cpu/wgsl (dtype via params.dtype ->
    // uniform int_mode), separate aliases mirror Python Compute.py registration
    ["IndexKernel",    "./_lib/index_kernel/descriptor", "./_lib/index_kernel/cpu", "./_lib/index_kernel/wgsl"],
    ["IndexKernelI32", "./_lib/index_kernel/descriptor", "./_lib/index_kernel/cpu", "./_lib/index_kernel/wgsl"],
  ];

  for (const entry of registry) {
    const [alias, descPath, cpuPath, wgslPath, descKey, cpuKey, wgslKey] = entry;
    let describe = () => ({ inputs: [], outputs: [], workspace: [], uniforms: {} });
    let cpuFn = () => {};
    let wgslSrc = null;

    try {
      const dMod = require(descPath);
      const key = descKey || 'describe';
      if (typeof dMod[key] === 'function') describe = dMod[key];
    } catch (e) { /* not available */ }

    try {
      const cMod = require(cpuPath);
      const key = cpuKey || 'cpu';
      if (typeof cMod[key] === 'function') cpuFn = cMod[key];
    } catch (e) { /* not available */ }

    if (wgslPath) {
      try {
        const wMod = require(wgslPath);
        if (wgslKey) {
          wgslSrc = wMod[wgslKey] || null;
        } else {
          wgslSrc = wMod.WGSL || wMod.wgslGenerator || wMod.wgsl || null;
        }
      } catch (e) { /* not available */ }
    }

    kernels.push([alias, describe, cpuFn, wgslSrc]);
  }
  return kernels;
}

function register_all(runtime) {
  const rt = runtime || { register_kernel: () => {} };
  const reg = rt.register_kernel || runtime;
  for (const [alias, describe, cpuFn, wgslSrc] of _load()) {
    reg(alias, { describe, cpu: cpuFn, wgsl: wgslSrc });
  }
}

function register_compute_kernels() {
  return _load();
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { register_all, register_compute_kernels };
  module.exports.PUBLIC = { register_compute_kernels };
}
