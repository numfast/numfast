/**
 * _runner.js — shared kernel execution helper for the public API layer.
 *
 * CPU execution path, same pattern as test/test_npm.js CHECK 3:
 *   registry entry -> describe(params) -> ExecutionPlan.allocateOutputs(n)
 *   -> ExecutionContext -> entry.cpu(ctx) -> outputs[0] TypedArray.
 *
 * Runtime.execute() sizes no-input kernels to 1 element (frozen behaviour),
 * so creation-style calls allocate outputs explicitly here.
 */

"use strict";

const { ExecutionPlan, ExecutionContext } =
  require("../core/Runtime/_lib/mod_iface");
const { register_compute_kernels } = require("../math/Compute/Compute");

let _registry = null;

function kernelEntry(alias) {
  if (!_registry) {
    _registry = new Map(
      register_compute_kernels().map((k) => [k[0], k]));
  }
  const entry = _registry.get(alias);
  if (!entry) throw new Error(`Unknown kernel: ${alias}`);
  return entry;
}

/**
 * Run one kernel on the CPU path.
 * params         — descriptor params (mutated in place by describe, idempotent)
 * inputs         — array of TypedArrays bound positionally to plan inputs
 * outLen         — explicit output length
 * Returns plan.outputs[0].view._data (Float64Array for "float" dtype,
 * Int32Array for "int32", Uint32Array for "uint32").
 */
function runKernel(alias, params, inputs, outLen) {
  const entry = kernelEntry(alias);
  const plan = new ExecutionPlan(entry[1](params || {}));
  plan.allocateOutputs(outLen);
  const ctx = new ExecutionContext(plan, inputs || []);
  entry[2](ctx);
  return plan.outputs[0].view._data;
}

module.exports = { runKernel, kernelEntry };
