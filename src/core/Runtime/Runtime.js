/**
 * Runtime Extension — JS version.
 * Execution layer: compile, execute, register_kernel.
 * 
 * Mirrors Python Runtime._lib.runtime API:
 *   compile(jobs) -> tasks with plans
 *   execute(tasks, dataMap) -> { name: TypedArray }
 *   register_kernel(alias, opts)
 */

"use strict";

const { ExecutionPlan } = require("../Runtime/_lib/mod_iface");

// In-memory kernel registry
const _kernels = {};

function _getPlan(alias, params) {
  const entry = _kernels[alias];
  if (!entry) throw new Error(`Unknown kernel: ${alias}`);
  const raw = entry.describe(params || {});
  return new ExecutionPlan(raw);
}

function _executePlan(alias, plan, dataArrays) {
  const entry = _kernels[alias];
  const { ExecutionContext } = require("../Runtime/_lib/mod_iface");
  const ctx = new ExecutionContext(plan, dataArrays);
  entry.cpu(ctx);
  return plan.outputs.map(o => o.view._data);
}

/**
 * Compile jobs into executable tasks.
 * Jobs: [{op, inputs, params, out}]
 * Returns tasks with plans (no allocation — sized at execute time).
 */
function compile(jobs) {
  return jobs.map(job => {
    const plan = _getPlan(job.op, job.params);
    return { ...job, plan };
  });
}

/**
 * Execute compiled tasks with data from dataMap.
 * dataMap: { name -> TypedArray }
 * Results flow between tasks via results object.
 * Outputs are sized from actual input data (matching Python Runtime).
 */
function execute(tasks, dataMap) {
  const results = {};
  for (const task of tasks) {
    const dataArrays = (task.inputs || []).map(name => {
      if (dataMap[name] !== undefined) return dataMap[name];
      if (results[name] !== undefined) return results[name];
      throw new Error(`Input not found: ${name}`);
    });

    // Size outputs from actual input data
    const size = dataArrays.reduce((max, arr) => Math.max(max, arr.length), 0);
    task.plan.allocateOutputs(size || 1);

    const outputs = _executePlan(task.op, task.plan, dataArrays);
    results[task.out] = outputs[0];
  }
  return results;
}

function register_kernel(alias, opts) {
  opts = opts || {};
  _kernels[alias] = {
    describe: opts.describe || (() => ({ inputs: [], outputs: [], workspace: [], uniforms: {} })),
    cpu: opts.cpu || (() => {}),
    wgsl: opts.wgsl || null,
  };
}

function Runtime() {
  return { compile, execute, register_kernel };
}

function CpuDriver() {
  return { executePlan: _executePlan };
}

const __PUBLIC = { compile, execute, register_kernel, Runtime, CpuDriver, CpuDriver: CpuDriver };

if (typeof module !== "undefined" && module.exports) {
  module.exports = { compile, execute, register_kernel, Runtime, CpuDriver, _kernels };
  module.exports.PUBLIC = __PUBLIC;
}
