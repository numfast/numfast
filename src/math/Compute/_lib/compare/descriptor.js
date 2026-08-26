/**
 * Compare descriptor — JS version.
 * Input count depends on scalar mode.
 * op: string ("gt"/"ge"/"lt"/"le"/"eq"/"ne") or numeric code
 *     (0=gt, 1=ge, 2=lt, 3=le, 4=eq, 5=ne). Uniform "op" is always numeric.
 */

"use strict";

const OP_NAMES = { 0: "gt", 1: "ge", 2: "lt", 3: "le", 4: "eq", 5: "ne" };
const OP_CODES = { gt: 0, ge: 1, lt: 2, le: 3, eq: 4, ne: 5 };

function describe(params) {
  params = params || {};
  const op = params.op || "gt";
  const code = typeof op === "number" ? Number(op) : (OP_CODES[op] !== undefined ? OP_CODES[op] : 0);
  const inputs = [];
  if (!Number(params.use_scalar_a || 0)) {
    inputs.push({ name: "a", dtype: "float" });
  }
  if (!Number(params.use_scalar_b || 0)) {
    inputs.push({ name: "b", dtype: "float" });
  }
  return {
    inputs: inputs,
    outputs: [{ dtype: "uint32", template: "compare" }],
    workspace: [],
    uniforms: {
      op: code,
      scalar_a: Number(params.scalar_a || 0.0),
      scalar_b: Number(params.scalar_b || 0.0),
      use_scalar_a: Number(params.use_scalar_a || 0),
      use_scalar_b: Number(params.use_scalar_b || 0),
    },
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { describe, OP_NAMES, OP_CODES };
}
