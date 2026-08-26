/**
 * mod_iface.js — JS equivalent of Python's Runtime._lib.mod_iface
 * 
 * ExecutionContext, InputSlot, OutputSlot, ExecutionPlan
 * for use with APP Builder JS extensions.
 */

"use strict";

class BufferView {
  constructor(data) {
    this._data = data;
  }
  length() { return this._data.length; }
  read(i) { return this._data[i]; }
  write(i, v) { this._data[i] = v; }
}

class InputSlot {
  constructor(spec) {
    this.name = spec.name;
    this.dtype = spec.dtype || "float";
    this.view = null;
  }
  bind(data) {
    this.view = new BufferView(data);
  }
}

class OutputSlot {
  constructor(spec) {
    this.dtype = spec.dtype || "float";
    this.template = spec.template || "";
    this.view = null;
  }
  allocate(size) {
    let ctor;
    if (this.dtype === "uint32") ctor = Uint32Array;
    else if (this.dtype === "int32") ctor = Int32Array;
    else if (this.dtype === "int64") ctor = BigInt64Array;
    else ctor = Float64Array; // float, float64
    this.view = new BufferView(new ctor(size));
  }
}

class ExecutionPlan {
  constructor(spec) {
    this.inputs = (spec.inputs || []).map(s => new InputSlot(s));
    this.outputs = (spec.outputs || []).map(s => new OutputSlot(s));
    this.workspace = spec.workspace || [];
    this.uniforms = spec.uniforms || {};
  }
  allocateOutputs(size) {
    for (const out of this.outputs) out.allocate(size);
  }
  bindInputs(dataArrays) {
    for (let i = 0; i < this.inputs.length; i++) {
      if (i < dataArrays.length) this.inputs[i].bind(dataArrays[i]);
    }
  }
}

class ExecutionContext {
  constructor(plan, dataArrays) {
    this.inputs = plan.inputs.map((inp, i) => {
      const bv = new BufferView(i < dataArrays.length ? dataArrays[i] : new Float64Array(0));
      return { view: bv };
    });
    this.outputs = plan.outputs.map(out => {
      return { view: out.view };
    });
    this.uniforms = plan.uniforms || {};
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { BufferView, InputSlot, OutputSlot, ExecutionPlan, ExecutionContext };
}
