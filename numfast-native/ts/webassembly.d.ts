// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//
// Minimal ambient declarations for the WebAssembly JS API.
//
// TypeScript ships the `WebAssembly` namespace only in `lib.dom.d.ts`, so the
// alternatives are (a) add "DOM" to `lib`, which would also let browser-only
// globals (`document`, `window`, `fetch`) typecheck inside a Node package, or
// (b) declare the seven members actually used here. (b) is chosen: it keeps
// the package honest about running in Node, and if a future member is needed
// the compiler says so instead of silently inheriting a browser assumption.
//
// WHAT IS DECLARED, AND WHY EACH IS NOT OPTIONAL:
//   Memory.grow      DETACHES every existing TypedArray view. This is the
//                    single most dangerous fact about the API and the reason
//                    `Bridge.ensure` re-fetches views after every growth.
//   RuntimeError     the trap channel; distinguishable from other errors.
//   instantiate      the only import path used: this module imports NOTHING,
//                    which is what makes the .wasm portable.

declare namespace WebAssembly {
  interface Memory {
    readonly buffer: ArrayBuffer;
    grow(delta: number): number;
  }

  interface Module {}
  interface Instance {
    readonly exports: Exports;
  }
  interface Exports {
    [name: string]: ExportValue;
  }
  type ExportValue = Memory | Function | Global | Table;
  interface Global { value: unknown; }
  interface Table {}

  type ImportValue = Memory | Function | Global | Table;

  class CompileError extends Error {}
  class LinkError extends Error {}
  class RuntimeError extends Error {}

  interface WebAssemblyInstantiatedSource {
    instance: Instance;
    module: Module;
  }

  function instantiate(
    bytes: BufferSource,
    importObject?: Imports,
  ): Promise<WebAssemblyInstantiatedSource>;
  function compile(bytes: BufferSource): Promise<Module>;
  function validate(bytes: BufferSource): boolean;

  interface Imports {
    [module: string]: { [name: string]: ImportValue };
  }
}