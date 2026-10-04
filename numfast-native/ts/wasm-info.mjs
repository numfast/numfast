// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//
// WHY THIS FILE EXISTS
//
// `WebAssembly.Module.exports()` reports names and kinds but NOT signatures,
// so a host cannot tell which of the 85 kernels cross the JavaScript BigInt
// boundary without either parsing the binary or probing all 553 call sites.
// This reads the four sections that answer it -- type, import, function,
// export -- plus global/memory/data, with no dependency.
//
// Sections are length-delimited and the cursor is reset to the declared end
// after each one, so a wrong guess INSIDE a section cannot desynchronise the
// next: a parse failure is contained, never silently corrupting later reads.
//
// The 20-of-85 i64 figure quoted in the README and asserted by
// test/abi.test.mjs is computed here, from the bytes, on every build.
import { readFileSync } from "node:fs";

const VT = { 0x7f: "i32", 0x7e: "i64", 0x7d: "f32", 0x7c: "f64", 0x70: "funcref", 0x6f: "externref" };

export function readWasm(bytes) {
  const buf = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes);
  if (buf.readUInt32LE(0) !== 0x6d736100) throw new Error("not a wasm module (bad magic)");
  if (buf.readUInt32LE(4) !== 1) throw new Error("unsupported wasm version");
  let p = 8;
  const u8 = () => buf[p++];
  const uleb = () => { let r = 0, s = 0, b; do { b = buf[p++]; r += (b & 0x7f) * 2 ** s; s += 7; } while (b & 0x80); return r; };
  const sleb = () => { let r = 0, s = 0, b; do { b = buf[p++]; r |= (b & 0x7f) << s; s += 7; } while (b & 0x80); if (s < 32 && (b & 0x40)) r |= -(1 << s); return r; };
  const str = () => { const n = uleb(); const s = buf.toString("utf8", p, p + n); p += n; return s; };
  const skip = (n) => { p += n; };

  const out = { types: [], imports: [], funcTypes: [], globals: [], memories: [], exports: [], dataSegments: [] };

  while (p < buf.length) {
    const id = u8();
    const size = uleb();
    const end = p + size;
    switch (id) {
      case 1: { // type
        const n = uleb();
        for (let i = 0; i < n; i++) {
          if (u8() !== 0x60) throw new Error(`type ${i}: not a func type`);
          const np = uleb(); const params = []; for (let j = 0; j < np; j++) params.push(VT[u8()]);
          const nr = uleb(); const results = []; for (let j = 0; j < nr; j++) results.push(VT[u8()]);
          out.types.push({ params, results });
        }
        break;
      }
      case 2: { // import
        const n = uleb();
        for (let i = 0; i < n; i++) {
          const module = str(); const name = str(); const kind = u8();
          if (kind === 0) out.imports.push({ module, name, kind: "func", type: uleb() });
          else if (kind === 1) { u8(); const lim = u8(); const mn = uleb(); if (lim) uleb(); out.imports.push({ module, name, kind: "table" }); }
          else if (kind === 2) { const lim = u8(); const mn = uleb(); if (lim) uleb(); out.imports.push({ module, name, kind: "memory" }); }
          else { const t = VT[u8()]; u8(); out.imports.push({ module, name, kind: "global", vt: t }); }
        }
        break;
      }
      case 3: { const n = uleb(); for (let i = 0; i < n; i++) out.funcTypes.push(uleb()); break; }
      case 5: { const n = uleb(); for (let i = 0; i < n; i++) { const lim = u8(); const mn = uleb(); const mx = lim ? uleb() : null; out.memories.push({ min: mn, max: mx }); } break; }
      case 6: { // global
        const n = uleb();
        for (let i = 0; i < n; i++) {
          const vt = VT[u8()]; const mut = u8() === 1;
          const op = u8(); let value = null;
          if (op === 0x41) value = sleb();
          else if (op === 0x42) value = Number(sleb());
          else if (op === 0x43) { value = buf.readFloatLE(p); p += 4; }
          else if (op === 0x44) { value = buf.readDoubleLE(p); p += 8; }
          else if (op === 0x23) value = { global: uleb() };
          else throw new Error(`global ${i}: unsupported init opcode 0x${op.toString(16)}`);
          if (u8() !== 0x0b) throw new Error(`global ${i}: init expr not terminated`);
          out.globals.push({ vt, mut, value });
        }
        break;
      }
      case 7: { // export
        const n = uleb();
        for (let i = 0; i < n; i++) {
          const name = str(); const kind = ["func", "table", "memory", "global"][u8()]; const index = uleb();
          out.exports.push({ name, kind, index });
        }
        break;
      }
      case 11: { // data
        const n = uleb();
        for (let i = 0; i < n; i++) {
          const flags = uleb();
          let offset = null;
          if (flags === 0 || flags === 2) { const op = u8(); if (op === 0x41) offset = sleb(); else if (op === 0x42) offset = Number(sleb()); else if (op === 0x23) offset = { global: uleb() }; else throw new Error(`data ${i}: unsupported offset`); }
          const len = uleb();
          out.dataSegments.push({ offset, bytes: len });
          skip(len);
        }
        break;
      }
      default: break; // custom/code/element/start/datacount: not needed here
    }
    p = end;
  }
  return out;
}

/** Per-export signature table, plus the two facts the docs quote. */
export function describe(bytes) {
  const m = readWasm(bytes);
  const sigs = {};
  const i64 = [];
  for (const e of m.exports) {
    if (e.kind !== "func") continue;
    const t = m.types[m.funcTypes[e.index]];
    if (!t) throw new Error(`export ${e.name}: no func type`);
    sigs[e.name] = { params: t.params, results: t.results };
    if (t.params.includes("i64") || t.results.includes("i64")) i64.push(e.name);
  }
  const funcExports = m.exports.filter((e) => e.kind === "func");
  return {
    bytes: bytes.length,
    exportCount: m.exports.length,
    funcCount: funcExports.length,
    funcExports: funcExports.map((e) => e.name).sort(),
    memoryExports: m.exports.filter((e) => e.kind === "memory").map((e) => e.name),
    importCount: m.imports.length,
    imports: m.imports,
    i64Exports: i64.sort(),
    i64Count: i64.length,
    signatures: sigs,
    initialMemoryPages: m.memories.length ? m.memories[0].min : null,
    stackPointer: m.globals.length ? m.globals[0].value : null,
    dataSegments: m.dataSegments,
/** Highest byte covered by a declared data segment. Reported for reference
 *  only: it UNDERSTATES the module's real footprint, because `.rodata` that
 *  wasm-ld folds into a passive/merged segment is not visible here. The
 *  authoritative figure for buffer placement is the runtime probe in
 *  bridge.ts (`staticDataEnd`). */
staticDataEnd: m.dataSegments.reduce((a, d) => Math.max(a, (d.offset || 0) + d.bytes), 0),
  };
}

if (process.argv[1] && process.argv[1].endsWith("wasm-info.mjs")) {
  const d = describe(readFileSync(process.argv[2]));
  process.stdout.write(JSON.stringify({ ...d, signatures: undefined, dataSegments: d.dataSegments.length }, null, 2) + "\n");
  process.stdout.write("i64: " + d.i64Exports.join(", ") + "\n");
}