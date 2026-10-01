import { readFileSync } from "node:fs";
import { BASE, alloc, ensureMem, guardFill, blit, blitBytes, fileView, loadWasm } from "./wasm_mem.mjs";
const wasm = "C:/App/numfast/numfast/numfast-native/target/wasm32-unknown-unknown/release/numfast_native.wasm";
const T = "C:/Users/Mikech/AppData/Local/Temp/";
const tag = process.argv[2] ?? "p300";
const n = Number(process.argv[3] ?? "300");
const inst = await loadWasm(wasm);
const mem = inst.exports.memory;
const data = new Uint8Array(readFileSync(T + tag + ".bin"));
const offs = fileView(T + tag + ".i32", Int32Array);
let cur = BASE;
const [dO, c1] = alloc(cur, data.length, 4);
const [oO, c2] = alloc(c1, (n + 1) * 4, 4);
const [uO, need] = alloc(c2, n * 4, 4);
let u8 = ensureMem(mem, need);
u8 = guardFill(mem);
blitBytes(u8, data, dO);
blit(u8, offs, oO);
u8.fill(0xaa, uO, uO + n * 4); // prefill: distinguishes kernel-cleared (0) from untouched (aa)
console.log("debugRowFirst", Number(inst.exports.nf_text_debug_row(dO, data.length, oO, n)));
const rc = inst.exports.nf_text_length(dO, data.length, oO, n, uO);
console.log("rc", rc);
console.log("debugRow", Number(inst.exports.nf_text_debug_row(dO, data.length, oO, n)));
const mu8 = new Uint8Array(mem.buffer);
console.log("memrow0", Array.from(mu8.slice(dO, dO + 8)).map((b) => b.toString(16).padStart(2, "0")).join(" "));
console.log("filerow0", Array.from(data.slice(0, 8)).map((b) => b.toString(16).padStart(2, "0")).join(" "));
console.log("memoffs0-2", new Int32Array(mem.buffer.slice(oO, oO + 12)).join(","));
console.log("dO", dO.toString(16), "oO", oO.toString(16));
const mo2 = new Int32Array(mem.buffer.slice(oO, oO + (n + 1) * 4));
console.log("memoffs183-187", Array.from(mo2.slice(183, 188)).join(","));
console.log("fileoffs183-187", Array.from(offs.slice(183, 188)).join(","));
// readback: what does the kernel's data window actually hold at row 184?
const rb = Array.from(mu8.slice(dO + 1238, dO + 1248)).map((b) => b.toString(16).padStart(2, "0")).join(" ");
console.log("memrow184", rb);
console.log("filerow184", Array.from(data.slice(1238, 1248)).map((b) => b.toString(16).padStart(2, "0")).join(" "));
const lens = new Int32Array(mem.buffer.slice(uO, uO + n * 4));
const td = new TextDecoder("utf-8", { fatal: true });
let shown = 0;
for (let i = 0; i < n && shown < 5; i++) {
  const s = offs[i], e = offs[i + 1];
  let exp;
  try {
    exp = [...td.decode(data.slice(s, e))].length;
  } catch {
    exp = "INVALID";
  }
  if (lens[i] !== exp) {
    console.log("mismatch row", i, "offs", s, e, "got", lens[i], "exp", exp,
      JSON.stringify(Buffer.from(data.slice(s, e)).toString("utf8").slice(0, 20)));
    shown++;
  }
}
console.log("scan done");
const raw = new Uint8Array(mem.buffer).slice(uO + 170 * 4, uO + 182 * 4);
console.log("raw170-181", Buffer.from(raw).toString("hex"));
console.log("uO", uO.toString(16), "need", need.toString(16));
