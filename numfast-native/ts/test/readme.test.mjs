// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//
// THE DOCUMENTED ssspCsr RESULT, CHECKED AGAINST THE ARTEFACT.
//
// WHAT THIS PINS, AND WHY IT HAD TO BE WRITTEN. The README's quick-start
// example claimed `Uint32Array [ 0, 10, 30, Infinity-as-0xffffffff ]` for the
// graph
//
//     indptr [0,1,2,3]  indices [1,2,0]  weights [10,20,30]  source 0
//
// The kernel returns `[0, 10, 30]`. The claim was wrong in BOTH directions at
// once: it invented a fourth lane, and it called a graph in which every vertex
// is reachable "one with an unreachable one". V is `indptr.length - 1` = 3, and
// 0 -> 1 -> 2 -> 0 reaches all three. A reader checking the output by hand
// would have found the comment wrong and had no way to tell which half was the
// typo.
//
// It survived a release because the suite asserted what the DOCUMENTATION said
// and never compared it to the kernel: `test/abi.test.mjs` pins counts and
// signatures, `parity.test.mjs` pins kernels against Python fixtures and oracles,
// and `nf_sssp_csr` is in the not-fixtured list, so the quick-start example had
// no test at all. A documented example that nothing executes is a comment.
//
// THE RULE, applied to every place the same claim appears: every documented
// `ssspCsr` result in these files is EXTRACTED FROM THE PROSE and compared
// against the kernel. Not retyped here. If the README drifts, this fails and
// names the file and the line; if the kernel changes, this fails too, and the
// fix is to decide which of the two was right rather than to edit a number.
//
// The three sites are checked because all three are the same claim:
//   numfast-native/ts/README.md        the quick-start comment (the wrong one)
//   numfast-native/ts/README.md        the browser table row
//   docs/INSTALL.md                    the minimal example + the verify snippet
//
// The oracle discipline is the same as the rest of this directory: the graph
// is read out of the prose, so a wrong GRAPH cannot be laundered by a right
// answer, and an independent Dijkstra (parity.test.mjs's, re-derived here in a
// few lines rather than shared, so the two cannot drift together) says what the
// answer must be for the graph the documentation actually describes.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test, before } from "node:test";
import { fileURLToPath } from "node:url";

import { bridgeFor } from "./artifact.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const TS = join(HERE, "..");
const REPO = join(TS, "..", "..");

/** Every file that documents an `ssspCsr` result. Adding one is the point. */
const DOC_SITES = [
  join(TS, "README.md"),
  join(REPO, "docs", "INSTALL.md"),
];

/**
 * The quick-start graph, parsed out of the prose that documents it.
 *
 * Deliberately not typed in here: the numbers a reader sees are the numbers
 * this runs, so a documentation typo in the INPUTS fails this test instead of
 * producing a plausible answer for a graph nobody described.
 */
function documentedGraph(text) {
  // `new Uint32Array([...]),<spaces>// indptr` and its two siblings, plus the
  // `0);<spaces>// source` line. Anchored on the trailing label so the three
  // arrays cannot be read in the wrong order.
  const labelled = (label) => {
    const re = new RegExp(
      `new Uint32Array\\(\\[([^\\]]*)\\]\\)\\s*,?\\s*//\\s*${label}\\b`);
    const m = text.match(re);
    assert.ok(m, `the documentation no longer spells out the ${label} array ` +
      `as "new Uint32Array([...]) // ${label}"; this test reads the graph out ` +
      `of the prose and cannot follow it any more`);
    return m[1].split(",").map((s) => s.trim()).filter((s) => s.length)
      .map(Number);
  };
  const src = text.match(/\n\s*(\d+)\);\s*\/\/\s*source\b/);
  assert.ok(src, "the documentation no longer spells out the source vertex as " +
    `"<n>); // source"; this test reads the graph out of the prose`);
  const indptr = labelled("indptr");
  const indices = labelled("indices");
  const weights = labelled("weights");
  return { indptr, indices, weights, source: Number(src[1]) };
}

/** O(V^2) Dijkstra over a CSR graph -- the answer, computed here. */
function dijkstraRef({ indptr, indices, weights, source }) {
  const v = indptr.length - 1;
  const dist = new Uint32Array(v).fill(0xffffffff);
  const done = new Uint8Array(v);
  dist[source] = 0;
  for (;;) {
    let u = -1;
    let best = 0xffffffff;
    for (let i = 0; i < v; i++) {
      if (!done[i] && dist[i] < best) { best = dist[i]; u = i; }
    }
    if (u < 0) break;
    done[u] = 1;
    for (let e = indptr[u], end = indptr[u + 1]; e < end; e++) {
      const w = indices[e];
      const nd = best + weights[e];
      if (nd < dist[w]) dist[w] = nd;
    }
  }
  return [...dist];
}

/** `"Infinity-as-0xffffffff"`, `"Q_INF"`, `4294967295` -> the u32 it means. */
function laneValue(tok, file, line) {
  if (/Infinity|Q_INF|4294967295|0xffffffff/i.test(tok)) return 0xffffffff;
  const n = Number(tok);
  assert.ok(Number.isInteger(n), `${file}:${line}: cannot read "${tok}" as a lane ` +
    `of a documented ssspCsr result`);
  return n;
}

/**
 * Every `ssspCsr(...)` result a file states, as { line, lanes, values }.
 *
 * Three shapes, and ONLY these three, so an INPUT array can never be read as a
 * RESULT -- an input line carries a `// <label>` comment naming which array it
 * is, a result line is the whole comment or a lane list:
 *
 *   `// Uint32Array [ 0, 10, 30 ]`     a comment naming the whole array
 *   `` | `[0, 10, 30]` | ``            a backticked lane list in prose
 *   `console.log(dist[1], dist[2]); // 10 30`   lanes stated by index
 *
 * The third shape is why the function returns `indices` as well as `values`:
 * `dist[1], dist[2]` claims two specific lanes, not a two-lane array, and
 * conflating the two is the mistake this file exists over.
 */
function documentedResults(text, file) {
  const found = [];
  text.split("\n").forEach((line, i) => {
    const trimmed = line.trim();
    const at = i + 1;
    const wholeArray = /^\/\/\s*Uint32Array \[/.test(trimmed);
    const prose = /`\[[^\]]*\]`/.test(line) && /ssspCsr|typed wrapper/.test(line);
    if (wholeArray || prose) {
      const lanes = line.match(/\[([^\]]*)\]/)[1].split(",")
        .map((s) => s.trim()).filter((s) => s.length);
      if (lanes.length === 0) return;
      found.push({ line: at, text: trimmed, shape: "array",
        values: lanes.map((t) => laneValue(t, file, at)) });
      return;
    }
    // `console.log(TOTAL_EXPORTS, d[1], d[2]);   // 86 10 30` -- arguments are
    // paired with the documented values POSITIONALLY, because the example
    // prints a constant next to two lanes and matching them by eye is how a
    // mismatch hides. A bare name is a non-lane claim (TOTAL_EXPORTS); an
    // indexed one is a lane claim, at that index.
    const call = trimmed.match(/^console\.log\(([^)]*)\);/);
    if (!call || !/\/\/\s*[\d\s]+$/.test(trimmed)) return;
    const args = call[1].split(",").map((s) => s.trim()).filter((s) => s.length);
    const want = trimmed.slice(trimmed.lastIndexOf("//") + 2).trim().split(/\s+/);
    assert.equal(want.length, args.length,
      `${file}:${at}: console.log prints ${args.length} value(s) and the ` +
      `comment documents ${want.length}`);
    found.push({ line: at, text: trimmed, shape: "indexed",
      args: args.map((a, j) => {
        const m = a.match(/^(?:dist|d)\[(\d+)\]$/);
        return { arg: a, index: m ? Number(m[1]) : null,
          value: laneValue(want[j], file, at) };
      }) });
  });
  return found;
}

let bridge;
let K;

before(async () => {
  bridge = await bridgeFor("readme");
  K = await import("../dist/kernels.js");
});

test("the documented graph has one lane per vertex and no unreachable one", () => {
  // The structural fact the wrong comment missed. Stated on its own so a
  // failure here reads as "the prose describes a different graph", not as a
  // number mismatch three lines away.
  const text = readFileSync(DOC_SITES[0], "utf8");
  const g = documentedGraph(text);
  const v = g.indptr.length - 1;
  assert.equal(v, 3, `indptr.length - 1 = ${v}; the README's quick-start is a ` +
    `3-vertex graph`);
  // Every vertex is reachable from the source on this graph, so no lane is
  // Q_INF and none may be documented as one.
  const reach = dijkstraRef(g);
  assert.ok(reach.every((d) => d !== 0xffffffff),
    `the quick-start graph has an unreachable vertex (${JSON.stringify(reach)}); ` +
    `if that is meant, the README's edge list has to change with this test`);
});

test("every documented ssspCsr result is what the kernel returns", () => {
  for (const file of DOC_SITES) {
    const text = readFileSync(file, "utf8");
    const g = documentedGraph(text);
    const got = [...K.ssspCsr(bridge, new Uint32Array(g.indptr),
      new Uint32Array(g.indices), new Uint32Array(g.weights), g.source)];
    const want = dijkstraRef(g);
    assert.deepEqual(got, want,
      `${file}: the documented graph and the kernel disagree about the answer`);
    // The lane count is the claim that was wrong; check it explicitly so the
    // failure says so.
    assert.equal(got.length, g.indptr.length - 1,
      `${file}: ssspCsr returned ${got.length} lanes for ` +
      `${g.indptr.length - 1} vertices`);

    const documented = documentedResults(text, file);
    assert.ok(documented.length > 0,
      `${file}: no documented ssspCsr result was found. This test exists ` +
      `because one was wrong and nothing noticed; if the shape of the ` +
      `documentation changed, teach this parser the new shape rather than ` +
      `deleting the check`);
    for (const d of documented) {
      const fail =
        `${file}:${d.line} documents ` +
        `${d.shape === "array" ? `the array ${JSON.stringify(d.values)}` : ""}` +
        `${d.shape === "indexed" ? d.args.map((a) => `${a.arg} = ${a.value}`).join(", ") : ""}` +
        `, and the kernel returns ${JSON.stringify(got)} for the graph that ` +
        `file describes.\n  ${d.text}`;
      if (d.shape === "indexed") {
        // Each argument checked against what it names: a lane at its index, a
        // bare name against the exported constant. The wrong one is named.
        for (const a of d.args) {
          if (a.index === null) {
            assert.equal(K[a.arg], a.value,
              `${file}:${d.line}: ${a.arg} is documented as ${a.value}, the ` +
              `package exports ${K[a.arg]}\n  ${d.text}`);
            continue;
          }
          assert.equal(got[a.index], a.value, fail);
        }
        continue;
      }
      assert.deepEqual(d.values, got, fail);
    }
  }
});

test("the README no longer shows an unreachable lane in the quick-start", () => {
  // Named separately because this exact sentence is the defect, and a future
  // edit that reintroduces the four-lane claim should fail with a message that
  // says which sentence it was rather than "deepEqual failed".
  const text = readFileSync(DOC_SITES[0], "utf8");
  const quick = text.slice(0, text.indexOf("## What this is"));
  assert.ok(!/Infinity-as-0xffffffff/.test(quick),
    "the quick-start example again claims an Infinity lane for a graph in " +
    "which all 3 vertices are reachable");
  assert.ok(/Uint32Array \[ 0, 10, 30 \]/.test(quick),
    "the quick-start example no longer shows the three lanes the kernel " +
    "returns");
});