# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Execution-portability pipeline: RNG -> ARITH -> FILTER -> SORT/UNIQUE ->
GROUPBY -> AGGREGATE, seed 42, N=200K..1M.

Runs each stage as its own ExecutionGraph on CPU + GPU-attempt (explicit
fallback recorded, never silent) + WASM RNG check via node (importless
scratch/nf_rng_test.wasm, same Rust core). int32 exact everywhere:
parity is bit-for-bit, RNG pinned by SHA256.

Usage (Git Bash, sequential, timeout 550):
  /c/App/numfast/.venv/Scripts/python tools/scratch/bench_exec_portability_pipeline.py --n 200000
  /c/App/numfast/.venv/Scripts/python tools/scratch/bench_exec_portability_pipeline.py --n 1000000

Stdout only: stage table (backend x stages + parity) + timings + SHAs.
No output files. JOIN branch (RNG->RNG->JOIN->AGGREGATE) out of scope.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def run_graph(a, jobs, n, backend):
    """compile -> optimize -> evaluate; RuntimeError captured, never silent."""
    t = {}
    s = time.perf_counter()
    g = a["compile"](jobs)
    t["compile"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    g = a["optimize"](g)
    t["optimize"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    try:
        res = a["evaluate"](g, backend, int(n))
        err = None
    except Exception as e:  # explicit routing/exec errors only (no silent path)
        res = None
        err = f"{type(e).__name__}: {e}"[:260]
    t["execute"] = (time.perf_counter() - s) * 1000
    return res, err, t


def info_of(res):
    ei = res["execution_info"]
    return {k: ei.get(k) for k in (
        "requested", "actual", "reason", "dispatches", "h2d", "d2h",
        "placement")}


def as_i32(buf):
    return np.ascontiguousarray(np.asarray(buf, dtype=np.int32))


def wasm_rng_sha(n, seed, stream, offset, lo, hi):
    """Drive scratch/nf_rng_test.wasm from node (no files, -e only).

    Returns sha256 hex of the i32 lanes, or (None, reason) when node/wasm
    cannot run. Same Rust core as CPU/GPU -> bit-exact expected.
    """
    node = (r"C:\Program Files\nodejs\node.exe"
            if Path(r"C:\Program Files\nodejs\node.exe").exists() else "node")
    wasm = (ROOT / "scratch" / "nf_rng_test.wasm").as_posix()
    js = (
        "const fs=require('node:fs'),crypto=require('node:crypto');"
        f"const b=fs.readFileSync('{wasm}');"
        "const m=new WebAssembly.Module(b);const inst=new WebAssembly.Instance(m,{});"
        "const mem=inst.exports.memory;"
        f"const N={int(n)},PTR=64,NEED=PTR+N*4+16;"
        "if(mem.buffer.byteLength<NEED){mem.grow(Math.ceil((NEED-mem.buffer.byteLength)/65536));}"
        f"const rc=inst.exports.nf_rng_fill_i32(PTR,N,{int(seed)}n,{int(stream)}n,"
        f"{int(offset)}n,{int(lo)},{int(hi)},0);"
        "if(rc!==0){console.log(JSON.stringify({rc}));process.exit(0);}"
        "const v=new Int32Array(mem.buffer,PTR,N);"
        "console.log(JSON.stringify({rc,sha:crypto.createHash('sha256')"
        ".update(Buffer.from(v.buffer,v.byteOffset,v.byteLength)).digest('hex')}));"
    )
    try:
        p = subprocess.run([node, "-e", js], capture_output=True, text=True,
                           timeout=120, cwd=str(ROOT))
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"node run failed: {e}"
    if p.returncode != 0:
        return None, f"node rc={p.returncode}: {p.stderr.strip()[:160]}"
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None, f"node unparsable: {p.stdout.strip()[:160]}"
    if d.get("rc") != 0:
        return None, f"wasm rc={d.get('rc')}"
    return d.get("sha"), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--no-wasm", action="store_true")
    a_ = ap.parse_args()
    N, SEED, M = a_.n, a_.seed, a_.m

    import numfast as nf
    k = nf.get_kernel(fresh=True)
    a = k.alias
    stages = {}   # stage -> backend -> {info|error, ms, sha/parity}

    def both(name, jobs, n, need_bufs=True):
        """Run stage on CPU + GPU-attempt; return cpu bufs + records."""
        rec = {}
        res_c, err_c, t_c = run_graph(a, jobs, n, "cpu")
        assert res_c is not None, f"{name} cpu failed: {err_c}"
        rec["cpu"] = {"info": info_of(res_c), "ms": t_c,
                      "total_ms": sum(t_c.values())}
        out = {"cpu_bufs": res_c["result"], "rec": rec}
        if not a_.no_gpu:
            res_g, err_g, t_g = run_graph(a, jobs, n, "gpu")
            if res_g is not None:
                rec["gpu"] = {"info": info_of(res_g), "ms": t_g,
                              "total_ms": sum(t_g.values())}
                out["gpu_bufs"] = res_g["result"]
            else:
                rec["gpu"] = {"error": err_g, "ms": t_g,
                              "fallback": "cpu (explicit, no silent route)"}
        stages[name] = rec
        return out

    t_all = time.perf_counter()
    # 0 RNG: two counter-domain streams (values + keys), SHA-pinned.
    r1 = both("RNG/values", [a["ir_rng_fill_i32"]("r", N, SEED, 0, 0, -500, 500)], N)
    r2 = both("RNG/keys", [a["ir_rng_fill_i32"]("r", N, SEED, 1, 0, 0, M)], N)
    V0, K0 = as_i32(r1["cpu_bufs"]), as_i32(r2["cpu_bufs"])
    sha_v, sha_k = sha(V0), sha(K0)
    stages["RNG/values"]["cpu"]["sha256"] = sha_v
    stages["RNG/keys"]["cpu"]["sha256"] = sha_k
    if "gpu_bufs" in r1:
        assert (as_i32(r1["gpu_bufs"]) == V0).all(), "RNG/values CPU!=GPU"
        stages["RNG/values"]["parity_cpu_gpu"] = "EXACT"
    if "gpu_bufs" in r2:
        assert (as_i32(r2["gpu_bufs"]) == K0).all(), "RNG/keys CPU!=GPU"
        stages["RNG/keys"]["parity_cpu_gpu"] = "EXACT"

    # 1 ARITH: v = V0*2+7 (two map nodes, one graph). GPU: explicit error.
    ar = both("ARITH", [a["ir_series"]("s", V0, "int32"),
                        a["ir_map"]("m1", "s", "mul", 2),
                        a["ir_map"]("m", "m1", "add", 7)], N)
    VA = as_i32(ar["cpu_bufs"])
    assert (VA == (V0.astype(np.int64) * 2 + 7).astype(np.int32)).all()
    stages["ARITH"]["cpu"]["sha256"] = sha(VA)
    if "gpu_bufs" in ar:
        assert (as_i32(ar["gpu_bufs"]) == VA).all()
        stages["ARITH"]["parity_cpu_gpu"] = "EXACT"
    else:
        stages["ARITH"]["parity_cpu_gpu"] = "N/A (gpu explicit error)"

    # 2 FILTER: WHERE VA>0 -> keep VA/K0. GPU eligible (compare+filter).
    # Mask must be a compare node in-graph (BoolMask); one graph per output.
    def _fjobs(vtag, vcol, ftag):
        return [a["ir_series"]("s", VA, "int32"),
                a["ir_series"]("kk", K0, "int32"),
                a["ir_compare"]("c", "s", 0, ">"),
                a["ir_filter"](ftag, vtag, "c")]
    f1 = both("FILTER/values", _fjobs("s", VA, "f"), N)
    f2 = both("FILTER/keys", _fjobs("kk", K0, "f"), N)
    VF, KF = as_i32(f1["cpu_bufs"]), as_i32(f2["cpu_bufs"])
    n_f = len(VF)
    assert len(KF) == n_f
    oracle_m = VA > 0
    assert int(oracle_m.sum()) == n_f
    assert (VF == VA[oracle_m]).all() and (KF == K0[oracle_m]).all()
    for st, got, want in (("FILTER/values", f1, VF), ("FILTER/keys", f2, KF)):
        if "gpu_bufs" in got:
            assert (as_i32(got["gpu_bufs"]) == want).all(), f"{st} CPU!=GPU"
            stages[st]["parity_cpu_gpu"] = "EXACT"
        else:
            stages[st]["parity_cpu_gpu"] = "N/A (gpu explicit error)"
    stages["FILTER/values"]["kept"] = f"{n_f}/{N}"
    stages["FILTER/keys"]["kept"] = f"{n_f}/{N}"

    # 3 SORT: perm of KF + gather both. GPU: pow2-only -> explicit error noted.
    s1 = both("SORT/perm", [a["ir_series"]("s", KF, "int32"),
                            a["ir_sort"]("p", "s")], n_f)
    perm = as_i32(s1["cpu_bufs"])
    assert sorted(perm.tolist()) == list(range(n_f)), "perm not a permutation"
    assert (np.sort(KF, kind="stable") == KF[perm]).all(), "perm mis-sorts"
    g1 = both("SORT/keys", [a["ir_series"]("s", KF, "int32"),
                            a["ir_series"]("p", perm, "int32"),
                            a["ir_gather"]("g", "s", "p")], n_f)
    g2 = both("SORT/values", [a["ir_series"]("s", VF, "int32"),
                              a["ir_series"]("p", perm, "int32"),
                              a["ir_gather"]("g", "s", "p")], n_f)
    KS, VS = as_i32(g1["cpu_bufs"]), as_i32(g2["cpu_bufs"])
    assert (KS == KF[perm]).all() and (VS == VF[perm]).all()
    for st, got, want in (("SORT/keys", g1, KS), ("SORT/values", g2, VS)):
        if "gpu_bufs" in got:
            assert (as_i32(got["gpu_bufs"]) == want).all(), f"{st} CPU!=GPU"
            stages[st]["parity_cpu_gpu"] = "EXACT"
        else:
            stages[st]["parity_cpu_gpu"] = "N/A (gpu explicit error)"

    # 4 UNIQUE: sorted-order unique+inverse on KS. GPU: CPU-only op.
    uq = both("UNIQUE", [a["ir_series"]("s", KS, "int32"),
                         a["ir_unique_inverse"]("u", "s")], n_f)
    bufs = a["cpu_execute"](a["optimize"](
        a["compile"]([a["ir_series"]("s", KS, "int32"),
                      a["ir_unique_inverse"]("u", "s")]))["nodes"])
    UQ, INV, NG = (as_i32(bufs["u"]), as_i32(bufs["u#inv"]),
                   int(bufs["u#ng"]))
    eu, ei = np.unique(KS, return_inverse=True)
    assert UQ.tolist() == eu.tolist() and INV.tolist() == ei.tolist()
    assert (UQ[INV] == KS).all()
    stages["UNIQUE"]["cpu"]["ng"] = NG
    stages["UNIQUE"]["parity_vs_numpy"] = "EXACT"
    stages["UNIQUE"]["parity_cpu_gpu"] = "N/A (gpu explicit error)"

    # 5 GROUPBY: sum + count of VS by KS. GPU dense path (keys in [0,M)).
    gb1 = both("GROUPBY/sum", [a["ir_series"]("v", VS, "int32"),
                               a["ir_series"]("k", KS, "int32"),
                               a["ir_groupby"]("g", "v", "k", "sum")], n_f)
    gb2 = both("GROUPBY/count", [a["ir_series"]("v", VS, "int32"),
                                 a["ir_series"]("k", KS, "int32"),
                                 a["ir_groupby"]("g", "v", "k", "count")], n_f)
    d1, d2 = gb1["cpu_bufs"], gb2["cpu_bufs"]
    # oracle: numpy bincount on (KS, VS)
    assert set(d1) == set(np.unique(KS).tolist())
    for kk in d1:
        sel = VS[KS == kk]
        assert d1[kk] == int(sel.sum()) and d2[kk] == int(sel.size), kk
    for st, got in (("GROUPBY/sum", gb1), ("GROUPBY/count", gb2)):
        if "gpu_bufs" in got:
            dg = got["gpu_bufs"]
            dc = got["cpu_bufs"]
            assert list(dg) == list(dc) and all(dg[k] == dc[k] for k in dc)
            stages[st]["parity_cpu_gpu"] = "EXACT"

    # 6 AGGREGATE: global checksums over group results (exact int64).
    gs = np.array([d1[k] for k in sorted(d1)], dtype=np.int32)
    gc = np.array([d2[k] for k in sorted(d2)], dtype=np.int32)
    ag1 = both("AGG/sum", [a["ir_series"]("s", gs, "int32"),
                           a["ir_reduce"]("r", "s", "sum")], len(gs))
    ag2 = both("AGG/count", [a["ir_series"]("s", gc, "int32"),
                             a["ir_reduce"]("r", "s", "sum")], len(gc))
    tot, cnt = ag1["cpu_bufs"], ag2["cpu_bufs"]
    tot = tot.item() if isinstance(tot, np.generic) else tot
    cnt = cnt.item() if isinstance(cnt, np.generic) else cnt
    assert int(tot) == int(VS.sum()) and int(cnt) == n_f, (tot, cnt)
    stages["AGG/sum"]["cpu"]["value"] = int(tot)
    stages["AGG/count"]["cpu"]["value"] = int(cnt)
    stages["AGG/sum"]["oracle"] = f"== sum(VF) {int(VS.sum())} EXACT"
    stages["AGG/count"]["oracle"] = f"== n_f {n_f} EXACT"
    for st, got, want in (("AGG/sum", ag1, int(tot)),
                          ("AGG/count", ag2, int(cnt))):
        if "gpu_bufs" in got:
            gv = got["gpu_bufs"]
            gv = gv.item() if isinstance(gv, np.generic) else gv
            assert int(gv) == want, f"{st} CPU!=GPU"
            stages[st]["parity_cpu_gpu"] = "EXACT"
        elif "gpu" in got["rec"] and "error" in got["rec"]["gpu"]:
            stages[st]["parity_cpu_gpu"] = "N/A (gpu explicit error)"

    # WASM RNG (what it can do): same counter domain, SHA compare.
    wasm = {}
    if not a_.no_wasm:
        for name, stream, lo, hi, ref in (
                ("RNG/values", 0, -500, 500, sha_v),
                ("RNG/keys", 1, 0, M, sha_k)):
            d, reason = wasm_rng_sha(N, SEED, stream, 0, lo, hi)
            wasm[name] = {"sha256": d, "parity_cpu_wasm":
                          ("EXACT" if d == ref else f"MISMATCH vs {ref}"),
                          "note": reason}
    total_ms = (time.perf_counter() - t_all) * 1000

    print(f"== exec-portability pipeline N={N} seed={SEED} M={M} ==")
    print(f"RNG sha values={sha_v}\nRNG sha keys   ={sha_k}")
    print(f"FILTER kept {n_f}/{N}  UNIQUE ng={NG}  "
          f"AGG total={tot} count={cnt}")
    print("--- backend x stages (requested->actual | ms | parity) ---")
    for name, rec in stages.items():
        for be in ("cpu", "gpu"):
            if be not in rec:
                continue
            r = rec[be]
            if "error" in r:
                print(f"{name:14s} {be}: ERROR -> {r['fallback']} | "
                      f"{r['error'][:100]}")
            else:
                i = r["info"]
                par = rec.get("parity_cpu_gpu",
                              stages[name].get("parity_cpu_gpu", "-"))
                extra = ""
                if "sha256" in r:
                    extra += f" sha={r['sha256'][:16]}.."
                if "value" in r:
                    extra += f" val={r['value']}"
                if "ng" in r:
                    extra += f" ng={r['ng']}"
                print(f"{name:14s} {be}: {i['requested']}->{i['actual']} | "
                      f"tot={r['total_ms']:.1f}ms "
                      f"(c={r['ms']['compile']:.1f}/o={r['ms']['optimize']:.1f}/"
                      f"e={r['ms']['execute']:.1f}) | {par}{extra}")
    if wasm:
        for name, w in wasm.items():
            print(f"{name:14s} wasm: sha={w['sha256']} | {w['parity_cpu_wasm']}"
                  + (f" | {w['note']}" if w["note"] else ""))
    print(f"TOTAL wall={total_ms:.0f}ms")
    print("--- json ---")
    print(json.dumps({"n": N, "seed": SEED, "m": M, "sha": {"v": sha_v,
                      "k": sha_k}, "kept": n_f, "ng": NG, "total": int(tot),
                      "count": int(cnt), "stages": stages, "wasm": wasm,
                      "total_ms": total_ms}, default=str))


if __name__ == "__main__":
    main()
