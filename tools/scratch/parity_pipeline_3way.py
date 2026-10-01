# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""3-way pipeline parity: generate -> encode -> filter -> sort -> gather ->
groupby -> reduce on Python-CPU vs native-CPU vs WASM. Seed 42, N=200K.

Backends:
  py     = NUMFAST_NATIVE_DISABLE=1 (numpy reference / Python CPU)
  native = Rust DLL via ctypes/Series path (native CPU)
  wasm   = node drivers over the fresh wasm32 build (same Rust core)

Graph (int32-only, bit-exact parity, no float tolerance needed):
  S0 generate: rng_fill_i32(N, seed, stream=0, lo=0, hi=1000) -> raw
  S1 encode:   map((raw*7+13) mod 1024) -> codes [nf_map_scalar_i32 x3]
  S2 filter:   mask=codes<768; scatter codes/raw     [mask: host; scatter: kernel]
  S3 sort:     stable perm of filtered codes         [nf_sort_perm_i32]
  S4 gather:   codes/vals by perm                    [nf_join_gather_i32 or CPU gap]
  S5 groupby:  sum+count of vals by codes            [nf_group_sum_count, dense g=768]
  S6 reduce:   total=sum(sums), count=sum(counts)    [host sum; no reduce export]

Explicit gaps (recorded, never silent):
  G1 compare (mask build): no WASM export -> host segment in wasm exec.
  G2 gather: nf_join_gather_i32 traps on wasm32 (unreachable, scoped
     threads) -> explicit CPU segment in wasm exec.
  G3 reduce: no WASM reduce export -> host sum in wasm exec.

Usage (Git Bash, sequential, timeout 550):
  /c/App/numfast/.venv/Scripts/python tools/scratch/parity_pipeline_3way.py --n 200000
  /c/App/numfast/.venv/Scripts/python tools/scratch/parity_pipeline_3way.py --worker py --outdir <d>
Stdout only: stage x backend table + parity + gaps. No files left behind
(stage files live in a temp dir, removed at the end).
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SEED, LO, HI, MOD, KEEP, GROUPS = 42, 0, 1000, 1024, 768, 768
TOOLS = ROOT / "numfast-native" / "tools"
WASM_CANDS = [ROOT / "numfast-native" / "target" / "wasm32-unknown-unknown"
              / "release" / "numfast_native.wasm",
              TOOLS / "numfast_native.wasm"]
NODE = (r"C:\Program Files\nodejs\node.exe"
        if Path(r"C:\Program Files\nodejs\node.exe").exists() else "node")
PY = r"C:\App\numfast\.venv\Scripts\python.exe"


def sha(a):
    import numpy as np
    return hashlib.sha256(
        np.ascontiguousarray(a).tobytes()).hexdigest()


def find_wasm():
    for c in WASM_CANDS:
        if c.exists():
            return c.as_posix()
    raise FileNotFoundError("no numfast_native.wasm; build wasm target first")


# ---------------------------------------------------------------- worker
def run_graph(a, jobs, n, backend):
    import numpy as np  # noqa: F401 (kept local: worker-only import)
    t = {}
    s = time.perf_counter()
    g = a["compile"](jobs)
    t["compile"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    g = a["optimize"](g)
    t["optimize"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    res = a["evaluate"](g, backend, int(n))
    t["execute"] = (time.perf_counter() - s) * 1000
    return res, t


def as_i32(buf):
    import numpy as np
    return np.ascontiguousarray(np.asarray(buf, dtype=np.int32))


def worker(mode, outdir):
    import numpy as np
    import numfast as nf
    t_all = time.perf_counter()
    k = nf.get_kernel(fresh=True)
    a = k.alias
    N = WORKER_N
    stages, files = {}, {}

    def keep(name, arr, extra=None):
        p = os.path.join(outdir, name + ".i32")
        np.ascontiguousarray(arr, dtype=np.int32).tofile(p)
        files[name] = p
        return p

    def rec(name, res, t, extra=None):
        ei = res["execution_info"]
        d = {"actual": ei.get("actual"), "requested": ei.get("requested"),
             "reason": ei.get("reason"), "ms": {k_: round(v, 3)
                                                for k_, v in t.items()},
             "total_ms": round(sum(t.values()), 3)}
        if extra:
            d.update(extra)
        stages[name] = d
        return res["result"]

    # S0 generate
    r, t = run_graph(a, [a["ir_rng_fill_i32"]("r", N, SEED, 0, 0, LO, HI)],
                     N, "cpu")
    raw = as_i32(rec("generate", r, t, {"sha256": sha(as_i32(r["result"]))}))
    keep("raw", raw)
    # S1 encode (map: (raw*7+13) mod 1024 -> dense codes in [0,1024))
    r, t = run_graph(a, [a["ir_series"]("s", raw, "int32"),
                         a["ir_map"]("m1", "s", "mul", 7),
                         a["ir_map"]("m2", "m1", "add", 13),
                         a["ir_map"]("m", "m2", "mod", MOD)], N, "cpu")
    codes = as_i32(rec("encode", r, t, {"sha256": sha(as_i32(r["result"]))}))
    assert (codes == (raw.astype(np.int64) * 7 + 13) % MOD).all()
    keep("codes", codes)
    # S2 filter (compare + scatter, one graph per lane)
    def fjobs(vtag, vcol, ftag):
        return [a["ir_series"]("s", codes, "int32"),
                a["ir_series"]("v", vcol, "int32"),
                a["ir_compare"]("c", "s", KEEP, "<"),
                a["ir_filter"](ftag, vtag, "c")]
    r, t = run_graph(a, fjobs("s", codes, "f"), N, "cpu")
    fcodes = as_i32(rec("filter/codes", r, t))
    r, t = run_graph(a, fjobs("v", raw, "f"), N, "cpu")
    fraw = as_i32(rec("filter/values", r, t))
    m = codes < KEEP
    assert len(fcodes) == int(m.sum()) and (fcodes == codes[m]).all()
    assert (fraw == raw[m]).all()
    nf_ = len(fcodes)
    stages["filter/codes"]["kept"] = f"{nf_}/{N}"
    stages["filter/values"]["kept"] = f"{nf_}/{N}"
    stages["filter/codes"]["sha256"] = sha(fcodes)
    stages["filter/values"]["sha256"] = sha(fraw)
    keep("fcodes", fcodes)
    keep("fraw", fraw)
    # S3 sort perm
    r, t = run_graph(a, [a["ir_series"]("s", fcodes, "int32"),
                         a["ir_sort"]("p", "s")], nf_, "cpu")
    perm = as_i32(rec("sort/perm", r, t, {"sha256": sha(as_i32(r["result"]))}))
    assert sorted(perm.tolist()) == list(range(nf_))
    assert (np.sort(fcodes, kind="stable") == fcodes[perm]).all()
    keep("perm", perm)
    # S4 gather both lanes
    r, t = run_graph(a, [a["ir_series"]("s", fcodes, "int32"),
                         a["ir_series"]("p", perm, "int32"),
                         a["ir_gather"]("g", "s", "p")], nf_, "cpu")
    scodes = as_i32(rec("gather/codes", r, t,
                        {"sha256": sha(as_i32(r["result"]))}))
    r, t = run_graph(a, [a["ir_series"]("s", fraw, "int32"),
                         a["ir_series"]("p", perm, "int32"),
                         a["ir_gather"]("g", "s", "p")], nf_, "cpu")
    sraw = as_i32(rec("gather/values", r, t,
                      {"sha256": sha(as_i32(r["result"]))}))
    assert (scodes == fcodes[perm]).all() and (sraw == fraw[perm]).all()
    keep("scodes", scodes)
    keep("sraw", sraw)
    # S5 groupby sum+count (engine dicts -> dense [0, GROUPS))
    r, t = run_graph(a, [a["ir_series"]("v", sraw, "int32"),
                         a["ir_series"]("k", scodes, "int32"),
                         a["ir_groupby"]("g", "v", "k", "sum")], nf_, "cpu")
    dsum = rec("groupby/sum", r, t)
    r, t = run_graph(a, [a["ir_series"]("v", sraw, "int32"),
                         a["ir_series"]("k", scodes, "int32"),
                         a["ir_groupby"]("g", "v", "k", "count")], nf_, "cpu")
    dcnt = rec("groupby/count", r, t)
    dense_s = np.zeros(GROUPS, dtype=np.int64)
    dense_c = np.zeros(GROUPS, dtype=np.int64)
    for kk_, vv in dict(dsum).items():
        dense_s[int(kk_)] = int(vv)
    for kk_, vv in dict(dcnt).items():
        dense_c[int(kk_)] = int(vv)
    eu = np.unique(scodes)
    assert set(int(x) for x in dict(dsum)) == set(int(x) for x in eu)
    for kk_ in dict(dsum):
        sel = sraw[scodes == int(kk_)]
        assert dense_s[int(kk_)] == int(sel.sum())
        assert dense_c[int(kk_)] == int(sel.size)
    stages["groupby/sum"]["sha256"] = sha(dense_s)
    stages["groupby/count"]["sha256"] = sha(dense_c)
    stages["groupby/sum"]["ngroups"] = len(dsum)
    dense_s.astype(np.float64).tofile(os.path.join(outdir, "gsums.f64"))
    dense_c.tofile(os.path.join(outdir, "gcounts.i64"))
    np.ascontiguousarray(sraw, dtype=np.float64).tofile(
        os.path.join(outdir, "sraw.f64"))
    # S6 reduce (global checksums over group results)
    gs = dense_s.astype(np.int32)
    gc = dense_c.astype(np.int32)
    r, t = run_graph(a, [a["ir_series"]("s", gs, "int32"),
                         a["ir_reduce"]("r", "s", "sum")], len(gs), "cpu")
    tot = r["result"]
    tot = tot.item() if isinstance(tot, np.generic) else tot
    rec("reduce/total", r, t, {"value": int(tot)})
    r, t = run_graph(a, [a["ir_series"]("s", gc, "int32"),
                         a["ir_reduce"]("r", "s", "sum")], len(gc), "cpu")
    cnt = r["result"]
    cnt = cnt.item() if isinstance(cnt, np.generic) else cnt
    rec("reduce/count", r, t, {"value": int(cnt)})
    assert int(tot) == int(sraw.sum()) and int(cnt) == nf_

    out = {"mode": mode, "n": N, "seed": SEED, "kept": nf_,
           "ngroups": len(dsum), "total": int(tot), "count": int(cnt),
           "native": nf.native_info(), "stages": stages,
           "wall_ms": round((time.perf_counter() - t_all) * 1000, 1)}
    print(json.dumps(out))
    return out


# ---------------------------------------------------------------- wasm chain
def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=550, **kw)


def wasm_chain(tmp, n):
    import numpy as np
    tmp = Path(tmp).as_posix()  # forward slashes: safe inside node -e strings
    W = find_wasm()
    S, T = {}, {}
    t_all = time.perf_counter()

    def qkeys(path, dtype):
        return np.fromfile(path, dtype=dtype)

    # S0 generate (kernel; inline node, file handoff)
    js = ("const fs=require('node:fs'),crypto=require('node:crypto');"
          f"const b=fs.readFileSync('{W}');"
          "const m=new WebAssembly.Module(b);"
          "const instance=new WebAssembly.Instance(m,{});"
          "const mem=instance.exports.memory,BASE=0x101000;"
          f"const N={n},NEED=BASE+N*4+65536;"
          "if(mem.buffer.byteLength<NEED)"
          "{mem.grow(Math.ceil((NEED-mem.buffer.byteLength)/65536));}"
          "const t0=performance.now();"
          f"const rc=instance.exports.nf_rng_fill_i32(BASE,N,{SEED}n,0n,0n,"
          f"{LO},{HI},0);"
          "const ms=performance.now()-t0;"
          "if(rc!==0){console.log(JSON.stringify({rc}));process.exit(0);}"
          "const v=new Int32Array(mem.buffer,BASE,N);"
          f"fs.writeFileSync('{tmp}/raw.i32',"
          "Buffer.from(v.buffer,v.byteOffset,v.byteLength));"
          "console.log(JSON.stringify({rc,ms,sha:crypto.createHash('sha256')"
          ".update(Buffer.from(v.buffer,v.byteOffset,v.byteLength))"
          ".digest('hex')}));")
    p = sh([NODE, "-e", js], cwd=str(ROOT))
    assert p.returncode == 0, p.stderr[:300]
    d = json.loads(p.stdout.strip().splitlines()[-1])
    assert d.get("rc") == 0, d
    S["generate"] = {"backend": "wasm-kernel:nf_rng_fill_i32",
                     "ms": round(d["ms"], 3), "sha256": d["sha"]}
    raw = qkeys(f"{tmp}/raw.i32", np.int32)
    # S1 encode (kernel chain via wasm_map.mjs: *7 -> +13 -> mod 1024)
    ems, hms = [], 0.0
    prev, suf = f"{tmp}/raw.i32", "i32"
    for op, sval, out in ((2, 7, "enc1"), (0, 13, "enc2"), (6, MOD, "codes")):
        t0 = time.perf_counter()
        p = sh([NODE, str(TOOLS / "wasm_map.mjs"), W, prev, "-", "i32",
                str(op), str(n), "i32", str(sval), "parity", tmp],
               cwd=str(ROOT))
        hms += (time.perf_counter() - t0) * 1000
        assert p.returncode == 0, p.stderr[:300]
        d = json.loads(p.stdout.strip().splitlines()[-1])
        assert d.get("rc") == 0, d
        ems.append(d["ms"])
        shutil.copy(f"{tmp}/map_out.i32", f"{tmp}/{out}.i32")
        prev = f"{tmp}/{out}.i32"
    codes = qkeys(f"{tmp}/codes.i32", np.int32)
    assert (codes == (raw.astype(np.int64) * 7 + 13) % MOD).all()
    S["encode"] = {"backend": "wasm-kernel:nf_map_scalar_i32(*7,+13,mod)",
                   "ms": round(sum(ems), 3), "harness_ms": round(hms, 1),
                   "sha256": sha(codes)}
    # S2 filter (GAP G1: mask built on host; scatter on kernel)
    t0 = time.perf_counter()
    mask = (codes < KEEP).astype(np.uint8)
    mask_ht = (time.perf_counter() - t0) * 1000
    mask.tofile(f"{tmp}/mask.u8")
    codes.tofile(f"{tmp}/codes.i32")
    shk = {}
    for lane, src in (("codes", f"{tmp}/codes.i32"),
                      ("values", f"{tmp}/raw.i32")):
        p = sh([NODE, str(TOOLS / "wasm_select.mjs"), W, "scatter", "i32",
                src, f"{tmp}/mask.u8", str(n), "parity", tmp], cwd=str(ROOT))
        assert p.returncode == 0, p.stderr[:300]
        d = json.loads(p.stdout.strip().splitlines()[-1])
        nf_ = int(d["m"])
        out = qkeys(f"{tmp}/sel_out.i32", np.int32)
        assert len(out) == nf_
        shk[lane] = (d["ms"], out)
        shutil.copy(f"{tmp}/sel_out.i32", f"{tmp}/f_{lane}.i32")
    fcodes, fraw = shk["codes"][1], shk["values"][1]
    assert (fcodes == codes[mask.astype(bool)]).all()
    assert (fraw == raw[mask.astype(bool)]).all()
    S["filter/mask"] = {"backend": "CPU-segment(G1):host compare, no export",
                        "ms": round(mask_ht, 3)}
    S["filter/codes"] = {"backend": "wasm-kernel:nf_select_scatter_i32",
                         "ms": round(shk["codes"][0], 3),
                         "sha256": sha(fcodes), "kept": f"{nf_}/{n}"}
    S["filter/values"] = {"backend": "wasm-kernel:nf_select_scatter_i32",
                          "ms": round(shk["values"][0], 3),
                          "sha256": sha(fraw), "kept": f"{nf_}/{n}"}
    # S3 sort perm (kernel; inline node — no checked-in driver exists)
    fcodes.tofile(f"{tmp}/fcodes.i32")
    js = ("const fs=require('node:fs');"
          f"const b=fs.readFileSync('{W}');"
          "const m=new WebAssembly.Module(b);"
          "const instance=new WebAssembly.Instance(m,{});"
          "const mem=instance.exports.memory,BASE=0x101000;"
          f"const N={nf_};"
          "const need=BASE+N*4*5+65536;"
          "if(mem.buffer.byteLength<need)"
          "{mem.grow(Math.ceil((need-mem.buffer.byteLength)/65536));}"
          "const K=BASE,P=K+N*4,T0=P+N*4,T1=T0+N*4,TP=T1+N*4;"
          f"const fb=fs.readFileSync('{tmp}/fcodes.i32');"
          "new Int32Array(mem.buffer,K,N).set("
          "new Int32Array(fb.buffer,fb.byteOffset,N));"
          "const t0=performance.now();"
          "const rc=instance.exports.nf_sort_perm_i32(K,N,0,P,T0,T1,TP);"
          "const ms=performance.now()-t0;"
          "if(rc!==0){console.log(JSON.stringify({rc}));process.exit(0);}"
          "const pv=new Int32Array(mem.buffer,P,N);"
          f"fs.writeFileSync('{tmp}/perm.i32',"
          "Buffer.from(pv.buffer,pv.byteOffset,pv.byteLength));"
          "console.log(JSON.stringify({rc,ms}));")
    p = sh([NODE, "-e", js], cwd=str(ROOT))
    assert p.returncode == 0, p.stderr[:300]
    d = json.loads(p.stdout.strip().splitlines()[-1])
    assert d.get("rc") == 0, d
    perm = qkeys(f"{tmp}/perm.i32", np.int32)
    assert sorted(perm.tolist()) == list(range(nf_))
    assert (np.sort(fcodes, kind="stable") == fcodes[perm]).all()
    S["sort/perm"] = {"backend": "wasm-kernel:nf_sort_perm_i32(inline driver)",
                      "ms": round(d["ms"], 3), "sha256": sha(perm)}
    # S4 gather (GAP G2: join_gather traps on wasm32 -> CPU segment)
    t0 = time.perf_counter()
    scodes = np.ascontiguousarray(fcodes[perm])
    sraw = np.ascontiguousarray(fraw[perm])
    S["gather/codes"] = {"backend": "CPU-segment(G2):numpy take "
                                    "(nf_join_gather_i32 traps: unreachable)",
                         "ms": round((time.perf_counter() - t0) * 1000, 3),
                         "sha256": sha(scodes)}
    t0 = time.perf_counter()
    sraw = np.ascontiguousarray(fraw[perm])
    S["gather/values"] = {"backend": "CPU-segment(G2):numpy take",
                          "ms": round((time.perf_counter() - t0) * 1000, 3),
                          "sha256": sha(sraw)}
    # S5 groupby (kernel via wasm_run.mjs; vals bridged i32->f64, exact)
    scodes.tofile(f"{tmp}/scodes.i32")
    sraw.astype(np.float64).tofile(f"{tmp}/sraw.f64")
    p = sh([NODE, str(TOOLS / "wasm_run.mjs"), W, f"{tmp}/scodes.i32",
            f"{tmp}/sraw.f64", str(nf_), str(GROUPS), "parity", tmp],
           cwd=str(ROOT))
    assert p.returncode == 0, p.stderr[:300]
    d = json.loads(p.stdout.strip().splitlines()[-1])
    assert d.get("rc") == 0, d
    wsums = qkeys(f"{tmp}/sums.f64", np.float64)
    wcnts = qkeys(f"{tmp}/counts.i64", np.int64)
    # sums are exact integers in f64 -> int64 view for byte-comparable sha
    S["groupby/sum"] = {"backend": "wasm-kernel:nf_group_sum_count(f64 bridge)",
                        "ms": round(d["ms"], 3),
                        "sha256": sha(wsums.astype(np.int64))}
    S["groupby/count"] = {"backend": "wasm-kernel:nf_group_sum_count",
                          "ms": round(d["ms"], 3), "sha256": sha(wcnts)}
    # S6 reduce (GAP G3: no reduce export -> host sum)
    t0 = time.perf_counter()
    tot, cnt = float(wsums.sum()), int(wcnts.sum())
    S["reduce/total"] = {"backend": "CPU-segment(G3):host sum, no export",
                         "ms": round((time.perf_counter() - t0) * 1000, 3),
                         "value": int(tot)}
    S["reduce/count"] = {"backend": "CPU-segment(G3):host sum",
                         "ms": 0.0, "value": cnt}
    T["wall_ms"] = round((time.perf_counter() - t_all) * 1000, 1)
    return {"stages": S, "timings": T, "kept": nf_, "total": int(tot),
            "count": cnt, "wsums": wsums, "wcnts": wcnts, "scodes": scodes,
            "sraw": sraw}


# ---------------------------------------------------------------- main
def main():
    global WORKER_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--worker", choices=["py", "native"], default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-wasm", action="store_true")
    a_ = ap.parse_args()
    WORKER_N = int(a_.n)
    N = int(a_.n)

    if a_.worker:
        os.makedirs(a_.outdir, exist_ok=True)
        worker(a_.worker, a_.outdir)
        return

    import numpy as np
    t_all = time.perf_counter()
    tmp = tempfile.mkdtemp(prefix="nf3way_")
    try:
        # Exec A/B: fresh subprocess each (DLL load is per-process).
        runs = {}
        for mode in ("py", "native"):
            d = os.path.join(tmp, mode)
            os.makedirs(d, exist_ok=True)
            env = dict(os.environ)
            if mode == "py":
                env["NUMFAST_NATIVE_DISABLE"] = "1"
            else:
                env.pop("NUMFAST_NATIVE_DISABLE", None)
            t0 = time.perf_counter()
            p = sh([PY, __file__, "--worker", mode, "--outdir", d,
                    "--n", str(N)], cwd=str(ROOT), env=env)
            wall = (time.perf_counter() - t0) * 1000
            assert p.returncode == 0, f"worker {mode} failed: {p.stderr[-2000:]}"
            r = json.loads(p.stdout.strip().splitlines()[-1])
            r["proc_wall_ms"] = round(wall, 1)
            runs[mode] = (r, d)
        # Exec C: wasm chain.
        wdir = os.path.join(tmp, "wasm")
        os.makedirs(wdir, exist_ok=True)
        wr = None if a_.no_wasm else wasm_chain(wdir, N)

        # ---- parity: py vs native (all stages bit-exact) ----
        py, native = runs["py"][0], runs["native"][0]
        order = ["generate", "encode", "filter/codes", "filter/values",
                 "sort/perm", "gather/codes", "gather/values",
                 "groupby/sum", "groupby/count", "reduce/total",
                 "reduce/count"]
        parity = {}
        for st in order:
            ps, ns = py["stages"][st], native["stages"][st]
            if st.startswith("reduce/"):
                ok = ps.get("value") == ns.get("value")
            else:
                ok = ps.get("sha256") == ns.get("sha256")
            parity[st] = ("EXACT" if ok else
                          f"MISMATCH py={ps.get('sha256', ps.get('value'))} "
                          f"native={ns.get('sha256', ns.get('value'))}")
            assert ok, f"{st}: {parity[st]}"
        assert py["total"] == native["total"] and py["count"] == native["count"]
        # oracle: totals match filtered lane sums (int64 exact)
        fraw = np.fromfile(os.path.join(runs["py"][1], "fraw.i32"),
                           dtype=np.int32)
        assert py["total"] == int(fraw.sum()) and py["count"] == len(fraw)

        # ---- parity: engine vs wasm ----
        wpar = {}
        if wr is not None:
            w = wr["stages"]
            pairs = [("generate", py["stages"]["generate"]["sha256"],
                      w["generate"]["sha256"]),
                     ("encode", py["stages"]["encode"]["sha256"],
                      w["encode"]["sha256"]),
                     ("filter/codes", py["stages"]["filter/codes"]["sha256"],
                      w["filter/codes"]["sha256"]),
                     ("filter/values",
                      py["stages"]["filter/values"]["sha256"],
                      w["filter/values"]["sha256"]),
                     ("sort/perm", py["stages"]["sort/perm"]["sha256"],
                      w["sort/perm"]["sha256"]),
                     ("gather/codes", py["stages"]["gather/codes"]["sha256"],
                      w["gather/codes"]["sha256"]),
                     ("gather/values",
                      py["stages"]["gather/values"]["sha256"],
                      w["gather/values"]["sha256"])]
            for st, e, g in pairs:
                wpar[st] = ("EXACT" if e == g else f"MISMATCH eng={e} wasm={g}")
                assert e == g, f"{st}: {wpar[st]}"
            # groupby: counts bit-exact; sums exact via f64 bridge
            ec = np.fromfile(os.path.join(runs["py"][1], "gcounts.i64"),
                             dtype=np.int64)
            es = np.fromfile(os.path.join(runs["py"][1], "gsums.f64"),
                             dtype=np.float64)
            wpar["groupby/count"] = ("EXACT" if (ec == wr["wcnts"]).all()
                                     else "MISMATCH")
            wpar["groupby/sum"] = ("EXACT" if (es == wr["wsums"]).all()
                                   else f"maxdiff="
                                   f"{np.max(np.abs(es - wr['wsums']))}")
            assert (ec == wr["wcnts"]).all()
            assert (es == wr["wsums"]).all()
            wpar["reduce/total"] = ("EXACT" if wr["total"] == py["total"]
                                    else "MISMATCH")
            wpar["reduce/count"] = ("EXACT" if wr["count"] == py["count"]
                                    else "MISMATCH")
            assert wr["total"] == py["total"] and wr["count"] == py["count"]

        total_ms = (time.perf_counter() - t_all) * 1000
        print(f"== 3-way pipeline N={N} seed={SEED} lo/hi=[{LO},{HI}) "
              f"mod={MOD} keep<{KEEP} ==")
        print(f"FILTER kept {py['kept']}/{N}  GROUPS ng={py['ngroups']}  "
              f"TOTAL={py['total']} COUNT={py['count']}")
        print("--- stage x backend (route | ms | parity) ---")
        for st in order:
            ps, ns = py["stages"][st], native["stages"][st]
            tag = ps.get("sha256", ps.get("value"))
            if isinstance(tag, str):
                tag = "sha=" + tag[:12] + ".."
            else:
                tag = f"val={tag}"
            line = (f"{st:14s} py:{ps['actual']}({ps['total_ms']:.1f}ms) "
                    f"native:{ns['actual']}({ns['total_ms']:.1f}ms) "
                    f"| {parity[st]} {tag}")
            if wr is not None and st in wr["stages"]:
                ws = wr["stages"][st]
                wtag = ws.get("sha256", ws.get("value"))
                wtag = ("sha=" + wtag[:12] + "..") if isinstance(wtag, str) \
                    else f"val={wtag}"
                line += (f" | wasm:{ws['backend'].split(':')[0]}"
                         f"({ws['ms']:.1f}ms) {wpar.get(st, '')} {wtag}")
            print(line)
        if wr is not None:
            for st in ("filter/mask",):
                ws = wr["stages"][st]
                print(f"{st:14s} wasm:{ws['backend']}({ws['ms']:.1f}ms)")
        print(f"py native_info: {py['native']}")
        print(f"native native_info: {native['native']}")
        print("GAPS (explicit, by design): "
              "G1 compare/mask-build: no WASM export (host segment); "
              "G2 gather: nf_join_gather_i32 traps on wasm32 (unreachable, "
              "scoped threads) -> CPU segment; "
              "G3 reduce: no WASM reduce export (host sum).")
        print(f"TOTAL wall={total_ms:.0f}ms "
              f"(py={py['proc_wall_ms']:.0f} native={native['proc_wall_ms']:.0f}"
              + (f" wasm={wr['timings']['wall_ms']:.0f}" if wr else "") + ")")
        print("--- json ---")
        print(json.dumps({"n": N, "seed": SEED, "kept": py["kept"],
                          "ngroups": py["ngroups"], "total": py["total"],
                          "count": py["count"],
                          "parity_py_native": parity,
                          "parity_engine_wasm": wpar,
                          "py": py["stages"], "native": native["stages"],
                          "wasm": wr["stages"] if wr else None,
                          "total_ms": total_ms}, default=str))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
