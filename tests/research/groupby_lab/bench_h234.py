# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""bench_h234: HIERARCHICAL + BLOCK STATS + PARTIAL COALESCE (groups 2-4).

One sequential job. All golden-verified. Usage:
  python -m tests.research.groupby_lab.bench_h234 OUT.json
Covers:
 H2  L0->L1->L2 hierarchy for all 15 algos (per-level meta + state funnel)
 H3  fanin sweep 2/4/8 for D-linear vs E-balanced (total tree work)
 H7  per-block adaptive (meta dispatch local + E-tree merge) vs best fixed
 B3  block_meta build cost/bytes/load + planner-per-block speedup (O vs A)
 B4  partial coalesce B seg=2/4/8 (reduction vs next-level cost)
 T5  reduction tree: total rows 1M, blocks 32K..512K, merges A/D/E
"""

import json
import sys
import time

import numpy as np

from .blockgen import gen_block
from .candidates import ALGOS, get
from .metadata import block_meta
from .runners import hierarchical, merge_all
from .verifier import verify


def warmup(mod):
    rng = np.random.default_rng(0)
    kd = rng.integers(0, 50, size=2048).astype(np.int32)
    vd = rng.integers(-9, 9, size=2048).astype(np.int32)
    mod.local(kd, vd)
    ks = np.sort(rng.integers(0, 1000000, size=8192).astype(np.int32))
    vs = rng.integers(-9, 9, size=8192).astype(np.int32)
    a, _ = mod.local(ks, vs)
    kh = rng.integers(0, 10000000, size=8192).astype(np.int32)
    b, _ = mod.local(kh, vs)
    mod.merge([a, b])


def _mix8():
    specs = [(65536, 1000000, "uniform"), (65536, 100000, "skewed"),
             (65536, 1000000, "clustered"), (65536, 100000, "sorted"),
             (65536, 1000000, "random"), (65536, 100, "uniform"),
             (65536, 70000, "unique95"), (65536, 1000000, "uniform")]
    blocks = []
    for i, (n, M, pat) in enumerate(specs):
        rng = np.random.default_rng(7000 + i)
        k, v = gen_block(n, M, pat, rng)
        blocks.append({"keys": k, "vals": v, "M": M, "pattern": pat})
    return blocks


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    res = {"hier": [], "fanin": [], "adaptive": [], "blockstats": [],
           "partial": [], "tree": []}
    for n in list("ABCDEFGHIJKLMNO"):
        warmup(ALGOS[n])

    blocks = _mix8()
    big_k = np.concatenate([b["keys"] for b in blocks])
    big_v = np.concatenate([b["vals"] for b in blocks])

    # ---- H2: hierarchy all 15 ----
    for name in list("ABCDEFGHIJKLMNO"):
        mod = get(name)
        h = hierarchical(mod, blocks, fanin=4)
        chk = verify(h["final"], big_k, big_v, label=f"{name}/hier")
        res["hier"].append({
            "algo": name, "l0_meta_ms": h["l0_meta_ms"], "l1_ms": h["l1_ms"],
            "l2_ms": h["l2_ms"], "final_ms": h["final_ms"],
            "total_ms": h["l0_meta_ms"] + h["l1_ms"] + h["l2_ms"] + h["final_ms"],
            "raw_bytes": h["raw_bytes"], "l1_bytes": h["l1_bytes"],
            "l2_bytes": h["l2_bytes"], "final_bytes": h["final_bytes"],
            "l1_reduction": h["l1_reduction"], "l2_reduction": h["l2_reduction"],
            "total_reduction": h["total_reduction"],
            "l1_meta": h["l1_meta"], "ngroups": chk["ngroups"],
            "merge_rounds": h["finfo"].get("rounds"),
            "comparisons": h["finfo"].get("comparisons"),
        })
        print(f"ok hier {name} total={res['hier'][-1]['total_ms']:.1f}ms "
              f"l1red={h['l1_reduction']:.2f} groups={chk['ngroups']}", flush=True)

    # ---- H3: fanin sweep D vs E ----
    for name in ["D", "E", "A"]:
        mod = get(name)
        for fanin in [2, 4, 8]:
            h = hierarchical(mod, blocks, fanin=fanin)
            verify(h["final"], big_k, big_v, label=f"{name}/fanin{fanin}")
            res["fanin"].append({
                "algo": name, "fanin": fanin,
                "total_ms": h["l1_ms"] + h["l2_ms"] + h["final_ms"],
                "l2_ms": h["l2_ms"], "final_ms": h["final_ms"],
                "comparisons": h["finfo"].get("comparisons"),
                "rounds": h["finfo"].get("rounds"),
                "l2_bytes": h["l2_bytes"], "final_bytes": h["final_bytes"],
            })
            print(f"ok fanin {name}/{fanin} total={res['fanin'][-1]['total_ms']:.1f}ms "
                  f"comps={h['finfo'].get('comparisons')}", flush=True)

    # ---- H7/B3: per-block adaptive vs fixed-A (planner-per-block question) --
    t0 = time.perf_counter()
    fixed = [get("A").local(b["keys"], b["vals"])[0] for b in blocks]
    t_fixed = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    disp = []
    for b in blocks:
        m = block_meta(b["keys"])
        span = m["range"]
        n = m["rows"]
        if span <= min(4 * n, 1_000_000):
            st, _ = get("O").local(b["keys"], b["vals"])
            how = "dense"
        elif m["sorted"]:
            st, _ = get("D").local(b["keys"], b["vals"])
            how = "sorted"
        elif m["uniq_ratio"] > 0.9:
            st, _ = get("A").local(b["keys"], b["vals"])
            how = "hash"
        else:
            st, _ = get("C").local(b["keys"], b["vals"])
            how = "part"
        disp.append((st, how))
    t_disp_local = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    fin_ad, info_ad = get("E").merge([s for s, _ in disp])
    t_ad_merge = (time.perf_counter() - t0) * 1000.0
    fin_fx, info_fx = merge_all(get("A"), fixed)
    verify(fin_ad, big_k, big_v, label="H7/adaptive")
    verify(fin_fx, big_k, big_v, label="H7/fixed")
    res["adaptive"].append({
        "fixed_A_ms": t_fixed, "dispatch_local_ms": t_disp_local,
        "adaptive_merge_ms": info_ad.get("merge_ms", 0.0),
        "fixed_merge_ms": info_fx.get("merge_ms", 0.0),
        "speedup": t_fixed / max(t_disp_local, 1e-9),
        "dispatch": [h for _, h in disp],
    })
    print(f"ok adaptive fixed={t_fixed:.1f}ms dispatch={t_disp_local:.1f}ms "
          f"speedup={t_fixed / max(t_disp_local, 1e-9):.2f}x "
          f"plan={[h for _, h in disp]}", flush=True)

    # ---- B3: block_meta cost/bytes/load ----
    for n, M, pat in [(65536, 1000000, "uniform"), (1048576, 10000000, "uniform"),
                      (65536, 100, "uniform"), (65536, 100000, "sorted")]:
        rng = np.random.default_rng(31337 + n)
        k, _ = gen_block(n, M, pat, rng)
        t0 = time.perf_counter()
        m = block_meta(k)
        el = (time.perf_counter() - t0) * 1000.0
        res["blockstats"].append({
            "n": n, "M": M, "pattern": pat, "meta_ms": el, "meta": m,
            "meta_overhead": el / max(n / 1e6, 1e-9) * 1e3 / 1e3,
            "bytes_ratio": 200 / max(m["raw_bytes"], 1),
        })
        print(f"ok blockmeta n={n} {pat} {el:.2f}ms meta={m}", flush=True)

    # ---- B4: partial coalesce seg sweep ----
    from .candidates.b_partial import describe as bdesc
    rng = np.random.default_rng(777)
    k100, v100 = gen_block(100000, 1000000, "uniform", rng)
    B = get("B")
    for seg in [2, 4, 8]:
        t0 = time.perf_counter()
        st, s = B.local(k100, v100, seg=seg)
        el = (time.perf_counter() - t0) * 1000.0
        d = bdesc(k100, s["partial_rows"])
        fin, info = merge_all(B, [st])
        verify(fin, k100, v100, label=f"B/seg{seg}")
        res["partial"].append({
            "seg": seg, "local_ms": el, "merge_ms": info.get("merge_ms", 0.0),
            "total_ms": el + info.get("merge_ms", 0.0), **d,
        })
        print(f"ok partial seg={seg} partial_rows={s['partial_rows']} "
              f"total={el + info.get('merge_ms', 0.0):.2f}ms {d}", flush=True)
    # direct baseline
    t0 = time.perf_counter()
    st, s = get("A").local(k100, v100)
    fin, info = merge_all(get("A"), [st])
    el = (time.perf_counter() - t0) * 1000.0
    verify(fin, k100, v100, label="A/direct100K")
    res["partial"].append({"seg": "direct-A", "total_ms": el,
                           "local_ms": s.get("local_ms", 0.0),
                           "merge_ms": info.get("merge_ms", 0.0)})
    print(f"ok partial direct-A total={el:.2f}ms", flush=True)

    # ---- T5: reduction tree, total 1M rows ----
    for bsize in [32768, 65536, 131072, 262144, 524288]:
        nblk = 1048576 // bsize
        blks = []
        for i in range(nblk):
            rng = np.random.default_rng(5000 + i)
            k, v = gen_block(bsize, 10000000, "uniform", rng)
            blks.append({"keys": k, "vals": v})
        bk = np.concatenate([b["keys"] for b in blks])
        bv = np.concatenate([b["vals"] for b in blks])
        for name in ["A", "D", "E"]:
            mod = get(name)
            t0 = time.perf_counter()
            states = [mod.local(b["keys"], b["vals"])[0] for b in blks]
            t_loc = (time.perf_counter() - t0) * 1000.0
            fin, info = merge_all(mod, states)
            t_mg = info.get("merge_ms", 0.0)
            verify(fin, bk, bv, label=f"{name}/tree{bsize}")
            res["tree"].append({
                "algo": name, "block": bsize, "nblocks": nblk,
                "local_ms": t_loc, "merge_ms": t_mg,
                "total_ms": t_loc + t_mg,
                "comparisons": info.get("comparisons"),
                "rounds": info.get("rounds"),
                "final_bytes": int(fin.state_bytes()),
            })
            print(f"ok tree {name} blk={bsize} nblk={nblk} "
                  f"total={t_loc + t_mg:.1f}ms rounds={info.get('rounds')}", flush=True)

    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
