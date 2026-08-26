"""WalkForward Full Fused WGSL -- Stage1 v3 MEGA-FUSION + state checkpointing.

One dataset = 2 dispatches (seed -> main), 1 upload, 1 readback:
- SEED_WGSL ("WalkForwardFullFusedSeed"): grid (1, folds). One WG per fold
  walks the fold prefix serially EXACTLY like the old replay ([1, boundary),
  same upd() op sequence from the same row-0 seeded state) and publishes a
  10x f32 state checkpoint {s10,s20,s30,s50,s100,s200,ag14,al14,ag28,al28}
  at every chunk boundary b_j = fstart + j*chunk (slot (fy*dx+j)*10).
  Snapshot == serial state BEFORE evaluating row b_j -- bit-exact.
- WGSL ("WalkForwardFullFused"): grid (ceil(fold_sz/chunk), folds). Each WG
  loads its checkpoint instead of replaying [1, lo), then evaluates its chunk
  exactly as before (SMA/RSI/ret/tuples/atomic epilogue unchanged).
  Replaces the O(fold_sz^2/chunk) redundant replay per workgroup
  (wf_kernel_profile.json: 183ms of 192ms kernel) with one serial pass.

Math mirrors CPU oracle exactly (unchanged):
- sma warmup -> 0.0; sliding windows INCLUDE index 0 (seed close[0])
- rsi guard al>1e-10 -> rs=100 (row0 rsi ~= 99010 -> cond2 always true at row0)
- np.round ties-to-even == WGSL round ties-to-even
- row0: sh[0]==0 -> div=0 -> ret=-1.0 (oracle np.where semantics); mask counts it
- prev==0 rows (j>0): same -> ret=-1.0

COORD NOTE: chunk/fold coordinates come from @builtin(workgroup_id), NOT
global_invocation_id (global id includes the 64 lanes per workgroup; using it
as workgroup index silently disables all lanes except global x==0).
Guards are branchless select() collapses -- valid for any scheduling.

Snapshot loads are always in-bounds (dispatch guarantees fy<folds,
wgid.x<dx and snap_buf holds folds*dx*10 f32); unused/garbage slots are
discarded via select(valid...). All lanes execute identical control flow.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PAIRS = [("sma_10", "sma_20"), ("sma_20", "sma_50"), ("sma_30", "sma_100"),
          ("sma_50", "sma_200"), ("sma_20", "sma_100")]
_SMA_VAR = {"sma_10": "m10", "sma_20": "m20", "sma_30": "m30",
            "sma_50": "m50", "sma_100": "m100", "sma_200": "m200"}
_RSI_VAR = {"rsi_14": "rr14", "rsi_28": "rr28"}
NTUP = 20
NSNAP = 10  # f32 state vars per checkpoint: s10,s20,s30,s50,s100,s200,ag14,al14,ag28,al28
# Seed-pass replication: R duplicate walkers per fold. The serial walk is
# load-LATENCY-bound (~0.5us/row exposed at 1 warp/SM); R replicate warps
# RESIDENT PER SM hide each other's latency (redundant identical work is
# nearly free: same cache lines, ALU idle anyway). Only wgid.x==0 publishes
# snapshots -- identical upd() op sequence => bit-exact serial state.
SEED_REPLICAS = 256

# Exact serial-state update -- MUST stay byte-identical between both kernels
# (bit-exact f32 op sequence => snapshot state == replayed state).
_UPD_FN = """fn upd(i: u32) {{
    // rows i >= 1 only (row 0 is seeded/evaluated in main)
    let ci = close_buf[i];
    s10 = s10 + ci;
    if (i >= 10u) {{ s10 = s10 - close_buf[i - 10u]; }}
    s20 = s20 + ci;
    if (i >= 20u) {{ s20 = s20 - close_buf[i - 20u]; }}
    s30 = s30 + ci;
    if (i >= 30u) {{ s30 = s30 - close_buf[i - 30u]; }}
    s50 = s50 + ci;
    if (i >= 50u) {{ s50 = s50 - close_buf[i - 50u]; }}
    s100 = s100 + ci;
    if (i >= 100u) {{ s100 = s100 - close_buf[i - 100u]; }}
    s200 = s200 + ci;
    if (i >= 200u) {{ s200 = s200 - close_buf[i - 200u]; }}
    if (i != 0u) {{
        let cp = close_buf[i - 1u];
        let d = ci - cp;
        let g = max(d, 0.0);
        let l = max(-d, 0.0);
        if (i == 1u) {{
            ag14 = g; al14 = l; ag28 = g; al28 = l;
        }} else {{
            let a14 = 1.0 / 14.0;
            let b14 = 1.0 - a14;
            ag14 = a14 * g + b14 * ag14;
            al14 = a14 * l + b14 * al14;
            let a28 = 1.0 / 28.0;
            let b28 = 1.0 - a28;
            ag28 = a28 * g + b28 * ag28;
            al28 = a28 * l + b28 * al28;
        }}
    }}
}}"""

_STATE_VARS = """var<private> s10: f32 = 0.0;
var<private> s20: f32 = 0.0;
var<private> s30: f32 = 0.0;
var<private> s50: f32 = 0.0;
var<private> s100: f32 = 0.0;
var<private> s200: f32 = 0.0;
var<private> ag14: f32 = 0.0;
var<private> al14: f32 = 0.0;
var<private> ag28: f32 = 0.0;
var<private> al28: f32 = 0.0;"""

# Snapshot field order -- MUST match _UPD load order in main kernel.
_SNAP_FIELDS = ["s10", "s20", "s30", "s50", "s100", "s200",
                "ag14", "al14", "ag28", "al28"]


def build_tuples():
    """Same generator as benchmark/CPU oracle (parity by construction)."""
    out = []
    for i in range(NTUP):
        a1, b1 = _PAIRS[i % len(_PAIRS)]
        r = "rsi_14" if i % 2 == 0 else "rsi_28"
        th = 50 if i % 3 != 1 else 70
        logic = "and" if i % 2 == 0 else "or"
        t = {"id": f"t{i:02d}",
             "cond": [{"a": a1, "op": "gt", "b": b1},
                      {"a": r, "op": "gt", "b": float(th)}],
             "logic": logic, "metric": "mean_ret"}
        out.append(t)
    return out


def _gen_seed_wgsl() -> str:
    stores = "\n".join(
        f"            snap_buf[base + {k}u] = {v};"
        for k, v in enumerate(_SNAP_FIELDS))
    R = SEED_REPLICAS
    return f"""// WalkForwardFullFusedSeed -- serial state checkpoint pass.
// Grid: ({R}, folds). {R} duplicate walker workgroups per fold walk the fold prefix
// ONCE (rows [1, min(boundary, hi)) per boundary) -- the exact old replay
// upd() sequence from the row-0 seeded state (bit-exact serial state).
// The walk is load-latency-bound; replicate warps resident on the same SM
// hide each other's memory latency (redundant identical work is nearly
// free). Only wgid.x==0 && lane==0 publishes the 10x f32 snapshot at every
// chunk boundary b_j = fstart + j*chunk (state BEFORE evaluating row b_j).

struct Params {{
    n: u32,
    fold_sz: u32,
    folds: u32,
    chunk: u32,
}};

@group(0) @binding(0) var<storage, read> close_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> snap_buf: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// ---- serial state (per invocation) ----
{_STATE_VARS}

{_UPD_FN}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {{
    let n = params.n;
    let fold_sz = params.fold_sz;
    let chunk = params.chunk;
    let fy = wgid.y;
    let dxu = (fold_sz + chunk - 1u) / chunk;
    let fstart = fy * fold_sz;
    let hi = fstart + min(fold_sz, n - fstart);

    if (fstart < n) {{
        // state after global row 0 (oracle windows include index 0)
        s10 = close_buf[0];
        s20 = close_buf[0];
        s30 = close_buf[0];
        s50 = close_buf[0];
        s100 = close_buf[0];
        s200 = close_buf[0];

        var prev: u32 = 1u;      // next row to feed upd()
        var jx: u32 = 0u;
        loop {{
            if (jx >= dxu) {{ break; }}
            // snapshot BEFORE evaluating row b (== state after upd([1,b)))
            let b = min(fstart + jx * chunk, hi);
            var i: u32 = prev;
            loop {{
                if (i >= b) {{ break; }}
                upd(i);
                i = i + 1u;
            }}
            prev = max(prev, b);
            if (wgid.x == 0u && lane == 0u) {{
                let base = (fy * dxu + jx) * {NSNAP}u;
{stores}
            }}
            jx = jx + 1u;
        }}
    }}
}}
"""


def _gen_wgsl() -> str:
    tuples = build_tuples()
    acc_decl, cond_blocks, row0_blocks, epilogue = [], [], [], []
    for t, tp in enumerate(tuples):
        c1, c2 = tp["cond"]
        op = "&&" if tp["logic"] == "and" else "||"
        av, bv = _SMA_VAR[c1["a"]], _SMA_VAR[c1["b"]]
        rv, th = _RSI_VAR[c2["a"]], float(c2["b"])
        v = f"v{t:02d}"
        acc_decl.append(f"    var {v}s: f32 = 0.0;\n    var {v}c: u32 = 0u;")
        # row0: sma warmup 0.0 -> cond1 (0>0) false; rsi(0)=99010 -> cond2 true;
        # and -> false (skip), or -> count + ret=-1.0
        if tp["logic"] == "or":
            row0_blocks.append(
                f"    if (ev0) {{ {v}s = {v}s - 1.0; {v}c = {v}c + 1u; }}  // {tp['id']} row0")
        cond_blocks.append(
            f"        if (({av} > {bv}) {op} ({rv} > {th:.1f})) "
            f"{{ {v}s = {v}s + ret; {v}c = {v}c + 1u; }}  // {tp['id']}: "
            f"{c1['a']}>{c1['b']} {op.strip('&|')} {c2['a']}>{int(th)}")
        epilogue.append(
            f"""    if (lane == {t}u) {{
        let idx{t:02d} = ({t}u * 4u + fy) * 2u;
        atomicAdd(&out_buf[idx{t:02d} + 1u], {v}c);
        var old{t:02d} = atomicLoad(&out_buf[idx{t:02d}]);
        loop {{
            let nxt{t:02d} = bitcast<u32>(bitcast<f32>(old{t:02d}) + {v}s);
            let res{t:02d} = atomicCompareExchangeWeak(&out_buf[idx{t:02d}], old{t:02d}, nxt{t:02d});
            if (res{t:02d}.exchanged) {{ break; }}
            old{t:02d} = res{t:02d}.old_value;
        }}
    }}""")

    body = "\n".join(acc_decl)
    conds = "\n".join(cond_blocks)
    row0 = "\n".join(row0_blocks)
    epi = "\n".join(epilogue)

    snap_loads = "\n".join(
        f"        {v} = select({v}, k{k}, take);"
        for k, v in enumerate(_SNAP_FIELDS))
    snap_lets = "\n".join(
        f"    let k{k} = snap_buf[sidx + {k}u];"
        for k in range(NSNAP))

    return f"""// WalkForwardFullFused -- Stage1 v3 MEGA-FUSION + state checkpointing
// Grid: (ceil(fold_sz/chunk), folds). Each WG: 64 lanes, one chunk, one fold.
// Serial warmup state comes from snap_buf checkpoint written by the seed pass
// (WalkForwardFullFusedSeed) at the WG's chunk boundary -- replaces the old
// O(fold_sz^2/chunk) serial replay [1, lo). State values are bit-exact:
// seed walks the SAME upd() sequence from the SAME row-0 seeded state.
// All lanes execute identical control flow (divergence-free); lane t publishes
// tuple t. Invalid workgroups collapse to empty ranges via select().

struct Params {{
    n: u32,
    fold_sz: u32,
    folds: u32,
    chunk: u32,
}};

@group(0) @binding(0) var<storage, read> close_buf: array<f32>;
@group(0) @binding(1) var<storage, read> snap_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

// ---- serial state (per invocation) ----
{_STATE_VARS}

{_UPD_FN}

fn sma_v(sum_: f32, pf: f32, i: u32, pu: u32) -> f32 {{
    return select(0.0, sum_ / pf, i >= pu - 1u);  // warmup == 0.0 like oracle
}}

fn rsi_out(ag: f32, al: f32) -> f32 {{
    let rs = select(100.0, ag / max(al, 1e-10), al > 1e-10);
    return round((100.0 - 100.0 / (1.0 + rs)) * 1000.0);  // ties-to-even == np.round
}}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {{
    let n = params.n;
    let fold_sz = params.fold_sz;
    let fy = wgid.y;
    let fstart = fy * fold_sz;
    let hi = fstart + min(fold_sz, n - fstart);
    let lo = fstart + wgid.x * params.chunk;
    let valid = (lo < hi) & (lo < n);
    let end = select(0u, min(lo + params.chunk, hi), valid);  // main [max(lo,1), end)
    let ev0 = valid & (lo == 0u);                 // this WG owns row 0

    // seed row 0 into sliding sums (oracle windows include index 0)
    s10 = select(0.0, close_buf[0], valid);
    s20 = select(0.0, close_buf[0], valid);
    s30 = select(0.0, close_buf[0], valid);
    s50 = select(0.0, close_buf[0], valid);
    s100 = select(0.0, close_buf[0], valid);
    s200 = select(0.0, close_buf[0], valid);

    // tuple accumulators
{body}

    // row 0 evaluation (mask counted by oracle; ret = div 0 - 1 = -1.0)
{row0}

    // state checkpointing: load serial state at chunk boundary (seed pass
    // wrote it after walking [1, lo) exactly once per fold) -- no replay.
    let dxu = (fold_sz + params.chunk - 1u) / params.chunk;
    let sidx = (fy * dxu + wgid.x) * {NSNAP}u;
{snap_lets}
    let take = valid & (ev0 == false);
{snap_loads}

    // main chunk pass: rows [max(lo,1), end); row0 already seeded+evaluated
    var i: u32 = select(max(lo, 1u), 1u, ev0);
    loop {{
        if (i >= end) {{ break; }}
        upd(i);
        let m10 = sma_v(s10, 10.0, i, 10u);
        let m20 = sma_v(s20, 20.0, i, 20u);
        let m30 = sma_v(s30, 30.0, i, 30u);
        let m50 = sma_v(s50, 50.0, i, 50u);
        let m100 = sma_v(s100, 100.0, i, 100u);
        let m200 = sma_v(s200, 200.0, i, 200u);
        let rr14 = rsi_out(ag14, al14);
        let rr28 = rsi_out(ag28, al28);
        var ret: f32 = -1.0;
        if (i >= 1u) {{
            let cp = close_buf[i - 1u];
            ret = select(-1.0, (close_buf[i] - cp) / cp, cp != 0.0);
        }}
{conds}
        i = i + 1u;
    }}

    // aggregate to out[160]: lane t publishes tuple t, atomic combine across WGs
{epi}
}}
"""


WGSL = _gen_wgsl()
SEED_WGSL = _gen_seed_wgsl()

__all__ = ["WGSL", "SEED_WGSL", "build_tuples", "NSNAP"]
