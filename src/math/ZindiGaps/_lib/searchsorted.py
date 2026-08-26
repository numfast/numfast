"""SearchSorted — GPU primitive: binary search in a sorted (cell, month) index.

For each query row (cell_id, month) finds the history entry:
  mode=0 (exact): value at EXACT month (found=1 if present, else 0)
  mode=1 (le):    value of the RIGHTMOST month <= query month (found=1 if any)

History layout: months[] and vals[] sorted by (cell, month) ascending;
cell_start[c] / cell_len[c] give the contiguous range of cell c.

DAG from existing primitives is strictly worse:
  - le-mode could be a segmented scan-carry, but the history has 1 row/month
    (train) AND 2 rows/month (test, two TWS bands), so position arithmetic
    cannot locate month t-gap; exact-mode requires a per-cell search anyway.
  - Gather needs precomputed indices; computing them IS this search.
"""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan

WGSL = """
struct Params {
    mode: f32,
};

@group(0) @binding(0) var<storage, read> q_cell: array<f32>;
@group(0) @binding(1) var<storage, read> q_month: array<f32>;
@group(0) @binding(2) var<storage, read> months: array<f32>;
@group(0) @binding(3) var<storage, read> vals: array<f32>;
@group(0) @binding(4) var<storage, read> cell_start: array<f32>;
@group(0) @binding(5) var<storage, read> cell_len: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_val: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_found: array<f32>;
@group(0) @binding(8) var<storage, read_write> out_month: array<f32>;
@group(0) @binding(9) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&out_val);
    if (i >= n) { return; }

    out_val[i] = 0.0;
    out_found[i] = 0.0;
    out_month[i] = -1.0;

    let c = i32(q_cell[i]);
    let s = i32(cell_start[c]);
    let e = s + i32(cell_len[c]);
    if (e <= s) { return; }

    let m = i32(q_month[i]);

    // rightmost months[idx] <= m
    var lo = s;
    var hi = e - 1;
    while (lo < hi) {
        let mid = (lo + hi + 1) / 2;
        if (i32(months[mid]) <= m) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    let idx = lo;
    if (i32(months[idx]) <= m) {
        let mode = i32(params.mode);
        if (mode == 1 || i32(months[idx]) == m) {
            out_val[i] = vals[idx];
            out_month[i] = months[idx];
            out_found[i] = 1.0;
        }
    }
}
"""


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[
            InputSlot(name="q_cell", dtype="float"),
            InputSlot(name="q_month", dtype="float"),
            InputSlot(name="months", dtype="float"),
            InputSlot(name="vals", dtype="float"),
            InputSlot(name="cell_start", dtype="float"),
            InputSlot(name="cell_len", dtype="float"),
        ],
        outputs=[
            OutputSlot(dtype="float", template="val"),
            OutputSlot(dtype="float", template="found"),
            OutputSlot(dtype="float", template="month_out"),
        ],
        workspace=[],
        uniforms={"mode": float(params.get("mode", 0))},
    )


def cpu(ctx):
    src = [ctx.inputs[i].view for i in range(6)]
    dst = [ctx.outputs[i].view for i in range(3)]
    mode = int(ctx.uniforms.get("mode", 0))
    n = dst[0].length()
    for i in range(n):
        dst[0].write(i, 0.0)
        dst[1].write(i, 0.0)
        dst[2].write(i, -1.0)
        c = int(src[0].read(i))
        s = int(src[4].read(c))
        e = s + int(src[5].read(c))
        if e <= s:
            continue
        m = int(src[1].read(i))
        lo, hi = s, e - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if int(src[2].read(mid)) <= m:
                lo = mid
            else:
                hi = mid - 1
        idx = lo
        if int(src[2].read(idx)) <= m:
            if mode == 1 or int(src[2].read(idx)) == m:
                dst[0].write(i, src[3].read(idx))
                dst[1].write(i, 1.0)
                dst[2].write(i, src[2].read(idx))


__all__ = ["WGSL", "describe", "cpu"]