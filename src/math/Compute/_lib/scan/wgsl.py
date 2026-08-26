"""Scan WGSL — three chained kernels for multi-workgroup inclusive scan.

ScanLocal:  local scan per block of 64 (inclusive: первый элемент = вход)
ScanTotals: exclusive scan of block totals (sp[0] = neutral)
ScanFinal:  combine block offset with each element

op (uniform, f32): 0 = sum, 1 = mul, 2 = max, 3 = min.
Нейтрали (только Totals; Local и Final начинают с первого элемента):
  sum -> 0.0, mul -> 1.0, max -> -3.4028235e38, min -> 3.4028235e38.
Uniform binding идёт ПОСЛЕ storage-bindings (см. wgpu_driver: uniform_bi =
len(packet.bindings)): Local=3, Totals=2, Final=4.
"""

WGSL_COMBINE = """
fn combine(a: f32, b: f32, code: f32) -> f32 {
    if (code == 0.0) { return a + b; }
    if (code == 1.0) { return a * b; }
    if (code == 2.0) { return max(a, b); }
    return min(a, b);
}
"""

WGSL_NEUTRAL = """
fn neutral(code: f32) -> f32 {
    if (code == 1.0) { return 1.0; }
    if (code == 2.0) { return -3.4028235e38; }
    if (code == 3.0) { return 3.4028235e38; }
    return 0.0;
}
"""

WGSL_LOCAL = """
struct Params {
    op: f32,
};

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<storage, read_write> block_sums : array<f32>;
@group(0) @binding(3) var<uniform> params : Params;
""" + WGSL_COMBINE + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
    let n = arrayLength(&input);
    let BLOCK = 64u;

    if (lid.x != 0u || gid.x >= n) { return; }

    let block_id = gid.x / BLOCK;
    let block_start = block_id * BLOCK;

    var acc = input[block_start];
    output[block_start] = acc;

    let block_end = min(BLOCK, n - block_start);
    for (var i = 1u; i < block_end; i = i + 1u) {
        acc = combine(acc, input[block_start + i], params.op);
        output[block_start + i] = acc;
    }

    block_sums[block_id] = acc;
}
"""

WGSL_TOTALS = """
struct Params {
    op: f32,
};

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> params : Params;
""" + WGSL_COMBINE + WGSL_NEUTRAL + """
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (gid.x > 0u) { return; }
    let n = arrayLength(&input);
    if (n == 0u) { return; }
    var acc = neutral(params.op);
    output[0] = acc;
    for (var i = 1u; i < n; i = i + 1u) {
        acc = combine(acc, input[i - 1u], params.op);
        output[i] = acc;
    }
}
"""

WGSL_FINAL = """
struct Params {
    op: f32,
};

@group(0) @binding(0) var<storage, read> data : array<f32>;
@group(0) @binding(1) var<storage, read> local : array<f32>;
@group(0) @binding(2) var<storage, read> block_prefix : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;
@group(0) @binding(4) var<uniform> params : Params;
""" + WGSL_COMBINE + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    let n = arrayLength(&data);
    let BLOCK = 64u;

    if (idx >= n) { return; }

    let block_id = idx / BLOCK;
    if (block_id > 0u) {
        output[idx] = combine(local[idx], block_prefix[block_id], params.op);
    } else {
        output[idx] = local[idx];
    }
}
"""

wgsl_local = WGSL_LOCAL
wgsl_totals = WGSL_TOTALS
wgsl_final = WGSL_FINAL
