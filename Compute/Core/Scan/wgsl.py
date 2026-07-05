"""Scan WGSL — three chained kernels for multi-workgroup prefix sum.

ScanLocal:  local Hillis-Steele scan per block of 64
ScanTotals: sequential scan of block totals (single thread)
ScanFinal:  add block offset to each element
"""

WGSL_LOCAL = """
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<storage, read_write> block_sums : array<f32>;

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
        acc = acc + input[block_start + i];
        output[block_start + i] = acc;
    }

    block_sums[block_id] = acc;
}
"""

WGSL_TOTALS = """
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (gid.x > 0u) { return; }
    let n = arrayLength(&input);
    if (n == 0u) { return; }
    var acc = input[0];
    output[0] = acc;
    for (var i = 1u; i < n; i = i + 1u) {
        acc = acc + input[i];
        output[i] = acc;
    }
}
"""

WGSL_FINAL = """
@group(0) @binding(0) var<storage, read> data : array<f32>;
@group(0) @binding(1) var<storage, read> local : array<f32>;
@group(0) @binding(2) var<storage, read> block_prefix : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    let n = arrayLength(&data);
    let BLOCK = 64u;

    if (idx >= n) { return; }

    let block_id = idx / BLOCK;
    var offset = 0.0;
    if (block_id > 0u) {
        offset = block_prefix[block_id - 1u];
    }

    output[idx] = local[idx] + offset;
}
"""

wgsl_local = WGSL_LOCAL
wgsl_totals = WGSL_TOTALS
wgsl_final = WGSL_FINAL
