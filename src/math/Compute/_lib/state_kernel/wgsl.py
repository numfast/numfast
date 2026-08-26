"""StateKernel WGSL — GPU implementations for all modes.

Binding order: inputs -> workspace -> outputs -> uniform.

WGSL_SINGLE / WGSL_DUAL: serial (workgroup_size(1)) — full-array loop,
the recurrence out[i] = a*in[i] + b*out[i-1] is sequential by nature.

WGSL_SCAN_*: parallel 3-pass scan for the SAME linear recurrence
(SINGLE mode 0 / DUAL mode 1), distributed by GLOBAL invocation id:
each thread processes exactly one chunk of `chunk` elements
([tid*chunk, tid*chunk+chunk)), so the array is covered exactly once.
Chunk = 256 by default; builder dispatch = ceil(n/64) >= ceil(n/chunk),
excess threads return early (guard `tid >= nblocks`).

Passes:
  Local:   local run with ZERO state per chunk + last[tid] (last local value)
  Totals:  tid==0 only: sp[t] = last[t-1] + b^chunk * sp[t-1], sp[0] = 0
  Final:   out[i] = tmp[i] + b^(j+1) * sp[tid]  (exact carry transfer)

SUPERTREND: serial, workgroup_size(1) — state machine is not a linear
recurrence.
"""

# ---------------------------------------------------------------------------
# Serial reference implementations (recurrence is sequential)
# ---------------------------------------------------------------------------

WGSL_SINGLE = """
struct Params {
    mode: f32,
    a: f32,
    b: f32,
    _pad: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> state: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = i32(arrayLength(&src));
    if (n == 0) { return; }
    let a = params.a;
    let b = params.b;
    // Seed: out[0] = src[0] (serial semantics, сид = src[0])
    var prev: f32 = src[0];
    dst[0] = prev;
    state[0] = prev;
    for (var i: i32 = 1; i < n; i = i + 1) {
        let val = a * src[i] + b * prev;
        dst[i] = val;
        state[i] = val;
        prev = val;
    }
}
"""

WGSL_DUAL = """
struct Params {
    mode: f32,
    a: f32,
    b: f32,
    _pad: f32,
};

@group(0) @binding(0) var<storage, read> in0: array<f32>;
@group(0) @binding(1) var<storage, read> in1: array<f32>;
@group(0) @binding(2) var<storage, read_write> st0: array<f32>;
@group(0) @binding(3) var<storage, read_write> st1: array<f32>;
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = i32(arrayLength(&in0));
    let a = params.a;
    let b = params.b;
    // DUAL: NO seed, p0 = p1 = 0 -> out0[0] = a * in0[0]
    var p0: f32 = 0.0;
    var p1: f32 = 0.0;
    for (var i: i32 = 0; i < n; i = i + 1) {
        let v0 = a * in0[i] + b * p0;
        let v1 = a * in1[i] + b * p1;
        out0[i] = v0;
        out1[i] = v1;
        st0[i] = v0;
        st1[i] = v1;
        p0 = v0;
        p1 = v1;
    }
}
"""

WGSL_SUPERTREND = """
struct Params {
    mode: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> close: array<f32>;
@group(0) @binding(1) var<storage, read> upper_band: array<f32>;
@group(0) @binding(2) var<storage, read> lower_band: array<f32>;
@group(0) @binding(3) var<storage, read_write> st_upper: array<f32>;
@group(0) @binding(4) var<storage, read_write> st_lower: array<f32>;
@group(0) @binding(5) var<storage, read_write> st_dir: array<f32>;
@group(0) @binding(6) var<storage, read_write> dst: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = i32(arrayLength(&close));

    var pu: f32 = 0.0;  // prev_upper
    var pl: f32 = 0.0;  // prev_lower
    var pd: f32 = 1.0;  // prev_dir

    for (var i: i32 = 0; i < n; i = i + 1) {
        var direction: f32;
        var final_upper: f32;
        var final_lower: f32;

        if (i == 0) {
            direction = 1.0;
            final_upper = upper_band[i];
            final_lower = lower_band[i];
        } else {
            if (pd == 1.0) {
                if (close[i] < pl) {
                    direction = -1.0;
                } else {
                    direction = 1.0;
                }
            } else {
                if (close[i] > pu) {
                    direction = 1.0;
                } else {
                    direction = -1.0;
                }
            }

            if (direction == 1.0) {
                final_upper = max(upper_band[i], pu);
                final_lower = lower_band[i];
            } else {
                final_upper = upper_band[i];
                final_lower = min(lower_band[i], pl);
            }
        }

        dst[i] = direction;
        st_upper[i] = final_upper;
        st_lower[i] = final_lower;
        st_dir[i] = direction;

        pu = final_upper;
        pl = final_lower;
        pd = direction;
    }
}
"""

# ---------------------------------------------------------------------------
# Parallel 3-pass scan (SINGLE mode 0)
# ---------------------------------------------------------------------------

WGSL_SCAN_LOCAL = """
struct Params {
    mode: f32,
    a: f32,
    b: f32,
    chunk: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read_write> last: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let tid = i32(id.x);
    let n = i32(arrayLength(&src));
    let chunk_i = i32(params.chunk);
    let nblocks = (n + chunk_i - 1) / chunk_i;
    if (tid >= nblocks) { return; }
    let start = tid * chunk_i;
    let end = min(start + chunk_i, n);
    var p: f32 = 0.0;
    var i0: i32 = start;
    if (tid == 0) {
        p = src[start];
        dst[start] = p;
        i0 = start + 1;
    }
    for (var i: i32 = i0; i < end; i = i + 1) {
        let val = params.a * src[i] + params.b * p;
        dst[i] = val;
        p = val;
    }
    last[tid] = p;
}
"""

WGSL_SCAN_TOTALS = """
struct Params {
    mode: f32,
    a: f32,
    b: f32,
    chunk: f32,
};

@group(0) @binding(0) var<storage, read> last: array<f32>;
@group(0) @binding(1) var<storage, read_write> sp: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u) { return; }
    let nblocks = i32(arrayLength(&last));
    let chunk_i = i32(params.chunk);
    if (nblocks == 0) { return; }
    var pw: f32 = 1.0;
    for (var k: i32 = 0; k < chunk_i; k = k + 1) {
        pw = pw * params.b;
    }
    sp[0] = 0.0;
    for (var t: i32 = 1; t < nblocks; t = t + 1) {
        sp[t] = last[t - 1] + pw * sp[t - 1];
    }
}
"""

WGSL_SCAN_FINAL = """
struct Params {
    mode: f32,
    a: f32,
    b: f32,
    chunk: f32,
};

@group(0) @binding(0) var<storage, read> tmp: array<f32>;
@group(0) @binding(1) var<storage, read> sp: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let tid = i32(id.x);
    let n = i32(arrayLength(&tmp));
    let chunk_i = i32(params.chunk);
    let nblocks = (n + chunk_i - 1) / chunk_i;
    if (tid >= nblocks) { return; }
    let start = tid * chunk_i;
    let end = min(start + chunk_i, n);
    var c: f32 = 0.0;
    if (tid != 0) { c = sp[tid]; }
    for (var i: i32 = start; i < end; i = i + 1) {
        c = c * params.b;
        out[i] = tmp[i] + c;
    }
}
"""

# ---------------------------------------------------------------------------
# Parallel 3-pass scan (DUAL mode 1)
# ---------------------------------------------------------------------------

WGSL_SCAN_LOCAL_DUAL = """
struct Params {
    mode: f32,
    a: f32,
    b: f32,
    chunk: f32,
};

@group(0) @binding(0) var<storage, read> in0: array<f32>;
@group(0) @binding(1) var<storage, read> in1: array<f32>;
@group(0) @binding(2) var<storage, read_write> out0: array<f32>;
@group(0) @binding(3) var<storage, read_write> out1: array<f32>;
@group(0) @binding(4) var<storage, read_write> last0: array<f32>;
@group(0) @binding(5) var<storage, read_write> last1: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let tid = i32(id.x);
    let n = i32(arrayLength(&in0));
    let chunk_i = i32(params.chunk);
    let nblocks = (n + chunk_i - 1) / chunk_i;
    if (tid >= nblocks) { return; }
    let start = tid * chunk_i;
    let end = min(start + chunk_i, n);
    var p0: f32 = 0.0;
    var p1: f32 = 0.0;
    for (var i: i32 = start; i < end; i = i + 1) {
        let v0 = params.a * in0[i] + params.b * p0;
        let v1 = params.a * in1[i] + params.b * p1;
        out0[i] = v0;
        out1[i] = v1;
        p0 = v0;
        p1 = v1;
    }
    last0[tid] = p0;
    last1[tid] = p1;
}
"""

WGSL_SCAN_TOTALS_DUAL = """
struct Params {
    mode: f32,
    a: f32,
    b: f32,
    chunk: f32,
};

@group(0) @binding(0) var<storage, read> last0: array<f32>;
@group(0) @binding(1) var<storage, read> last1: array<f32>;
@group(0) @binding(2) var<storage, read_write> sp0: array<f32>;
@group(0) @binding(3) var<storage, read_write> sp1: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u) { return; }
    let nblocks = i32(arrayLength(&last0));
    let chunk_i = i32(params.chunk);
    if (nblocks == 0) { return; }
    var pw: f32 = 1.0;
    for (var k: i32 = 0; k < chunk_i; k = k + 1) {
        pw = pw * params.b;
    }
    sp0[0] = 0.0;
    sp1[0] = 0.0;
    for (var t: i32 = 1; t < nblocks; t = t + 1) {
        sp0[t] = last0[t - 1] + pw * sp0[t - 1];
        sp1[t] = last1[t - 1] + pw * sp1[t - 1];
    }
}
"""

WGSL_SCAN_FINAL_DUAL = """
struct Params {
    mode: f32,
    a: f32,
    b: f32,
    chunk: f32,
};

@group(0) @binding(0) var<storage, read> tmp0: array<f32>;
@group(0) @binding(1) var<storage, read> tmp1: array<f32>;
@group(0) @binding(2) var<storage, read> sp0: array<f32>;
@group(0) @binding(3) var<storage, read> sp1: array<f32>;
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let tid = i32(id.x);
    let n = i32(arrayLength(&tmp0));
    let chunk_i = i32(params.chunk);
    let nblocks = (n + chunk_i - 1) / chunk_i;
    if (tid >= nblocks) { return; }
    let start = tid * chunk_i;
    let end = min(start + chunk_i, n);
    var c0: f32 = 0.0;
    var c1: f32 = 0.0;
    if (tid != 0) {
        c0 = sp0[tid];
        c1 = sp1[tid];
    }
    for (var i: i32 = start; i < end; i = i + 1) {
        c0 = c0 * params.b;
        c1 = c1 * params.b;
        out0[i] = tmp0[i] + c0;
        out1[i] = tmp1[i] + c1;
    }
}
"""

__all__ = [
    "WGSL_SINGLE", "WGSL_DUAL", "WGSL_SUPERTREND",
    "WGSL_SCAN_LOCAL", "WGSL_SCAN_LOCAL_DUAL",
    "WGSL_SCAN_TOTALS", "WGSL_SCAN_TOTALS_DUAL",
    "WGSL_SCAN_FINAL", "WGSL_SCAN_FINAL_DUAL",
]
