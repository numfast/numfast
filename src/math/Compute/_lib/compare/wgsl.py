"""Compare WGSL — GPU: out[i] = select(0u, 1u, a[i] op b[i]).

Supported ops (set via uniform "op"):
  gt, ge, lt, le, eq, ne

WGSL uses select(false_case, true_case, condition).
Comparison operators return bool, select converts to 0u/1u.
"""

WGSL = """
struct Params {
    _pad: f32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&dst);
    if (i >= n) { return; }
"""

# Op-specific WGSL snippets (appended to WGSL template)
OPS = {
    "gt":  "    dst[i] = select(0u, 1u, a[i] > b[i]);",
    "ge":  "    dst[i] = select(0u, 1u, a[i] >= b[i]);",
    "lt":  "    dst[i] = select(0u, 1u, a[i] < b[i]);",
    "le":  "    dst[i] = select(0u, 1u, a[i] <= b[i]);",
    "eq":  "    dst[i] = select(0u, 1u, a[i] == b[i]);",
    "ne":  "    dst[i] = select(0u, 1u, a[i] != b[i]);",
}

WGSL_SUFFIX = """
}
"""


def wgsl_generator(uniforms: dict) -> str:
    """Callable WGSL generator for Runtime.

    Receives packet.uniforms, returns full WGSL source.
    Supports dynamic op selection via uniform "op".
    """
    op = uniforms.get("op", "gt")
    body = OPS.get(op, OPS["gt"])
    return WGSL + body + WGSL_SUFFIX


__all__ = ["WGSL", "wgsl_generator"]
