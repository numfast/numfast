"""Mask WGSL — GPU: out[i] = select(b, a, cond[i] != 0u).

Динамические биндинги по scalar-режиму (use_scalar_a/b) — как в
map_binary: входные массивы a/b объявляются только если реально
связаны. cond всегда массив (uint32), выход f32.
"""


def wgsl(uniforms: dict) -> str:
    """Callable WGSL generator for Runtime.

    Receives packet.uniforms, returns full WGSL source.
    """
    use_sa = float(uniforms.get("use_scalar_a", 0))
    use_sb = float(uniforms.get("use_scalar_b", 0))
    num_inputs = 1 + (0 if use_sa else 1) + (0 if use_sb else 1)

    bindings = ["    @group(0) @binding(0) var<storage, read> cond: array<u32>;"]
    if not use_sa:
        bindings.append("    @group(0) @binding(1) var<storage, read> a_arr: array<f32>;")
    if not use_sb:
        bindings.append(f"    @group(0) @binding({num_inputs - 1}) var<storage, read> b_arr: array<f32>;")
    bindings.append(f"    @group(0) @binding({num_inputs}) var<storage, read_write> dst: array<f32>;")
    bindings.append(f"    @group(0) @binding({num_inputs + 1}) var<uniform> params: Params;")

    body = f"""    let i = id.x;
    let n = arrayLength(&dst);
    if (i >= n) {{ return; }}
    let use_sa = params.use_scalar_a != 0.0;
    let use_sb = params.use_scalar_b != 0.0;
    let a_val = select({"a_arr[i]" if not use_sa else "0.0"}, params.scalar_a, use_sa);
    let b_val = select({"b_arr[i]" if not use_sb else "0.0"}, params.scalar_b, use_sb);
    dst[i] = select(b_val, a_val, cond[i] != 0u);
"""

    return f"""
struct Params {{
    scalar_a: f32,
    scalar_b: f32,
    use_scalar_a: f32,
    use_scalar_b: f32,
}};

{chr(10).join(bindings)}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
{body}
}}
"""


# Статичный 3-входовый вариант (backward compat: __init__.py реэкспортит WGSL).
WGSL = wgsl({})


__all__ = ["WGSL", "wgsl"]
