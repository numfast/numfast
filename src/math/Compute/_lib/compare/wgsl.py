"""Compare WGSL — GPU: out[i] = select(0u, 1u, a[i] op b[i]).

op задаётся числовым кодом в uniform "op":
  0=gt, 1=ge, 2=lt, 3=le, 4=eq, 5=ne (f32, как пакует pack_uniforms).

Динамические биндинги по scalar-режиму (use_scalar_a/b) — как в
map_binary: входные массивы объявляются только если реально связаны.
WGSL использует select(false_case, true_case, condition).
"""


def _make_wgsl(num_inputs: int) -> str:
    """Generate WGSL with correct number of input bindings (a, b)."""
    bindings = []
    body = ""
    n = "arrayLength(&dst)"

    if num_inputs == 0:
        # Degenerate: оба операнда — скаляры (никогда не используется,
        # builder требует минимум один вход; код для полноты).
        bindings.append("    @group(0) @binding(0) var<storage, read_write> dst: array<u32>;")
        bindings.append("    @group(0) @binding(1) var<uniform> params: Params;")
        body = f"""    let i = id.x;
    let n = {n};
    if (i >= n) {{ return; }}
    let a_val = params.scalar_a;
    let b_val = params.scalar_b;
"""
    elif num_inputs == 1:
        bindings.append("    @group(0) @binding(0) var<storage, read> a_arr: array<f32>;")
        bindings.append("    @group(0) @binding(1) var<storage, read_write> dst: array<u32>;")
        bindings.append("    @group(0) @binding(2) var<uniform> params: Params;")
        body = f"""    let i = id.x;
    let n = {n};
    if (i >= n) {{ return; }}
    let use_sa = params.use_scalar_a != 0.0;
    let use_sb = params.use_scalar_b != 0.0;
    let a_val = select(a_arr[i], params.scalar_a, use_sa);
    let b_val = select(a_arr[i], params.scalar_b, use_sb);
"""
    else:
        bindings.append("    @group(0) @binding(0) var<storage, read> a_arr: array<f32>;")
        bindings.append("    @group(0) @binding(1) var<storage, read> b_arr: array<f32>;")
        bindings.append("    @group(0) @binding(2) var<storage, read_write> dst: array<u32>;")
        bindings.append("    @group(0) @binding(3) var<uniform> params: Params;")
        body = f"""    let i = id.x;
    let n = {n};
    if (i >= n) {{ return; }}
    let use_sa = params.use_scalar_a != 0.0;
    let use_sb = params.use_scalar_b != 0.0;
    let a_val = select(a_arr[i], params.scalar_a, use_sa);
    let b_val = select(b_arr[i], params.scalar_b, use_sb);
"""

    src = f"""
struct Params {{
    op: f32,
    scalar_a: f32,
    scalar_b: f32,
    use_scalar_a: f32,
    use_scalar_b: f32,
}};

{chr(10).join(bindings)}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
{body}
    var r: u32;
    if (params.op == 0.0) {{
        r = select(0u, 1u, a_val > b_val);
    }} else if (params.op == 1.0) {{
        r = select(0u, 1u, a_val >= b_val);
    }} else if (params.op == 2.0) {{
        r = select(0u, 1u, a_val < b_val);
    }} else if (params.op == 3.0) {{
        r = select(0u, 1u, a_val <= b_val);
    }} else if (params.op == 4.0) {{
        r = select(0u, 1u, a_val == b_val);
    }} else {{
        r = select(0u, 1u, a_val != b_val);
    }}
    dst[i] = r;
}}
"""
    return src


def wgsl_generator(uniforms: dict) -> str:
    """Callable WGSL generator for Runtime.

    Receives packet.uniforms, returns full WGSL source.
    Числовой op-код: 0=gt, 1=ge, 2=lt, 3=le, 4=eq, 5=ne.
    """
    use_sa = float(uniforms.get("use_scalar_a", 0))
    use_sb = float(uniforms.get("use_scalar_b", 0))
    num_inputs = (0 if use_sa else 1) + (0 if use_sb else 1)
    return _make_wgsl(num_inputs)


# Статичный 2-входовый вариант (backward compat) + callable-алиас
# (паттерн map_binary/mask: __init__.py реэкспортит callable как wgsl).
WGSL = _make_wgsl(2)
wgsl = wgsl_generator


__all__ = ["WGSL", "wgsl_generator", "wgsl"]
