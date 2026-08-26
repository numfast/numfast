"""MapBinary WGSL — GPU: element-wise binary ops with scalar support.

Dynamic binding layout based on actual input count.
op: 0=add, 1=sub, 2=mul, 3=div, 4=max, 5=min, 6=mod

dtype-семантика (решение владельца, int32 domain):
uniform-флаг int_mode=1 -> все буферы array<i32>, скаляры i32 (u32-биты в
uniform), операции add/sub/mul в i32-арифметике WRAP (two's complement),
mod = C-trunc % с guard b==0 -> 0. В int32-режиме допустимы ops {0,1,2,6}
(валидация в descriptor). Выбор варианта — wgsl(uniforms); без uniforms
(compiler._extract_wgsl) возвращается f32-вариант.
"""


def _make_wgsl(num_inputs: int, int_mode: bool = False) -> str:
    """Generate WGSL with correct number of input bindings and dtype variant."""
    v = "i32" if int_mode else "f32"
    bindings = []
    body = ""
    n = "arrayLength(&dst)"

    bindings.append(f"    @group(0) @binding(0) var<storage, read> a_arr: array<{v}>;")
    a_val_expr = "a_arr[i]"
    param = "params.scalar_a"
    if num_inputs == 0:
        bindings.append(f"    @group(0) @binding(0) var<storage, read_write> dst: array<{v}>;")
        bindings.append("    @group(0) @binding(1) var<uniform> params: Params;")
        a_val_expr = "params.scalar_a"
        b_val_expr = "params.scalar_b"
        body = f"""    let i = id.x;
    let n = {n};
    if (i >= n) {{ return; }}
    let a_val = {a_val_expr};
    let b_val = {b_val_expr};
"""
    elif num_inputs == 1:
        bindings.append(f"    @group(0) @binding(1) var<storage, read_write> dst: array<{v}>;")
        bindings.append("    @group(0) @binding(2) var<uniform> params: Params;")
        body = f"""    let i = id.x;
    let n = {n};
    if (i >= n) {{ return; }}
    let use_sa = params.use_scalar_a != {'0u' if int_mode else '0.0'};
    let use_sb = params.use_scalar_b != {'0u' if int_mode else '0.0'};
    let a_val = select({a_val_expr}, params.scalar_a, use_sa);
    let b_val = select({a_val_expr}, params.scalar_b, use_sb);
    """
    else:
        bindings.append(f"    @group(0) @binding(1) var<storage, read> b_arr: array<{v}>;")
        bindings.append(f"    @group(0) @binding(2) var<storage, read_write> dst: array<{v}>;")
        bindings.append("    @group(0) @binding(3) var<uniform> params: Params;")
        body = f"""    let i = id.x;
    let n = {n};
    if (i >= n) {{ return; }}
    let use_sa = params.use_scalar_a != {'0u' if int_mode else '0.0'};
    let use_sb = params.use_scalar_b != {'0u' if int_mode else '0.0'};
    let a_val = select({a_val_expr}, params.scalar_a, use_sa);
    let b_n = arrayLength(&b_arr);
    let b_i = select(i, 0u, b_n <= 1u);
    let b_val = select(b_arr[b_i], params.scalar_b, use_sb);
"""

    zero_t = "0u" if int_mode else "0.0"
    if int_mode:
        # i32 domain: add/sub/mul WRAP; mod = C-trunc %, guard b==0 -> 0.
        # op validated in descriptor.describe() (single point, both backends).
        op_body = """    var result: i32 = 0;
    if (params.op == 0u) {
        result = a_val + b_val;   // i32 WRAP
    } else if (params.op == 1u) {
        result = a_val - b_val;   // i32 WRAP
    } else if (params.op == 2u) {
        result = a_val * b_val;   // i32 WRAP
    } else if (params.op == 6u) {
        // C-trunc modulo; guard mirrors div: b == 0 -> exact 0.
        if (b_val != 0i) { result = a_val % b_val; } else { result = 0i; }
    }
"""
        struct = """struct Params {
    op: u32,
    scalar_a: i32,
    scalar_b: i32,
    use_scalar_a: u32,
    use_scalar_b: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
"""
    else:
        op_body = """    // S49: NaN trap (defense-in-depth). The op code is validated in
    // descriptor.describe() (single point, both backends, before dispatch),
    // so the else->add default is unreachable. If a garbage op somehow
    // reaches the shader, the result is NaN, NOT add. The trap must be a
    // runtime expression (a_val * 0.0 / 0.0): a const-expression NaN literal
    // is rejected by WGSL validation ("Float literal is NaN").
    var result: f32 = a_val * 0.0 / 0.0;
    if (params.op == 0.0) {
        result = a_val + b_val;
    } else if (params.op == 1.0) {
        result = a_val - b_val;
    } else if (params.op == 2.0) {
        result = a_val * b_val;
    } else if (params.op == 3.0) {
        if (b_val != 0.0) { result = a_val / b_val; } else { result = 0.0; }
    } else if (params.op == 4.0) {
        result = max(a_val, b_val);
    } else if (params.op == 5.0) {
        result = min(a_val, b_val);
    } else if (params.op == 6.0) {
        // C-style fmod: WGSL % = a - b * trunc(a / b).
        // Guard mirrors op=3 (div, S50): b == 0 -> exact 0.0, sign +
        // (WGSL % by zero would be NaN).
        if (b_val != 0.0) { result = a_val % b_val; } else { result = 0.0; }
    }
"""
        struct = """struct Params {
    op: f32,
    scalar_a: f32,
    scalar_b: f32,
    use_scalar_a: f32,
    use_scalar_b: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};
"""

    src = f"""
{struct}

{chr(10).join(bindings)}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
{body}
{op_body}
    dst[i] = result;
}}
"""
    return src


def wgsl(uniforms: dict) -> str:
    int_mode = bool(int(uniforms.get("int_mode", 0)))
    use_sa = float(uniforms.get("use_scalar_a", 0))
    use_sb = float(uniforms.get("use_scalar_b", 0))
    num_inputs = (0 if use_sa else 1) + (0 if use_sb else 1)
    return _make_wgsl(num_inputs, int_mode)


__all__ = ["wgsl"]
