"""Expression WGSL — JIT генерация шейдера из формулы.

wgsl_generator(uniforms: dict) -> str
  Принимает uniforms из ExecutionPacket, возвращает WGSL-строку.
  Кешируется в wgpu_driver по (kernel, hash(wgsl_source)).
"""

import re


def _translate_formula(formula: str) -> str:
    """Translate Python-style formula to WGSL expression.

    - Variables A-Z are valid WGSL identifiers (same spelling)
    - Functions abs/max/min are WGSL built-ins
    - Integer literals get .0 suffix (1 -> 1.0)
    """
    # Integer literals to float: digits not part of identifier, not already decimal
    result = re.sub(r'(?<![.\w])(\d+)(?!\.\d)(?!\w)', r'\1.0', formula)
    return result


def _generate_wgsl(formula: str, num_vars: int) -> str:
    """Generate full WGSL shader code for formula."""
    vars = [chr(65 + i) for i in range(num_vars)]  # A, B, C, ...

    # Input bindings: @group(0) @binding(i) var<storage, read> in_i: array<f32>;
    input_bindings = []
    for i in range(num_vars):
        input_bindings.append(
            f"@group(0) @binding({i}) var<storage, read> in_{i}: array<f32>;"
        )

    # Variable assignments: let A = in_0[i];
    var_assignments = []
    for i, v in enumerate(vars):
        var_assignments.append(f"    let {v} = in_{i}[i];")

    out_binding = num_vars
    uniform_binding = num_vars + 1

    wgsl_formula = _translate_formula(formula)

    return f"""struct Params {{
    _pad: f32,
}};

{chr(10).join(input_bindings)}
@group(0) @binding({out_binding}) var<storage, read_write> dst: array<f32>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
    let n = arrayLength(&dst);
    if (i >= n) {{ return; }}

{chr(10).join(var_assignments)}

    dst[i] = {wgsl_formula};
}}
"""


def wgsl_generator(uniforms: dict) -> str:
    """Callable WGSL generator for Runtime.

    Receives packet.uniforms, returns WGSL source string.
    """
    formula = uniforms.get("formula", "A")
    num_vars = int(uniforms.get("num_vars", 1))
    return _generate_wgsl(formula, num_vars)


__all__ = ["wgsl_generator"]
