"""Combine WGSL — GPU: out = sum(wi * ini) + bias.

Generates WGSL dynamically based on num_inputs (1-4).
Each input gets its own binding; output and uniform follow.

Callable signature: wgsl(uniforms: dict) -> str
  uniforms["num_inputs"] — количество входов (1-4)
"""


def wgsl(uniforms: dict) -> str:
    """Generate Combine WGSL for 1-4 inputs.

    The struct always has w0-w3 (matching descriptor's uniforms),
    so uniform buffer layout is consistent regardless of num_inputs.
    """
    num_inputs_raw = int(uniforms.get("num_inputs", 2))
    num_inputs = max(1, min(4, num_inputs_raw))
    input_bindings = []
    accum_lines = []
    for i in range(num_inputs):
        input_bindings.append(
            f"@group(0) @binding({i}) var<storage, read> in{i}: array<f32>;"
        )
        if i == 0:
            accum_lines.append(f"    result = result + params.w0 * in0[i];")
        else:
            accum_lines.append(f"    if (ni >= {i+1}) {{ result = result + params.w{i} * in{i}[i]; }}")

    binding_out = num_inputs
    binding_uniform = num_inputs + 1

    accum_code = "    var result = params.bias;\n" + "\n".join(accum_lines)

    return f"""
struct Params {{
    num_inputs: f32,
    w0: f32,
    w1: f32,
    w2: f32,
    w3: f32,
    bias: f32,
}};

{chr(10).join(input_bindings)}
@group(0) @binding({binding_out}) var<storage, read_write> dst: array<f32>;
@group(0) @binding({binding_uniform}) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
    let n = arrayLength(&in0);
    if (i >= n) {{ return; }}
    let ni = i32(params.num_inputs);
    {accum_code}
    dst[i] = result;
}}
"""


__all__ = ["wgsl"]