"use strict";

function _translate_formula(formula) {
    return formula.replace(/(?<![.\w])(\d+)(?!\.\d)(?!\w)/g, "$1.0");
}

function _generate_wgsl(formula, num_vars) {
    const vars = [];
    for (let i = 0; i < num_vars; i++) {
        vars.push(String.fromCharCode(65 + i));
    }

    const input_bindings = [];
    for (let i = 0; i < num_vars; i++) {
        input_bindings.push(
            "@group(0) @binding(" + i + ") var<storage, read> in_" + i + ": array<f32>;"
        );
    }

    const var_assignments = [];
    for (let i = 0; i < vars.length; i++) {
        var_assignments.push("    let " + vars[i] + " = in_" + i + "[i];");
    }

    const out_binding = num_vars;
    const uniform_binding = num_vars + 1;
    const wgsl_formula = _translate_formula(formula);

    return "struct Params {\n" +
        "    _pad: f32,\n" +
        "};\n\n" +
        input_bindings.join("\n") + "\n" +
        "@group(0) @binding(" + out_binding + ") var<storage, read_write> dst: array<f32>;\n" +
        "@group(0) @binding(" + uniform_binding + ") var<uniform> params: Params;\n\n" +
        "@compute @workgroup_size(64)\n" +
        "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n" +
        "    let i = id.x;\n" +
        "    let n = arrayLength(&dst);\n" +
        "    if (i >= n) { return; }\n\n" +
        var_assignments.join("\n") + "\n\n" +
        "    dst[i] = " + wgsl_formula + ";\n" +
        "}\n";
}

function wgsl_generator(uniforms) {
    const formula = uniforms.formula || "A";
    const num_vars = parseInt(uniforms.num_vars || 1, 10);
    return _generate_wgsl(formula, num_vars);
}

module.exports = { wgsl_generator };
