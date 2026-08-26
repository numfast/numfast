"use strict";

function cpu(ctx) {
    const formula = ctx.uniforms.formula || "A";
    const num_vars = parseInt(ctx.uniforms.num_vars || 1, 10);

    const dst = ctx.outputs[0].view;
    const n = dst.length();

    const views = {};
    for (let i = 0; i < num_vars; i++) {
        const letter = String.fromCharCode(65 + i);
        views[letter] = ctx.inputs[i].view;
    }

    const safe_globals = {
        abs: Math.abs,
        max: Math.max,
        min: Math.min,
    };

    const fn = new Function(...Object.keys(safe_globals), ...Object.keys(views), "i",
        "const vars = {}; " +
        Object.keys(views).map(v => "vars['" + v + "'] = " + v + "[i]; ").join("") +
        "return (" + formula + ");"
    );

    const args = Object.values(safe_globals).concat(Object.values(views));
    for (let i = 0; i < n; i++) {
        const result = fn(...args, i);
        dst.write(i, result);
    }
}

module.exports = { cpu };
