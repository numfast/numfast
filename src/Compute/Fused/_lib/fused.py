# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Fused elementwise indicators: ExecutionGraph -> one WGSL source, one dispatch.

SPEC-DELTA-11 (v1 scope, honest boundary):
  FUSES: one float32 input series (length N) -> K outputs, each a pure
    function of the input neighbourhood, computed by one thread per row:
    pointwise maps (axpb, map_div), lag reads (mom, roc, returns), sliding
    windows with the SAME N output domain (sma, rsum, rmin, rmax, rstd, rsi,
    boll_up, boll_lo, zscore, stoch_k), and i32 signal masks derived inside
    the same thread (above_sma, rsi_lt, rsi_gt, cond_and over earlier i32
    outputs). min_periods = window (full windows only, IR default).
  NOT FUSED in v1 (explicit error, never silent): groupby/groupby_multi
    (different key spaces, per-key atomic domains sized by M != N), sort
    (global order, multi-pass bitonic), join/lookup (hash tables, probe
    order), scalar reductions to M=1 (tree + host merge need a second
    dispatch shape), EMA/recursive filters (cross-thread sequential
    dependency: y[i] needs y[i-1]), multi-input graphs (one input v1),
    non-float32 input (cast first).
  Self-contained emitters: every output recomputes what it needs from the
    input (no cross-output reads except cond_and over earlier i32 masks), so
    fused[K outputs] vs separate[K x 1 output] run identical instruction
    streams per output -> bit-exact by construction (test asserts it).
  Uniform contract (DELTA-9 style): N/windows/lags ride pu (array<vec4<u32>,2>,
    pu[0].x=N, slots allocated at build, literal lane indices emitted -- no
    dynamic indexing, stride-16 uniform rule honored); float scalars ride pf
    (array<vec4<f32>,2>). Source depends only on op sequence + dtypes, never
    on values/shapes -> pipeline cached across N.
  Caps v1: N <= 65535*256 (one dispatch dim x), <= 7 int params, <= 8 float
    params, float32 only. Beyond -> explicit error with fix.
  No imports from other Extensions (flat assembly): own minimal wgpu
    DeviceContext (GPU._lib internals are not PUBLIC surface, depending on
    them would need touching the frozen GPU extension).
"""

import time as _time

import numpy as _np

_WG = 256
_MAX_DISPATCH_X = 65535
MAX_N = _MAX_DISPATCH_X * _WG  # 16_776_960 rows, one dispatch dim x
_PU_SLOTS = 8  # pu[0] = N, pu[1..7] = int params (windows/lags/ddof)
_PF_SLOTS = 8  # float params (k-mult, levels, a/b, divisor)
_NAN = "bitcast<f32>(0x7FC00000u)"

# op -> (out_dtype, int_params, float_params, needs: window|lag|none|masks)
_OPS = {
    "sma": ("f32", ["w"], []),
    "rsum": ("f32", ["w"], []),
    "rmin": ("f32", ["w"], []),
    "rmax": ("f32", ["w"], []),
    "rstd": ("f32", ["w", "ddof"], []),
    "mom": ("f32", ["lag"], []),
    "roc": ("f32", ["lag"], []),
    "returns": ("f32", [], []),
    "rsi": ("f32", ["w"], []),
    "boll_up": ("f32", ["w"], ["k"]),
    "boll_lo": ("f32", ["w"], ["k"]),
    "zscore": ("f32", ["w"], []),
    "stoch_k": ("f32", ["w"], []),
    "axpb": ("f32", [], ["a", "b"]),
    "map_div": ("f32", [], ["v"]),
    "above_sma": ("i32", ["w"], []),
    "rsi_lt": ("i32", ["w"], ["level"]),
    "rsi_gt": ("i32", ["w"], ["level"]),
    "cond_and": ("i32", [], []),
}

_NO_FUSE_HINT = (
    "fused v1 fuses elementwise chains + same-N sliding windows over ONE "
    "f32 series (sma/rsum/rmin/rmax/rstd/mom/roc/returns/rsi/boll_up/boll_lo/"
    "zscore/stoch_k/axpb/map_div + i32 signals above_sma/rsi_lt/rsi_gt/"
    "cond_and). NOT fused v1: groupby (key spaces), sort (global order), "
    "join/lookup, scalar reductions, EMA (sequential). "
    "See specs/delta-11-fused-elementwise.md"
)


def _err(what, fix):
    return ValueError(f"{what} Fix: {fix}. "
                      "See specs/delta-11-fused-elementwise.md")


def _is_finite_float(v):
    try:
        import math as _m
        return _m.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def compile_spec(outputs):
    """Validate output specs -> compiled plan (pure, no GPU).

    outputs: [{name, op, params?}]. cond_and params: {a, b} = earlier i32
      output names. Returns {source, names, dtypes, pu_vals, pf_vals, n_int,
      n_float} where pu_vals/pf_vals are full slot arrays (pu[0] filled at
      run with N). Raises ValueError on any boundary violation.
    """
    if not isinstance(outputs, list) or not outputs:
        raise _err("fused needs a non-empty [outputs] list.",
                   "pass e.g. [{'name': 'sma20', 'op': 'sma', "
                   "'params': {'w': 20}}]")
    names, seen = [], set()
    for i, o in enumerate(outputs):
        if not isinstance(o, dict):
            raise _err(f"output #{i} must be {{name, op, params?}}.",
                       "pass dicts with 'name' and 'op'")
        nm, op = o.get("name"), o.get("op")
        if not isinstance(nm, str) or not nm:
            raise _err(f"output #{i} needs a non-empty str name.",
                       "pass 'name': 'sma20'")
        if nm in seen:
            raise _err(f"duplicate output name '{nm}'.",
                       "give each output a unique name")
        seen.add(nm)
        names.append(nm)
        if op not in _OPS:
            raise _err(f"unknown fused op '{op}' for '{nm}'. {_NO_FUSE_HINT}",
                       "use a fused v1 op, or backend='cpu' for the rest")
    champ = {}  # name -> index, for cond_and deps
    for i, nm in enumerate(names):
        champ[nm] = i
    pu_slot, pf_slot = 1, 0  # pu[0] reserved for N
    plan = []
    for i, o in enumerate(outputs):
        op = o["op"]
        prm = dict(o.get("params") or {})
        dt, iparams, fparams = _OPS[op]
        res = {}
        for p in iparams:
            if p == "ddof":
                d = int(prm.get("ddof", 0))
                w = int(prm.get("w", 0))
                if d < 0 or (w and d >= w):
                    raise _err(f"rstd '{names[i]}': need 0 <= ddof < w, "
                               f"got ddof={d} w={w}.", "pass ddof=0/1, w>ddof")
                res["ddof_v"] = d
            elif p == "w":
                w = prm.get("w")
                if isinstance(w, bool) or not isinstance(w, int) or w < 1:
                    raise _err(f"'{names[i]}': window w must be int >= 1, "
                               f"got {w!r}.", "pass e.g. params={'w': 20}")
                res["w_v"] = w
            elif p == "lag":
                lag = prm.get("lag")
                if (isinstance(lag, bool) or not isinstance(lag, int)
                        or lag < 1):
                    raise _err(f"'{names[i]}': lag must be int >= 1, "
                               f"got {lag!r}.", "pass e.g. params={'lag': 5}")
                res["lag_v"] = lag
            if pu_slot >= _PU_SLOTS:
                raise _err(f"'{names[i]}': int-param budget exceeded "
                           f"(>{_PU_SLOTS - 1} windows/lags/ddofs).",
                           "split into two fused runs (two dispatches)")
            res[p] = pu_slot
            pu_slot += 1
        for p in fparams:
            v = prm.get(p)
            if not _is_finite_float(v):
                raise _err(f"'{names[i]}': float param '{p}' must be "
                           f"finite, got {v!r}.", f"pass params={{'{p}': ...}}")
            if pf_slot >= _PF_SLOTS:
                raise _err(f"'{names[i]}': float-param budget exceeded "
                           f"(>{_PF_SLOTS}).",
                           "split into two fused runs (two dispatches)")
            res[p] = pf_slot
            res[p + "_v"] = float(v)
            pf_slot += 1
        if op == "cond_and":
            for k, dk in (("a", "dep_a"), ("b", "dep_b")):
                dep = prm.get(k)
                if dep not in champ or champ[dep] >= i:
                    raise _err(f"cond_and '{names[i]}': '{k}' must name an "
                               f"EARLIER i32 output, got {dep!r}.",
                               "list signal masks before their cond_and")
                j = champ[dep]
                if outputs[j]["op"] not in ("above_sma", "rsi_lt", "rsi_gt",
                                            "cond_and"):
                    raise _err(f"cond_and '{names[i]}': '{dep}' is "
                               f"'{outputs[j]['op']}', not an i32 mask.",
                               "pass above_sma/rsi_lt/rsi_gt/cond_and outputs")
                res[dk] = j
        plan.append({"name": names[i], "op": op, "dt": dt, "res": res,
                     "idx": i})
    pu_vals = [0] * _PU_SLOTS
    pf_vals = [0.0] * _PF_SLOTS
    for p in plan:
        r = p["res"]
        for k, vk in (("w", "w_v"), ("lag", "lag_v"), ("ddof", "ddof_v")):
            if k in r:
                pu_vals[r[k]] = int(r[vk])
        for k in ("k", "a", "b", "v", "level"):
            if k in r:
                pf_vals[r[k]] = float(r[k + "_v"])
    src = _emit(plan, len(outputs))
    return {"source": src, "names": names,
            "dtypes": [p["dt"] for p in plan], "plan": plan,
            "pu_vals": pu_vals, "pf_vals": pf_vals,
            "n_int": pu_slot - 1, "n_float": pf_slot}


_LANE = "xyzw"


def _pu(s):
    """Uniform u32 slot access (vec4-packed: stride-16 rule)."""
    return f"pu[{s // 4}].{_LANE[s % 4]}"


def _pf(s):
    """Uniform f32 slot access (vec4-packed: stride-16 rule)."""
    return f"pf[{s // 4}].{_LANE[s % 4]}"


def _isnan(v):
    """NaN test via exponent/mantissa bits (this naga lacks isNan builtin;
    `x != x` is folded to false -- never use it for NaN)."""
    return (f"((bitcast<u32>({v}) & 0x7F800000u) == 0x7F800000u && "
            f"(bitcast<u32>({v}) & 0x007FFFFFu) != 0u)")


def _wnd(pfx, wi):
    return (f"let {pfx}w: u32 = {_pu(wi)};\n"
            f"  let {pfx}lo: u32 = select(0u, i + 1u - {pfx}w, "
            f"i + 1u >= {pfx}w);\n"
            f"  let {pfx}cnt: u32 = i - {pfx}lo + 1u;")


def _emit_stmt(p, out_var):
    """One output's statements (identical text in fused & separate builds)."""
    op, r, t = p["op"], p["res"], f"t{p['idx']}_"
    if op == "sma":
        return (f"{_wnd(t, r['w'])}\n"
                f"  var {t}acc: f32 = 0.0f;\n"
                f"  for (var {t}j: u32 = {t}lo; {t}j <= i; "
                f"{t}j = {t}j + 1u) {{ {t}acc = {t}acc + x[{t}j]; }}\n"
                f"  var {t}v: f32 = {_NAN};\n"
                f"  if ({t}cnt >= {t}w) {{ {t}v = {t}acc / f32({t}cnt); }}\n"
                f"  {out_var}[i] = {t}v;")
    if op == "rsum":
        return (f"{_wnd(t, r['w'])}\n"
                f"  var {t}acc: f32 = 0.0f;\n"
                f"  for (var {t}j: u32 = {t}lo; {t}j <= i; "
                f"{t}j = {t}j + 1u) {{ {t}acc = {t}acc + x[{t}j]; }}\n"
                f"  var {t}v: f32 = {_NAN};\n"
                f"  if ({t}cnt >= {t}w) {{ {t}v = {t}acc; }}\n"
                f"  {out_var}[i] = {t}v;")
    if op in ("rmin", "rmax"):
        fn = "min" if op == "rmin" else "max"
        return (f"{_wnd(t, r['w'])}\n"
                f"  var {t}m: f32 = x[{t}lo];\n"
                f"  var {t}bad: u32 = 0u;\n"
                f"  if ({_isnan(t + 'm')}) {{ {t}bad = 1u; }}\n"
                f"  for (var {t}j: u32 = {t}lo + 1u; {t}j <= i; "
                f"{t}j = {t}j + 1u) {{\n"
                f"    let {t}vj: f32 = x[{t}j];\n"
                f"    if ({_isnan(t + 'vj')}) {{ {t}bad = 1u; }}\n"
                f"    else if ({t}bad == 0u) {{ {t}m = {fn}({t}m, {t}vj); }}\n"
                f"  }}\n"
                f"  var {t}v: f32 = {_NAN};\n"
                f"  if ({t}cnt >= {t}w && {t}bad == 0u) {{ {t}v = {t}m; }}\n"
                f"  {out_var}[i] = {t}v;")
    if op == "rstd":
        return (f"{_wnd(t, r['w'])}\n"
                f"  var {t}acc: f32 = 0.0f;\n"
                f"  for (var {t}j: u32 = {t}lo; {t}j <= i; "
                f"{t}j = {t}j + 1u) {{ {t}acc = {t}acc + x[{t}j]; }}\n"
                f"  let {t}mean: f32 = {t}acc / f32({t}cnt);\n"
                f"  var {t}ss: f32 = 0.0f;\n"
                f"  for (var {t}j: u32 = {t}lo; {t}j <= i; "
                f"{t}j = {t}j + 1u) {{ let {t}e: f32 = x[{t}j] - {t}mean; "
                f"{t}ss = {t}ss + {t}e * {t}e; }}\n"
                f"  var {t}v: f32 = {_NAN};\n"
                f"  if ({t}cnt >= {t}w) {{ {t}v = "
                f"sqrt({t}ss / f32({t}cnt - {r['ddof_v']}u)); }}\n"
                f"  {out_var}[i] = {t}v;")
    if op == "mom":
        L = r["lag_v"]
        return (f"var {t}v: f32 = {_NAN};\n"
                f"  if (i >= {L}u) {{ {t}v = x[i] - x[i - {L}u]; }}\n"
                f"  {out_var}[i] = {t}v;")
    if op == "roc":
        L = r["lag_v"]
        return (f"var {t}v: f32 = {_NAN};\n"
                f"  if (i >= {L}u) {{ {t}v = "
                f"(x[i] / x[i - {L}u] - 1.0f) * 100.0f; }}\n"
                f"  {out_var}[i] = {t}v;")
    if op == "returns":
        return (f"var {t}v: f32 = {_NAN};\n"
                f"  if (i >= 1u) {{ {t}v = x[i] / x[i - 1u] - 1.0f; }}\n"
                f"  {out_var}[i] = {t}v;")
    if op == "rsi":
        return (f"{_wnd(t, r['w'])}\n"
                f"  var {t}g: f32 = 0.0f;\n"
                f"  var {t}l: f32 = 0.0f;\n"
                f"  var {t}bad: u32 = 0u;\n"
                f"  for (var {t}j: u32 = {t}lo; {t}j <= i; "
                f"{t}j = {t}j + 1u) {{\n"
                f"    let {t}xj: f32 = x[{t}j];\n"
                f"    if ({_isnan(t + 'xj')}) {{ {t}bad = 1u; }}\n"
                f"    else if ({t}j > 0u) {{\n"
                f"      let {t}xj1: f32 = x[{t}j - 1u];\n"
                f"      if ({_isnan(t + 'xj1')}) {{ {t}bad = 1u; }}\n"
                f"      else {{ let {t}d: f32 = {t}xj - {t}xj1; "
                f"{t}g = {t}g + max({t}d, 0.0f); "
                f"{t}l = {t}l + max(-{t}d, 0.0f); }}\n"
                f"    }}\n"
                f"  }}\n"
                f"  var {t}v: f32 = {_NAN};\n"
                f"  if ({t}cnt >= {t}w && {t}bad == 0u) {{\n"
                f"    let {t}ag: f32 = {t}g / f32({t}cnt);\n"
                f"    let {t}al: f32 = {t}l / f32({t}cnt);\n"
                f"    if ({t}al == 0.0f) {{ {t}v = select(100.0f, 50.0f, "
                f"{t}ag == 0.0f); }}\n"
                f"    else {{ {t}v = 100.0f - 100.0f / "
                f"(1.0f + {t}ag / {t}al); }}\n"
                f"  }}\n"
                f"  {out_var}[i] = {t}v;")
    if op in ("boll_up", "boll_lo"):
        sgn = "+" if op == "boll_up" else "-"
        return (f"{_wnd(t, r['w'])}\n"
                f"  var {t}acc: f32 = 0.0f;\n"
                f"  for (var {t}j: u32 = {t}lo; {t}j <= i; "
                f"{t}j = {t}j + 1u) {{ {t}acc = {t}acc + x[{t}j]; }}\n"
                f"  let {t}mean: f32 = {t}acc / f32({t}cnt);\n"
                f"  var {t}ss: f32 = 0.0f;\n"
                f"  for (var {t}j: u32 = {t}lo; {t}j <= i; "
                f"{t}j = {t}j + 1u) {{ let {t}e: f32 = x[{t}j] - {t}mean; "
                f"{t}ss = {t}ss + {t}e * {t}e; }}\n"
                f"  var {t}v: f32 = {_NAN};\n"
                f"  if ({t}cnt >= {t}w) {{ {t}v = {t}mean {sgn} "
                f"{_pf(r['k'])} * sqrt({t}ss / f32({t}cnt)); }}\n"
                f"  {out_var}[i] = {t}v;")
    if op == "zscore":
        return (f"{_wnd(t, r['w'])}\n"
                f"  var {t}acc: f32 = 0.0f;\n"
                f"  for (var {t}j: u32 = {t}lo; {t}j <= i; "
                f"{t}j = {t}j + 1u) {{ {t}acc = {t}acc + x[{t}j]; }}\n"
                f"  let {t}mean: f32 = {t}acc / f32({t}cnt);\n"
                f"  var {t}ss: f32 = 0.0f;\n"
                f"  for (var {t}j: u32 = {t}lo; {t}j <= i; "
                f"{t}j = {t}j + 1u) {{ let {t}e: f32 = x[{t}j] - {t}mean; "
                f"{t}ss = {t}ss + {t}e * {t}e; }}\n"
                f"  let {t}sd: f32 = sqrt({t}ss / f32({t}cnt));\n"
                f"  var {t}v: f32 = {_NAN};\n"
                f"  if ({t}cnt >= {t}w) {{ {t}v = select((x[i] - {t}mean) / "
                f"{t}sd, 0.0f, {t}sd == 0.0f); }}\n"
                f"  {out_var}[i] = {t}v;")
    if op == "stoch_k":
        return (f"{_wnd(t, r['w'])}\n"
                f"  var {t}hi: f32 = x[{t}lo];\n"
                f"  var {t}lw: f32 = x[{t}lo];\n"
                f"  var {t}bad: u32 = 0u;\n"
                f"  if ({_isnan(t + 'hi')}) {{ {t}bad = 1u; }}\n"
                f"  for (var {t}j: u32 = {t}lo + 1u; {t}j <= i; "
                f"{t}j = {t}j + 1u) {{\n"
                f"    let {t}vj: f32 = x[{t}j];\n"
                f"    if ({_isnan(t + 'vj')}) {{ {t}bad = 1u; }}\n"
                f"    else if ({t}bad == 0u) {{ {t}hi = max({t}hi, {t}vj); "
                f"{t}lw = min({t}lw, {t}vj); }}\n"
                f"  }}\n"
                f"  var {t}v: f32 = {_NAN};\n"
                f"  if ({t}cnt >= {t}w && {t}bad == 0u) {{\n"
                f"    {t}v = select((x[i] - {t}lw) / ({t}hi - {t}lw) * 100.0f, "
                f"50.0f, {t}hi == {t}lw);\n"
                f"  }}\n"
                f"  {out_var}[i] = {t}v;")
    if op == "axpb":
        return (f"{out_var}[i] = {_pf(r['a'])} * x[i] + {_pf(r['b'])};")
    if op == "map_div":
        return (f"{out_var}[i] = x[i] / {_pf(r['v'])};")
    if op == "above_sma":
        # sub-emitter stores into a scalar (no temp arrays in WGSL locals).
        sma = _emit_stmt({"op": "sma", "res": {"w": r["w"]},
                          "idx": p["idx"]}, "@@")
        sma = sma.replace(f"@@[i] = {t}v;", f"let {t}s: f32 = {t}v;")
        return (f"{sma}\n"
                f"  {out_var}[i] = select(0i, 1i, x[i] > {t}s);")
    if op in ("rsi_lt", "rsi_gt"):
        cmp = "<" if op == "rsi_lt" else ">"
        rsi = _emit_stmt({"op": "rsi", "res": {"w": r["w"]},
                          "idx": p["idx"]}, "@@")
        rsi = rsi.replace(f"@@[i] = {t}v;", f"let {t}r: f32 = {t}v;")
        return (f"{rsi}\n"
                f"  {out_var}[i] = select(0i, 1i, "
                f"{t}r {cmp} {_pf(r['level'])});")
    if op == "cond_and":
        return (f"{out_var}[i] = o{r['dep_a']}[i] * o{r['dep_b']}[i];")
    raise _err(f"no emitter for '{op}'.", "use a fused v1 op")


def _emit(plan, k):
    decls = ["@group(0) @binding(0) var<storage,read> x: array<f32>;"]
    for j, p in enumerate(plan):
        decls.append(f"@group(0) @binding({j + 1}) "
                     f"var<storage,read_write> o{j}: array<{p['dt']}>;")
    decls.append(f"@group(0) @binding({k + 1}) var<uniform> pu: array<vec4<u32>, "
                 f"{_PU_SLOTS // 4}>;")
    decls.append(f"@group(0) @binding({k + 2}) var<uniform> pf: array<vec4<f32>, "
                 f"{_PF_SLOTS // 4}>;")
    body = "\n".join(f"  {_emit_stmt(p, 'o' + str(p['idx']))}"
                     for p in plan)
    return ("\n".join(decls) + "\n@compute @workgroup_size(256)\n"
            "fn main(@builtin(global_invocation_id) g: vec3<u32>) {\n"
            "  let i: u32 = g.x;\n"
            "  let n: u32 = pu[0].x;\n"
            "  if (i >= n) { return; }\n"
            f"{body}\n}}")


# ---- local wgpu dispatch (own context; GPU._lib is not PUBLIC surface) ----
_CTX = None


class _DeviceContext:
    def __init__(self):
        import wgpu
        a = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.dev = a.request_device_sync()
        self.pipes = {}
        self.hits = 0
        self.misses = 0

    def pipe(self, src, types):
        key = (src, tuple(types))
        p = self.pipes.get(key)
        if p is None:
            import wgpu
            dev = self.dev
            entries = [{"binding": i,
                        "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": t}} for i, t in enumerate(types)]
            bgl = dev.create_bind_group_layout(entries=entries)
            sm = dev.create_shader_module(code=src)
            pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])
            p = (dev.create_compute_pipeline(
                layout=pl, compute={"module": sm,
                                    "entry_point": "main"}), bgl)
            self.pipes[key] = p
            self.misses += 1
        else:
            self.hits += 1
        return p

    def stats(self):
        return {"pipelines": len(self.pipes), "hits": self.hits,
                "misses": self.misses}


def _ctx():
    global _CTX
    if _CTX is None:
        _CTX = _DeviceContext()
    return _CTX


def _submit(dev, pipe, bgl, allb, n_threads):
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe)
    cp.set_bind_group(0, dev.create_bind_group(
        layout=bgl,
        entries=[{"binding": i, "resource": {"buffer": b, "offset": 0,
                                            "size": b.size}}
                 for i, b in enumerate(allb)]), [], 0, 0)
    cp.dispatch_workgroups(max(1, (int(n_threads) + _WG - 1) // _WG), 1, 1)
    cp.end()
    dev.queue.submit([enc.finish()])


def r_upload(arr):
    """Resident upload: numpy -> device buffer (caller owns)."""
    import wgpu
    return _ctx().dev.create_buffer_with_data(
        data=_np.ascontiguousarray(arr).tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        | wgpu.BufferUsage.COPY_DST)


def r_alloc(nbytes):
    """Resident alloc: device-only buffer (min 4, wgpu floor)."""
    import wgpu
    return _ctx().dev.create_buffer(
        size=max(4, int(nbytes)),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        | wgpu.BufferUsage.COPY_DST)


def r_download(buf, dtype, n=None):
    """Resident download: device buffer -> numpy copy (graph exit only)."""
    raw = bytes(_ctx().dev.queue.read_buffer(buf))
    a = _np.frombuffer(raw, dtype=dtype).copy()
    return a if n is None else a[:int(n)].copy()


def execute_compiled(comp, x=None, x_buf=None):
    """Run a compiled plan: exactly ONE dispatch (zero-size: zero).

    x: f32 numpy (uploaded here) or x_buf: resident device buffer (bench
      reuse: upload once, dispatch many). Returns (results, info) with
      info {dispatches, n, h2d_ms, submit_ms, d2h_ms, e2e_ms, ...}.
    """
    import wgpu
    if (x is None) == (x_buf is None):
        raise _err("execute needs exactly one of x / x_buf.",
                   "pass x=<f32 array> or x_buf=<resident buffer>")
    t0 = _time.perf_counter()
    if x_buf is None:
        xa = _np.asarray(x)
        if xa.dtype != _np.dtype(_np.float32) or xa.ndim != 1:
            raise _err(f"fused v1 needs rank-1 float32 input, got "
                       f"{xa.dtype} {xa.ndim}-D (no silent casts).",
                       "cast with .astype(np.float32), 1-D")
        xa = _np.ascontiguousarray(xa).ravel()
        n = int(xa.size)
        if n > MAX_N:
            raise _err(f"fused v1 N={n} > cap {MAX_N} (dispatch dim x).",
                       "chunk the series (two fused runs) or backend='cpu'")
        dev = _ctx().dev
        b_x = (r_upload(xa) if n else None)
        h2d_ms = (_time.perf_counter() - t0) * 1000
    else:
        dev = _ctx().dev
        n = int(x_buf.size // 4)
        h2d_ms = 0.0
    k = len(comp["names"])
    if n == 0:
        res = [_np.zeros(0, dtype=_np.float32 if dt == "f32" else _np.int32)
               for dt in comp["dtypes"]]
        return res, {"dispatches": 0, "n": 0, "h2d_ms": h2d_ms,
                     "submit_ms": 0.0, "d2h_ms": 0.0, "e2e_ms": h2d_ms,
                     "h2d_bytes": 0, "d2h_bytes": 0}
    pu = _np.ascontiguousarray([n] + comp["pu_vals"][1:], dtype=_np.uint32)
    pf = _np.ascontiguousarray(comp["pf_vals"], dtype=_np.float32)
    outs = [r_alloc(n * 4) for _ in range(k)]
    ub1 = dev.create_buffer_with_data(
        data=pu.tobytes(),
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    ub2 = dev.create_buffer_with_data(
        data=pf.tobytes(),
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    b_x = b_x if x_buf is None else x_buf
    allb = [b_x] + outs + [ub1, ub2]
    types = ([wgpu.BufferBindingType.read_only_storage]
             + [wgpu.BufferBindingType.storage] * k
             + [wgpu.BufferBindingType.uniform] * 2)
    t1 = _time.perf_counter()
    pipe, bgl = _ctx().pipe(comp["source"], types)
    _submit(dev, pipe, bgl, allb, n)
    t2 = _time.perf_counter()
    res = [r_download(b, _np.float32 if dt == "f32" else _np.int32, n)
           for b, dt in zip(outs, comp["dtypes"])]
    t3 = _time.perf_counter()
    return res, {"dispatches": 1, "n": n, "h2d_ms": h2d_ms,
                 "submit_ms": (t2 - t1) * 1000, "d2h_ms": (t3 - t2) * 1000,
                 "e2e_ms": (t3 - t0) * 1000,
                 "h2d_bytes": n * 4 if x_buf is None else 0,
                 "d2h_bytes": n * 4 * k}


# ---- PUBLIC surface (Builder alias) ----
def fused_build(outputs):
    """compile_spec alias: validate + emit one WGSL source (pure)."""
    return compile_spec(outputs)


def fused_source(outputs):
    """WGSL source string for an output spec (inspect/debug)."""
    return compile_spec(outputs)["source"]


def fused_run(x, outputs):
    """Upload x once -> ONE dispatch -> K outputs. Returns (results, info)."""
    xa = _np.asarray(x)
    if xa.dtype != _np.dtype(_np.float32) or xa.ndim != 1:
        raise _err(f"fused v1 needs rank-1 float32 input, got {xa.dtype} "
                   f"{xa.ndim}-D (no silent casts).",
                   "cast with .astype(np.float32), 1-D")
    return execute_compiled(compile_spec(outputs), x=xa)


def fused_oracle(x, outputs):
    """Numpy float64 reference (same edge semantics; tolerance, not bits)."""
    xa = _np.ascontiguousarray(x, dtype=_np.float64).ravel()
    n = int(xa.size)
    comp = compile_spec(outputs)  # validates + resolves deps
    out = {}
    for p in comp["plan"]:
        out[p["name"]] = _oracle_one(xa, out, p, comp)
    return [out[nm] for nm in comp["names"]]


def _oracle_one(x, sib, p, comp):
    import math as _m
    n = int(x.size)
    op, r = p["op"], p["res"]
    if op == "cond_and":
        return (sib[comp["names"][r["dep_a"]]]
                * sib[comp["names"][r["dep_b"]]]).astype(_np.int32)
    f = _np.full(n, _m.nan)
    if op in ("sma", "rsum", "rmin", "rmax", "rstd", "boll_up", "boll_lo",
              "zscore", "stoch_k", "rsi"):
        w = int(r["w_v"])
        for i in range(n):
            lo = max(0, i - w + 1)
            win = x[lo:i + 1]
            if i + 1 < w or _np.isnan(win).any():
                continue
            if op == "sma":
                f[i] = win.mean()
            elif op == "rsum":
                f[i] = win.sum()
            elif op == "rmin":
                f[i] = win.min()
            elif op == "rmax":
                f[i] = win.max()
            elif op == "rstd":
                f[i] = win.std(ddof=int(r["ddof_v"]))
            elif op in ("boll_up", "boll_lo"):
                m, s = win.mean(), win.std()
                f[i] = m + (1.0 if op == "boll_up" else -1.0) * r["k_v"] * s
            elif op == "zscore":
                s = win.std()
                f[i] = 0.0 if s == 0.0 else (x[i] - win.mean()) / s
            elif op == "stoch_k":
                hi, lw = win.max(), win.min()
                f[i] = 50.0 if hi == lw else (x[i] - lw) / (hi - lw) * 100.0
            elif op == "rsi":
                # deltas d_j need x[j-1]: NaN in x[max(0,lo-1)..i] poisons
                # (mirrors the WGSL bad-flag over x[j] and x[j-1]).
                lo2 = max(0, lo - 1) if lo > 0 else 0
                if _np.isnan(x[lo2:i + 1]).any():
                    continue
                g = l = 0.0
                for j in range(lo, i + 1):
                    if j == 0:
                        continue
                    d = x[j] - x[j - 1]
                    g += max(d, 0.0)
                    l += max(-d, 0.0)
                ag, al = g / (i - lo + 1), l / (i - lo + 1)
                f[i] = (50.0 if ag == 0.0 else 100.0) if al == 0.0 else (
                    100.0 - 100.0 / (1.0 + ag / al))
        return f.astype(_np.float32)
    if op == "mom":
        lag = int(r["lag_v"])
        if lag < n:
            f[lag:] = x[lag:] - x[:-lag or None]
        return f.astype(_np.float32)
    if op == "roc":
        lag = int(r["lag_v"])
        if lag < n:
            with _np.errstate(divide="ignore", invalid="ignore"):
                f[lag:] = (x[lag:] / x[:-lag or None] - 1.0) * 100.0
        return f.astype(_np.float32)
    if op == "returns":
        if n > 1:
            with _np.errstate(divide="ignore", invalid="ignore"):
                f[1:] = x[1:] / x[:-1] - 1.0
        return f.astype(_np.float32)
    if op == "axpb":
        return (r["a_v"] * x + r["b_v"]).astype(_np.float32)
    if op == "map_div":
        with _np.errstate(divide="ignore", invalid="ignore"):
            return (x / r["v_v"]).astype(_np.float32)
    if op == "above_sma":
        s = _oracle_one(x, sib, {"op": "sma", "res": r}, comp)
        return (x > s).astype(_np.int32)
    if op in ("rsi_lt", "rsi_gt"):
        v = _oracle_one(x, sib, {"op": "rsi", "res": r}, comp)
        return ((v < r["level_v"]) if op == "rsi_lt"
                else (v > r["level_v"])).astype(_np.int32)
    raise _err(f"no oracle for '{op}'.", "use a fused v1 op")
