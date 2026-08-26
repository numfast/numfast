"""IndexKernel WGSL — creation primitives: const / arange / linspace / tile / repeat.

out[i] зависит ТОЛЬКО от global_invocation_id. Паттерн tile/repeat (mode 3/4) —
обычный GPU-resident входной буфер pat (любой k, binding 2, storage).

dtype-семантика (решение владельца, int32 index):
- params.dtype="float32" -> буферы array<f32>, арифметика f32 (legacy путь).
- params.dtype="int32"   -> буферы array<i32>, ЦЕЛОЧИСЛЕННАЯ арифметика:
  mode 1 arange: v = p0 + p1*i32(i) — ТОЧНО до 2^31 (не ограничено 2^24 f32).
  Переполнение i32 — WRAP (two's complement), документировано.
Выбор варианта — wgsl(uniforms) по uniform-флагу int_mode (0=f32, 1=i32) и
наличию pattern (mode 3/4 -> binding 2); без uniforms
(compiler._extract_wgsl) возвращается f32-вариант без pat.

Uniform-структура содержит ТОЛЬКО скаляры (n, mode, p0..p2, k, rep,
int_mode) = 32 байта: массив внутри uniform запрещён правилами выравнивания
(array stride обязан быть кратен 16; латентный bug прежнего pat_0..63
fast-path). Modes 3/4 читают pattern из storage-binding 2.
"""

_F32_HEAD = """
    var v: f32;
    if (params.mode == 0u) {
        // const(value = p0)
        v = params.p0;
    } else if (params.mode == 1u) {
        // arange(start = p0, step = p1): start + step * i
        v = params.p0 + params.p1 * f32(i);
    } else if (params.mode == 2u) {
        // linspace(a = p0, b = p1, denom = p2): a + (b - a) * i / denom
        v = params.p0 + (params.p1 - params.p0) * f32(i) / params.p2;
    }"""

_F32_PAT = """ else if (params.mode == 3u) {
        // tile(pattern, n): out[i] = pat[i % k] — pat: GPU-resident вход
        v = pat[i % params.k];
    } else {
        // repeat(pattern, repeats=r): out[i] = pat[min(i / rep, k - 1)] (u32 div)
        v = pat[min(i / params.rep, params.k - 1u)];
    }"""

_I32_HEAD = """
    var v: i32;
    if (params.mode == 0u) {
        // const(value = p0)
        v = params.p0;
    } else if (params.mode == 1u) {
        // arange(start = p0, step = p1): EXACT i32 arithmetic up to 2^31.
        // Signed overflow WRAPS (two's complement) — documented contract.
        v = params.p0 + params.p1 * i32(i);
    } else if (params.mode == 2u) {
        // linspace integer-trunc semantics: C-style trunc division,
        // all steps in i32 domain (WRAP on overflow, documented).
        v = params.p0 + (params.p1 - params.p0) * i32(i) / params.p2;
    }"""

_I32_PAT = """ else if (params.mode == 3u) {
        // tile(pattern, n): out[i] = pat[i % k]; pattern values truncate to i32
        v = pat[i % params.k];
    } else {
        // repeat(pattern, repeats=r): out[i] = pat[min(i / rep, k - 1)] (u32 div)
        v = pat[min(i / params.rep, params.k - 1u)];
    }"""


def _make_wgsl(int_mode: bool, has_pat: bool) -> str:
    scalar_t = "i32" if int_mode else "f32"
    head, pat_branch = ((_I32_HEAD, _I32_PAT) if int_mode
                        else (_F32_HEAD, _F32_PAT))
    pat_binding = (f"@group(0) @binding(2) var<storage, read> pat: array<{scalar_t}>;"
                   if has_pat else "")
    return f"""
struct Params {{
    n: u32,
    mode: u32,
    p0: {scalar_t},
    p1: {scalar_t},
    p2: {scalar_t},
    k: u32,
    rep: u32,
    int_mode: u32,
}};

@group(0) @binding(0) var<storage, read_write> dst: array<{scalar_t}>;
@group(0) @binding(1) var<uniform> params: Params;
{pat_binding}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
    if (i >= params.n) {{ return; }}
{head}{pat_branch if has_pat else ""}
    dst[i] = v;
}}
"""


WGSL_FLOAT32 = _make_wgsl(False, False)
WGSL_INT32 = _make_wgsl(True, False)

# Backward-compatible alias: default (float32) shader source.
WGSL = WGSL_FLOAT32


def wgsl(uniforms: dict) -> str:
    """Select shader variant by uniforms: int_mode flag + pattern presence.

    mode 3/4 (tile/repeat) -> storage pat binding included; without uniforms
    (compiler._extract_wgsl) returns the f32 variant without pat.
    """
    int_mode = bool(int(uniforms.get("int_mode", 0)))
    has_pat = int(uniforms.get("mode", 0)) in (3, 4)
    return _make_wgsl(int_mode, has_pat)


__all__ = ["WGSL", "WGSL_FLOAT32", "WGSL_INT32", "wgsl"]
