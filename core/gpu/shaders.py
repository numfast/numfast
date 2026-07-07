"""WGSL compute shaders для Phase 14.2 / 21.

Все шейдеры используют int32 (i32).
Storage Invariant: min(read()) == 1, поэтому zero-division не требуется.
"""


def sma_wgsl(w: int) -> str:
    """Generate SMA WGSL shader with parametric window size.
    
    Args:
        w: window size (embedded in shader source).
    
    Returns:
        WGSL shader source string.
    """
    return f"""
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = i32(id.x);
    let n = i32(arrayLength(&input));
    
    if (i < {w - 1}) {{
        output[i] = 0;
        return;
    }}
    
    var sum: i32 = 0;
    for (var j: i32 = i - {w} + 1; j <= i; j = j + 1) {{
        sum = sum + input[j];
    }}
    output[i] = sum / {w};
}}
"""


def copy_wgsl() -> str:
    return """
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&input)) { return; }
    output[i] = input[i];
}
"""


def pointwise_wgsl(op: str) -> str:
    """Generate Pointwise WGSL.
    
    Args:
        op: "add", "sub", "mul", "div", or expression like "a + b".
    
    Returns:
        WGSL shader source.
    """
    expr_map = {
        "add": "a[i] + b[i]",
        "sub": "a[i] - b[i]",
        "mul": "a[i] * b[i]",
        "div": "a[i] / b[i]",
    }
    expr = expr_map.get(op, op)
    return f"""
@group(0) @binding(0) var<storage, read> a: array<i32>;
@group(0) @binding(1) var<storage, read> b: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
    if (i >= arrayLength(&a)) {{ return; }}
    output[i] = {expr};
}}
"""


def roc_wgsl(period: int) -> str:
    """Generate ROC WGSL shader.

    ROC[i] = (input[i] - input[i-period]) * ROC_SCALE // input[i-period]
    """
    ROC_SCALE = 100000
    return f"""
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = i32(id.x);
    let n = i32(arrayLength(&input));

    if (i < {period}) {{
        output[i] = 0;
        return;
    }}

    let prev = input[i - {period}];
    // prev >= 1 (Storage Invariant)
    let diff = input[i] - prev;
    // Truncation toward zero via abs + sign restore (matches Python truncation)
    if (diff >= 0) {{
        output[i] = diff * {ROC_SCALE} / prev;
    }} else {{
        output[i] = -((-diff) * {ROC_SCALE} / prev);
    }}
}}
"""


def sliding_wgsl(op: str, w: int) -> str:
    """Generate Sliding Window WGSL (MIN/MAX).

    Args:
        op: "min" or "max".
        w: window size.
    """
    init_val = "2147483647" if op == "min" else "-2147483648"
    compare = "min(a, b)" if op == "min" else "max(a, b)"
    return f"""
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = i32(id.x);
    let n = i32(arrayLength(&input));

    if (i < {w - 1}) {{
        output[i] = 0;
        return;
    }}

    var best: i32 = {init_val};
    for (var j: i32 = i - {w} + 1; j <= i; j = j + 1) {{
        let a = input[j];
        let b = best;
        best = {compare};
    }}
    output[i] = best;
}}
"""


def cci_wgsl(w: int, scale: int) -> str:
    """Generate CCI WGSL shader.

    CCI = (TP - SMA_TP) * 200000 / (3 * MD)
    where TP = (H + L + C) / 3, MD = mean(|TP_j - SMA_TP|).

    Args:
        w: window size.
        scale: CCI_SCALE (1000).

    Returns:
        WGSL shader source string.
    """
    SCALE = scale
    return f"""
@group(0) @binding(0) var<storage, read> high: array<i32>;
@group(0) @binding(1) var<storage, read> low: array<i32>;
@group(0) @binding(2) var<storage, read> close: array<i32>;
@group(0) @binding(3) var<storage, read_write> output: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = i32(id.x);
    let n = i32(arrayLength(&high));
    let w = {w};

    if (i < w - 1) {{
        output[i] = 0;
        return;
    }}

    // Pass 1: sum of TP over window
    var sum_tp: i32 = 0;
    for (var j: i32 = i - w + 1; j <= i; j = j + 1) {{
        let tp = (high[j] + low[j] + close[j]) / 3;
        sum_tp = sum_tp + tp;
    }}
    let sma_tp = sum_tp / w;

    // Pass 2: Mean Deviation
    var md_sum: i32 = 0;
    for (var j: i32 = i - w + 1; j <= i; j = j + 1) {{
        let tp = (high[j] + low[j] + close[j]) / 3;
        let diff = tp - sma_tp;
        md_sum = md_sum + abs(diff);
    }}
    let md = md_sum / w;

    if (md > 0) {{
        let tp_cur = (high[i] + low[i] + close[i]) / 3;
        let diff = tp_cur - sma_tp;
        // Use f32 to avoid i32 overflow on diff * 200000
        let cci_f = f32(diff) * 200000.0 / (3.0 * f32(md));
        output[i] = i32(cci_f);
    }} else {{
        output[i] = 0;
    }}
}}
"""


# ── Add One (y = x + 1) — для Phase 14.1 совместимости ──────

ADD_ONE_WGSL = """
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&input)) { return; }
    output[i] = input[i] + 1;
}
"""


# ── Алиасы ────────────────────────────────────────────────────

SHADERS = {
    "copy": copy_wgsl(),
}
# sma_wgsl(), roc_wgsl(), pointwise_wgsl() are generated on demand


# ── Совместимость с Phase 14.1 тестами ────────────────────────

COPY_WGSL = copy_wgsl()
