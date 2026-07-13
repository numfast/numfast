# Viewport Engine — WGSL шейдеры
# RenderPass с storage buffer + uniform buffer

VIEWPORT_VERTEX = """
struct Viewport {
    x_min: u32,
    x_max: u32,
    y_min: f32,
    y_max: f32,
}

@group(0) @binding(0) var<uniform> vp: Viewport;
@group(0) @binding(1) var<storage, read> data: array<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let vi = vp.x_min + idx;
    let val = data[vi];
    let vp_size = f32(vp.x_max - vp.x_min);
    let y_range = vp.y_max - vp.y_min;
    let x_norm = (f32(idx) + 0.5) / vp_size * 2.0 - 1.0;
    let y_norm = select(0.0, (val - vp.y_min) / y_range * 2.0 - 1.0, y_range > 0.0);
    return VertexOutput(vec4<f32>(x_norm, y_norm, 0.0, 1.0));
}
"""

VIEWPORT_FRAGMENT = """
@group(0) @binding(2) var<uniform> color: vec4<f32>;

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return color;
}
"""
