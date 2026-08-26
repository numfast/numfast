"""Reduce WGSL — свёртка элементов: sum / min / max на GPU.

Двухступенчатая редукция:
  1. Каждый поток сворачивает свой chunk (шаг 256) в partials[id.x].
  2. Поток 0 сводит partials[0..min(256,n)) в dst[0].

op (uniform, f32): 0 = sum, 1 = min, 2 = max.
Аккумуляция по первому прочитанному элементу (флаг first):
min/max не зависят от инициализации, sum корректен при n >= 1.

Защита от poisoning (workspace может содержать мусор/нули):
  - запись partials только при id.x < n;
  - финальный цикл ограничен min(256u, n) — при n < 256
    незаписанные partials не читаются;
  - workgroupBarrier() перед чтением partials потоком 0 —
    все записи partials делает workgroup 0.
"""

WGSL = """
struct Params {
    op: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>,
         @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = arrayLength(&src);

    var acc = 0.0;
    var first = true;
    var i = id.x;
    while (i < n) {
        if (first) {
            acc = src[i];
            first = false;
        } else {
            if (params.op == 0.0) {
                acc = acc + src[i];
            } else if (params.op == 1.0) {
                acc = min(acc, src[i]);
            } else {
                acc = max(acc, src[i]);
            }
        }
        i = i + 256u;
    }

    // Guard: потоки id.x >= n не пишут partials (poisoning-защита).
    if (id.x < n) {
        partials[id.x] = acc;
    }

    // partials пишут только потоки workgroup 0 (id.x < 256 <= 4*n).
    workgroupBarrier();

    if (id.x == 0u) {
        var total = 0.0;
        var first2 = true;
        // Кап: при n < 256 не читать незаписанные partials.
        let limit = min(256u, n);
        for (var j = 0u; j < limit; j = j + 1u) {
            let v = partials[j];
            if (first2) {
                total = v;
                first2 = false;
            } else {
                if (params.op == 0.0) {
                    total = total + v;
                } else if (params.op == 1.0) {
                    total = min(total, v);
                } else {
                    total = max(total, v);
                }
            }
        }
        dst[0] = total;
    }
}
"""

__all__ = ["WGSL"]
