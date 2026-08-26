# Example 001: GPU Vector Math — Spec

## Назначение

Продемонстрировать базовый цикл NumFast:
1. Описать ядро (describe)
2. Реализовать CPU-эталон (cpu)
3. Написать WGSL-шейдер
4. Зарегистрировать ядро в Runtime
5. Выполнить на CPU и GPU
6. Сравнить результаты

## Ядра

### 1. Copy

**Описание:** Поэлементное копирование массива.

```
out[i] = in[i]
```

**Параметры:** нет

**Сигнатура:**
- inputs: data (float)
- outputs: copy (float)

**WGSL:**
```wgsl
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&input)) { return; }
    output[i] = input[i];
}
```

### 2. AddConstant

**Описание:** Прибавить константу к каждому элементу.

```
out[i] = in[i] + c
```

**Параметры:**
- value (float) — константа

**Сигнатура:**
- inputs: data (float)
- outputs: add_{value} (float)
- uniforms: value (float)

**WGSL:**
```wgsl
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> params : Uniform;

struct Uniform {
    value : f32,
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&input)) { return; }
    output[i] = input[i] + params.value;
}
```

### 3. Multiply

**Описание:** Поэлементное умножение двух массивов.

```
out[i] = a[i] * b[i]
```

**Параметры:** нет

**Сигнатура:**
- inputs: a (float), b (float)
- outputs: mul (float)

**WGSL:**
```wgsl
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&a)) { return; }
    output[i] = a[i] * b[i];
}
```

## Проверка

Каждое ядро проверяется:
1. CPU-эталон (python-функция через ExecutionContext)
2. GPU-выполнение (через WebGPU Driver)
3. Сравнение: max_abs_error < 1e-4

## Связанные концепты

- KernelTable: register_kernel(alias, describe, cpu, wgsl)
- ExecutionPlan: inputs, outputs, uniforms
- ExecutionContext: input views, output views, uniforms
- BlockView: read(i), write(i, v), length()
- Runtime: compile(jobs) → execute(tasks)
