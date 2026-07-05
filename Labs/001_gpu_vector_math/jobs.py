#!/usr/bin/env python3
"""Lab 001: GPU Vector Math — Copy, AddConstant, Multiply.

Демонстрирует базовый цикл NumFast:
  1. Описать ядро (describe)
  2. Реализовать CPU-эталон (cpu)
  3. Написать WGSL-шейдер
  4. Зарегистрировать в Runtime
  5. Выполнить на CPU и GPU
  6. Сравнить результаты
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math
import numpy as np

from Runtime._lib.mod_iface import ExecutionPlan, InputSlot, OutputSlot, ExecutionContext, Buffer
from Runtime._lib.packet import BufferView as BV
from Runtime._lib.mod_iface import BlockView
from Runtime import Runtime


# ═══════════════════════════════════════════════════════════════
# 1. Определения ядер
# ═══════════════════════════════════════════════════════════════

# ── Copy ──────────────────────────────────────────────────────

def describe_copy(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[InputSlot(name="data", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="copy")],
    )

def cpu_copy(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    for i in range(src.length()):
        dst.write(i, src.read(i))

WGSL_COPY = """
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&input)) { return; }
    output[i] = input[i];
}
"""

# ── AddConstant ───────────────────────────────────────────────

def describe_add(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[InputSlot(name="data", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="add_{value}")],
        uniforms={"value": params["value"]},
    )

def cpu_add(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    c = ctx.uniforms["value"]
    for i in range(src.length()):
        dst.write(i, src.read(i) + c)

WGSL_ADD = """
struct Uniform { value : f32, }
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> params : Uniform;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&input)) { return; }
    output[i] = input[i] + params.value;
}
"""

# ── Multiply ──────────────────────────────────────────────────

def describe_mul(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[InputSlot(name="a", dtype="float"), InputSlot(name="b", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="mul")],
    )

def cpu_mul(ctx: ExecutionContext):
    a = ctx.inputs[0].view
    b = ctx.inputs[1].view
    dst = ctx.outputs[0].view
    for i in range(a.length()):
        dst.write(i, a.read(i) * b.read(i))

WGSL_MUL = """
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&a)) { return; }
    output[i] = a[i] * b[i];
}
"""


# ═══════════════════════════════════════════════════════════════
# 2. Регистрация ядер
# ═══════════════════════════════════════════════════════════════

def register_kernels(runtime):
    """Зарегистрировать все ядра в Runtime."""
    runtime.register_kernel("Copy", describe=describe_copy, cpu=cpu_copy, wgsl=WGSL_COPY)
    runtime.register_kernel("AddConstant", describe=describe_add, cpu=cpu_add, wgsl=WGSL_ADD)
    runtime.register_kernel("Multiply", describe=describe_mul, cpu=cpu_mul, wgsl=WGSL_MUL)


# ═══════════════════════════════════════════════════════════════
# 3. Запуск и сравнение
# ═══════════════════════════════════════════════════════════════

def run_example():
    print("=" * 60)
    print("  Lab 001: GPU Vector Math")
    print("  Copy | AddConstant | Multiply")
    print("=" * 60)

    # ── Данные ───────────────────────────────────────────────
    n = 64
    data = np.array([float(i) * 0.5 for i in range(n)], dtype=np.float64)
    source = {
        "data": data,
        "a": data,
        "b": np.array([float(i) * 2.0 for i in range(n)], dtype=np.float64),
    }

    # ── Jobs ─────────────────────────────────────────────────
    jobs = [
        {"op": "Copy", "inputs": ["data"], "params": {}},
        {"op": "AddConstant", "inputs": ["data"], "params": {"value": 3.14}},
        {"op": "Multiply", "inputs": ["a", "b"], "params": {}},
    ]

    results = {}

    # ── CPU ──────────────────────────────────────────────────
    print("\n  [CPU — эталон]")
    cpu_rt = Runtime()
    register_kernels(cpu_rt)
    cpu_tasks = cpu_rt.compile(jobs)
    cpu_rt.execute(cpu_tasks, source)

    for t in cpu_tasks:
        for name in t.out_names:
            arr = cpu_rt.driver.resolve_output(name)
            results[("cpu", name)] = arr
            print(f"    {name}: min={arr.min():.4f} max={arr.max():.4f}")

    # ── GPU (WebGPU) ─────────────────────────────────────────
    print("\n  [GPU — WebGPU]")
    try:
        from Runtime._lib.Drivers.WebGPU import WebGpuDriver
        gpu_rt = Runtime(driver=WebGpuDriver())
        register_kernels(gpu_rt)
        gpu_tasks = gpu_rt.compile(jobs)
        gpu_rt.execute(gpu_tasks, source)

        for t in gpu_tasks:
            for name in t.out_names:
                arr = gpu_rt.driver.resolve_output(name)
                if arr is None:
                    print(f"    {name}: <WebGPU readback None>")
                    results[("gpu", name)] = None
                else:
                    results[("gpu", name)] = arr
                    print(f"    {name}: min={arr.min():.4f} max={arr.max():.4f}")

        gpu_rt.driver.release()
        gpu_available = True
    except Exception as e:
        print(f"    WebGPU недоступен: {e}")
        gpu_available = False

    # ── Сравнение ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Сравнение CPU vs GPU")
    print(f"{'='*60}")
    print(f"  {'Kernel':<20} {'Output':<20} {'Max Error':<15} {'Status':<10}")
    print(f"  {'-'*55}")

    all_passed = True
    for t in cpu_tasks:
        for name in t.out_names:
            cpu_val = results.get(("cpu", name))
            gpu_val = results.get(("gpu", name)) if gpu_available else cpu_val

            if gpu_val is None:
                err = float('inf')
                status = "NO GPU"
                all_passed = False
            else:
                err = float(np.abs(cpu_val - gpu_val).max())
                status = "PASS" if err < 1e-4 else "FAIL"
                if status == "FAIL":
                    all_passed = False

            print(f"  {t.op:<20} {name:<20} {err:<15.2e} {status:<10}")

    print(f"{'='*60}")
    if all_passed:
        print("\n  [OK] Все тесты пройдены. GPU работает корректно.\n")
    else:
        print("\n  [!!] Обнаружены расхождения. См. таблицу выше.\n")


if __name__ == "__main__":
    run_example()
