"""WebGPU Step 5: EMA — первое реальное ядро на WebGPU.

Сравнение CPU vs GPU через KernelTestHarness.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from Runtime._lib.mod_iface import InputSlot, OutputSlot, BufferSpec, ExecutionPlan
from Runtime._lib.packet import ExecutionPacket, BufferView
from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Runtime._lib.harness import KernelTestHarness
from Runtime import Runtime
from Trading import register_all as register_trading_kernels


EMA_WGSL_SERIAL = """
struct Params {
    period: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = arrayLength(&src);
    let multiplier = 2.0 / (params.period + 1.0);

    dst[0] = src[0];

    var i = 1u;
    while (i < n) {
        let prev = dst[i - 1u];
        dst[i] = (src[i] - prev) * multiplier + prev;
        i = i + 1u;
    }
}
"""


def test_ema_via_driver():
    """Direct WebGpuDriver test with EMA period=3."""
    driver = WebGpuDriver()
    driver.kernel_table["EMA"] = {
        "drivers": {"wgsl": EMA_WGSL_SERIAL},
        "abi_version": 1,
    }
    driver.compile("EMA")

    n = 10
    src_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
    dst_data = np.zeros(n, dtype=np.float64)
    period = 3
    multiplier = 2.0 / (period + 1.0)

    from Runtime._lib.mod_iface import BlockView

    packet = ExecutionPacket(
        kernel="EMA",
        input_buffers=[BufferView(view=BlockView(src_data), dtype="float", size=n)],
        output_buffers=[BufferView(view=BlockView(dst_data), dtype="float", size=n)],
        uniforms={"period": float(period)},
        bindings=[
            BufferView(view=BlockView(src_data), dtype="float", size=n),
            BufferView(view=BlockView(dst_data), dtype="float", size=n),
        ],
        dispatch=(1, 1, 1),
        abi_version=1,
    )

    driver.execute(packet)

    expected = np.zeros(n)
    expected[0] = src_data[0]
    for i in range(1, n):
        expected[i] = (src_data[i] - expected[i-1]) * multiplier + expected[i-1]

    for i in range(n):
        if abs(dst_data[i] - expected[i]) > 1e-5:
            print(f"  MISMATCH at [{i}]: expected={expected[i]:.4f}, got={dst_data[i]:.4f}")
            raise AssertionError(f"EMA failed at index {i}")

    print(f"  [ok] EMA(period={period}) via driver: {n} elements, all match")
    print(f"    GPU last={dst_data[-1]:.4f}, CPU expected={expected[-1]:.4f}")
    driver.release()


def test_ema_via_runtime():
    """Full Runtime chain with EMA on WebGPU, compare with CPU golden."""
    runtime = Runtime(driver=WebGpuDriver())

    from Trading.EMA import describe as ema_desc
    from Trading.EMA import cpu as ema_cpu

    runtime.register_kernel(
        "EMA",
        describe=ema_desc,
        cpu=ema_cpu,
        wgsl=EMA_WGSL_SERIAL,
        abi_version=1,
    )

    n = 10
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)

    # CPU reference
    cpu_runtime = Runtime()
    register_trading_kernels(cpu_runtime)
    ref = KernelTestHarness(cpu_runtime).run("EMA", {"period": 3}, {"Close": data})
    expected_last = ref["outputs"]["ema_3"][-1]

    # GPU
    gpu_runtime = Runtime(driver=WebGpuDriver())
    gpu_runtime.register_kernel("EMA", describe=ema_desc, cpu=ema_cpu, wgsl=EMA_WGSL_SERIAL, abi_version=1)

    gpu_tasks = gpu_runtime.compile([{"op": "EMA", "inputs": ["Close"], "params": {"period": 3}}])
    gpu_runtime.execute(gpu_tasks, {"Close": data})

    gpu_result = gpu_runtime.driver.resolve_output("ema_3")

    assert gpu_result is not None, "No GPU output"

    for i in range(n):
        cpu_val = ref["outputs"]["ema_3"][i]
        gpu_val = gpu_result[i]
        if abs(cpu_val - gpu_val) > 1e-5:
            print(f"  MISMATCH at [{i}]: CPU={cpu_val:.4f}, GPU={gpu_val:.4f}")
            raise AssertionError(f"CPU vs GPU EMA mismatch")

    print(f"  [ok] EMA via Runtime: CPU=GPU, {n} elements")
    print(f"    CPU last={ref['outputs']['ema_3'][-1]:.4f}, GPU last={gpu_result[-1]:.4f}")
    gpu_runtime.driver.release()


if __name__ == "__main__":
    print("WebGPU Step 5: EMA (first real kernel)")
    print("=" * 50)
    test_ema_via_driver()
    test_ema_via_runtime()
    print("\n=== Step 5 PASSED ===")
