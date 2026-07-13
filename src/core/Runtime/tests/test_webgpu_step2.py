"""WebGPU Step 2: dst[i] = src[i] * multiplier (uniform test)."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from Runtime._lib.mod_iface import InputSlot, OutputSlot, BufferSpec, ExecutionPlan
from Runtime._lib.packet import ExecutionPacket, BufferView
from Runtime._lib.Drivers.WebGPU import WebGpuDriver


# WGSL with uniform multiplier
# Binding 0: src, Binding 1: dst, Binding 2: uniform params
MUL_WGSL = """
struct Params {
    multiplier: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&src)) {
        dst[i] = src[i] * params.multiplier;
    }
}
"""


def test_uniform_via_driver():
    """Direct WebGpuDriver test with uniform multiplier."""
    driver = WebGpuDriver()
    driver.kernel_table["MUL"] = {
        "drivers": {"wgsl": MUL_WGSL},
        "abi_version": 1,
    }
    driver.compile("MUL")

    n = 64
    src_data = np.array([float(i) for i in range(n)], dtype=np.float64)
    dst_data = np.zeros(n, dtype=np.float64)
    multiplier = 3.5

    from Runtime._lib.mod_iface import BlockView

    packet = ExecutionPacket(
        kernel="MUL",
        input_buffers=[BufferView(view=BlockView(src_data), dtype="float", size=n)],
        output_buffers=[BufferView(view=BlockView(dst_data), dtype="float", size=n)],
        uniforms={"multiplier": multiplier},
        bindings=[
            BufferView(view=BlockView(src_data), dtype="float", size=n),
            BufferView(view=BlockView(dst_data), dtype="float", size=n),
        ],
        dispatch=((n + 63) // 64, 1, 1),
        abi_version=1,
    )

    driver.execute(packet)

    for i in range(n):
        expected = src_data[i] * multiplier
        actual = dst_data[i]
        if abs(expected - actual) > 1e-5:
            print(f"  MISMATCH at [{i}]: expected={expected:.4f}, got={actual:.4f}")
            raise AssertionError(f"Uniform mul failed at index {i}")

    print(f"  ok MUL via driver * {multiplier}: {n} elements, all match")
    driver.release()


def test_uniform_via_runtime():
    """Full Runtime chain with uniform multiplier."""
    from Runtime import Runtime

    runtime = Runtime(driver=WebGpuDriver())
    runtime.register_kernel(
        "MUL",
        describe=lambda p: ExecutionPlan(
            inputs=[InputSlot(name="x", dtype="float")],
            outputs=[OutputSlot(dtype="float", template="y")],
            workspace=[],
            uniforms={"multiplier": p.get("multiplier", 2.0)},
        ),
        cpu=lambda ctx: None,
        wgsl=MUL_WGSL,
        abi_version=1,
    )

    n = 64
    data = np.array([float(i) * 0.1 for i in range(n)], dtype=np.float64)
    multiplier = 2.5

    jobs = [{"op": "MUL", "inputs": ["x"], "params": {"multiplier": multiplier}}]
    tasks = runtime.compile(jobs)
    runtime.execute(tasks, {"x": data})

    result = runtime.driver._output_store.get("y")
    assert result is not None

    for i in range(n):
        expected = data[i] * multiplier
        if abs(result[i] - expected) > 1e-5:
            print(f"  MISMATCH at [{i}]: expected={expected:.4f}, got={result[i]:.4f}")
            raise AssertionError(f"Runtime MUL failed at index {i}")

    print(f"  ok MUL via runtime * {multiplier}: {n} elements, all match")
    runtime.driver.release()


if __name__ == "__main__":
    print("WebGPU Step 2: dst[i] = src[i] * multiplier")
    print("=" * 50)
    test_uniform_via_driver()
    test_uniform_via_runtime()
    print("\n=== Step 2 PASSED ===")
