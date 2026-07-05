"""WebGPU Step 1: minimal dst[i] = src[i] compute shader.

Проверяет полный цикл:
  Compiler -> Scheduler -> Dispatcher -> WebGpuDriver -> GPU -> readback
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
from numfast.Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan, BlockView
from numfast.Runtime._lib.packet import ExecutionPacket, BufferView
from numfast.Runtime._lib.Drivers.WebGPU import WebGpuDriver


COPY_WGSL = """
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&src)) {
        dst[i] = src[i];
    }
}
"""


def test_copy_via_driver():
    """Minimal test: create ExecutionPacket and execute via WebGpuDriver."""
    driver = WebGpuDriver()
    driver.kernel_table["COPY"] = {
        "drivers": {"wgsl": COPY_WGSL},
        "abi_version": 1,
    }

    n = 64
    src_data = np.array([float(i) * 0.5 for i in range(n)], dtype=np.float64)
    dst_data = np.zeros(n, dtype=np.float64)

    packet = ExecutionPacket(
        kernel="COPY",
        input_buffers=[BufferView(view=BlockView(src_data), dtype="float", size=n)],
        output_buffers=[BufferView(view=BlockView(dst_data), dtype="float", size=n)],
        uniforms={},
        bindings=[
            BufferView(view=BlockView(src_data), dtype="float", size=n),
            BufferView(view=BlockView(dst_data), dtype="float", size=n),
        ],
        dispatch=(1, 1, 1),
        abi_version=1,
    )

    driver.compile("COPY")
    driver.execute(packet)

    for i in range(n):
        expected = src_data[i]
        actual = dst_data[i]
        if abs(expected - actual) > 1e-6:
            print(f"  MISMATCH at [{i}]: expected={expected}, got={actual}")
            raise AssertionError(f"Copy failed at index {i}")

    print(f"  ok COPY via driver: {n} elements, all match")
    driver.release()


def test_copy_via_runtime():
    """Test via Runtime.execute() — full chain with WebGpuDriver."""
    from numfast.Runtime import Runtime

    runtime = Runtime(driver=WebGpuDriver())

    runtime.register_kernel(
        "COPY",
        describe=lambda p: ExecutionPlan(
            inputs=[InputSlot(name="x", dtype="float")],
            outputs=[OutputSlot(dtype="float", template="y")],
            workspace=[],
            uniforms={},
        ),
        cpu=lambda ctx: None,
        wgsl=COPY_WGSL,
        abi_version=1,
    )

    n = 64
    data = np.array([float(i) * 2.0 for i in range(n)], dtype=np.float64)

    jobs = [{"op": "COPY", "inputs": ["x"], "params": {}}]
    tasks = runtime.compile(jobs)
    runtime.execute(tasks, {"x": data})

    result = runtime.driver._output_store.get("y")
    assert result is not None, "No output 'y'"

    for i in range(n):
        if abs(result[i] - data[i]) > 1e-6:
            print(f"  MISMATCH at [{i}]: expected={data[i]}, got={result[i]}")
            raise AssertionError(f"Runtime COPY failed at index {i}")

    print(f"  ok COPY via runtime: {n} elements, all match")
    runtime.driver.release()


if __name__ == "__main__":
    print("WebGPU Step 1: dst[i] = src[i]")
    print("=" * 50)

    try:
        test_copy_via_driver()
        test_copy_via_runtime()
        print("\n=== Step 1 PASSED ===")
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
