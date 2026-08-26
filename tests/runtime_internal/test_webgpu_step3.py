"""WebGPU Step 3: workspace buffer.

Pipeline: src[i] -> ws[i] (copy), then ws[i] * 2 -> dst[i]
Uses workspace as temporary storage.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "core")))

import numpy as np
from Runtime._lib.mod_iface import InputSlot, OutputSlot, BufferSpec, ExecutionPlan
from Runtime._lib.packet import ExecutionPacket, BufferView
from Runtime._lib.Drivers.WebGPU import WebGpuDriver


# WGSL with workspace:
# Binding 0: src (read)
# Binding 1: ws (read_write, workspace)
# Binding 2: dst (read_write)
WS_WGSL = """
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> ws: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&src)) {
        // Step 1: copy to workspace
        ws[i] = src[i];
        // Step 2: workspace * 2 to dst
        dst[i] = ws[i] * 2.0;
    }
}
"""


def test_workspace_via_driver():
    """Direct driver test with workspace."""
    driver = WebGpuDriver()
    driver.kernel_table["WS"] = {
        "drivers": {"wgsl": WS_WGSL},
        "abi_version": 1,
    }
    driver.compile("WS")
    
    n = 64
    src_data = np.array([float(i) * 0.5 for i in range(n)], dtype=np.float64)
    dst_data = np.zeros(n, dtype=np.float64)
    ws_data = np.zeros(n, dtype=np.float64)  # workspace
    
    from Runtime._lib.mod_iface import BlockView
    
    packet = ExecutionPacket(
        kernel="WS",
        input_buffers=[BufferView(view=BlockView(src_data), dtype="float", size=n)],
        output_buffers=[BufferView(view=BlockView(dst_data), dtype="float", size=n)],
        workspace_buffers=[BufferView(view=BlockView(ws_data), dtype="float", size=n)],
        uniforms={},
        bindings=[
            BufferView(view=BlockView(src_data), dtype="float", size=n),  # 0: input
            BufferView(view=BlockView(ws_data), dtype="float", size=n),   # 1: workspace
            BufferView(view=BlockView(dst_data), dtype="float", size=n),  # 2: output
        ],
        dispatch=((n + 63) // 64, 1, 1),
        abi_version=1,
    )
    
    driver.execute(packet)
    
    for i in range(n):
        expected = src_data[i] * 2.0
        actual = dst_data[i]
        if abs(expected - actual) > 1e-5:
            print(f"  MISMATCH at [{i}]: expected={expected:.4f}, got={actual:.4f}")
            raise AssertionError(f"Workspace step failed at index {i}")
    
    print(f"  [ok] Workspace: {n} elements, src*2 == dst, all match")
    driver.release()


def test_workspace_via_runtime():
    """Full Runtime chain with workspace."""
    from Runtime import Runtime
    
    runtime = Runtime(driver=WebGpuDriver())
    runtime.register_kernel(
        "WS",
        describe=lambda p: ExecutionPlan(
            inputs=[InputSlot(name="x", dtype="float")],
            outputs=[OutputSlot(dtype="float", template="y")],
            workspace=[BufferSpec(dtype="float", elements=0)],
            uniforms={},
        ),
        cpu=lambda ctx: None,
        wgsl=WS_WGSL,
        abi_version=1,
    )
    
    n = 128
    data = np.array([float(i) * 0.3 for i in range(n)], dtype=np.float64)
    
    jobs = [{"op": "WS", "inputs": ["x"], "params": {}}]
    tasks = runtime.compile(jobs)
    runtime.execute(tasks, {"x": data})
    
    result = runtime.driver._output_store.get("y")
    assert result is not None
    
    for i in range(n):
        expected = data[i] * 2.0
        if abs(result[i] - expected) > 1e-5:
            print(f"  MISMATCH at [{i}]: expected={expected:.4f}, got={result[i]:.4f}")
            raise AssertionError(f"Runtime WS failed at index {i}")
    
    print(f"  [ok] Runtime workspace: {n} elements, all match")
    runtime.driver.release()


if __name__ == "__main__":
    print("WebGPU Step 3: workspace support")
    print("=" * 50)
    test_workspace_via_driver()
    test_workspace_via_runtime()
    print("\n=== Step 3 PASSED ===")
