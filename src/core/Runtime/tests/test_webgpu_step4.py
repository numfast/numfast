"""WebGPU Step 4: multi-output.

Shader computes:
  out0[i] = src[i] + 1.0
  out1[i] = src[i] * 2.0
  out2[i] = src[i] * src[i]
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan
from Runtime._lib.packet import ExecutionPacket, BufferView
from Runtime._lib.Drivers.WebGPU import WebGpuDriver


# WGSL: 3 outputs
MO_WGSL = """
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> out0: array<f32>;
@group(0) @binding(2) var<storage, read_write> out1: array<f32>;
@group(0) @binding(3) var<storage, read_write> out2: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&src)) {
        out0[i] = src[i] + 1.0;
        out1[i] = src[i] * 2.0;
        out2[i] = src[i] * src[i];
    }
}
"""


def test_multioutput_via_driver():
    """Direct driver with 3 outputs."""
    driver = WebGpuDriver()
    driver.kernel_table["MO"] = {
        "drivers": {"wgsl": MO_WGSL},
        "abi_version": 1,
    }
    driver.compile("MO")
    
    n = 64
    src = np.array([float(i) * 0.1 for i in range(n)], dtype=np.float64)
    out0 = np.zeros(n, dtype=np.float64)
    out1 = np.zeros(n, dtype=np.float64)
    out2 = np.zeros(n, dtype=np.float64)
    
    from Runtime._lib.mod_iface import BlockView
    
    packet = ExecutionPacket(
        kernel="MO",
        input_buffers=[BufferView(view=BlockView(src), dtype="float", size=n)],
        output_buffers=[
            BufferView(view=BlockView(out0), dtype="float", size=n),
            BufferView(view=BlockView(out1), dtype="float", size=n),
            BufferView(view=BlockView(out2), dtype="float", size=n),
        ],
        uniforms={},
        bindings=[
            BufferView(view=BlockView(src), dtype="float", size=n),   # 0: input
            BufferView(view=BlockView(out0), dtype="float", size=n),  # 1: out0
            BufferView(view=BlockView(out1), dtype="float", size=n),  # 2: out1
            BufferView(view=BlockView(out2), dtype="float", size=n),  # 3: out2
        ],
        dispatch=((n + 63) // 64, 1, 1),
        abi_version=1,
    )
    
    driver.execute(packet)
    
    for i in range(n):
        if abs(out0[i] - (src[i] + 1.0)) > 1e-5:
            raise AssertionError(f"out0[{i}] fail")
        if abs(out1[i] - (src[i] * 2.0)) > 1e-5:
            raise AssertionError(f"out1[{i}] fail")
        if abs(out2[i] - (src[i] * src[i])) > 1e-5:
            raise AssertionError(f"out2[{i}] fail")
    
    print(f"  [ok] Multi-output (3): {n} elements, all outputs match")
    driver.release()


def test_multioutput_via_runtime():
    """Full Runtime with multi-output kernel."""
    from Runtime import Runtime
    
    runtime = Runtime(driver=WebGpuDriver())
    runtime.register_kernel(
        "MO",
        describe=lambda p: ExecutionPlan(
            inputs=[InputSlot(name="x", dtype="float")],
            outputs=[
                OutputSlot(dtype="float", template="plus1"),
                OutputSlot(dtype="float", template="times2"),
                OutputSlot(dtype="float", template="squared"),
            ],
            workspace=[],
            uniforms={},
        ),
        cpu=lambda ctx: None,
        wgsl=MO_WGSL,
        abi_version=1,
    )
    
    n = 128
    data = np.array([float(i) * 0.5 for i in range(n)], dtype=np.float64)
    
    jobs = [{"op": "MO", "inputs": ["x"], "params": {}}]
    tasks = runtime.compile(jobs)
    runtime.execute(tasks, {"x": data})
    
    for t in tasks:
        for out_name in t.out_names:
            result = runtime.driver.resolve_output(out_name)
            assert result is not None, f"No output '{out_name}'"
    
    plus1 = runtime.driver.resolve_output("plus1")
    times2 = runtime.driver.resolve_output("times2")
    squared = runtime.driver.resolve_output("squared")
    
    for i in range(n):
        if abs(plus1[i] - (data[i] + 1.0)) > 1e-5:
            raise AssertionError(f"Runtime plus1[{i}] fail")
        if abs(times2[i] - (data[i] * 2.0)) > 1e-5:
            raise AssertionError(f"Runtime times2[{i}] fail")
        if abs(squared[i] - (data[i] * data[i])) > 1e-5:
            raise AssertionError(f"Runtime squared[{i}] fail")
    
    print(f"  [ok] Runtime multi-output: {n} elements, 3 outputs match")
    runtime.driver.release()


if __name__ == "__main__":
    print("WebGPU Step 4: multi-output")
    print("=" * 50)
    test_multioutput_via_driver()
    test_multioutput_via_runtime()
    print("\n=== Step 4 PASSED ===")
