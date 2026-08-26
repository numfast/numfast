"""S204 - zero-input creation plans in Runtime builder (minimal fix proof).

Sign-off: Coordinator 2026-08-26.
Change under test: Runtime/_lib/builder.py validation only:
  a node with len(inputs)==0 is VALID iff it has outputs AND non-empty params
  (uniforms). Otherwise the previous ValueError is raised unchanged.
No Executor / Kernel / Driver ABI changes. No dummy inputs introduced.

ASCII-only file (tests are read in locale encoding by test_backend.py).
"""

import numpy as np
import pytest

from Runtime._lib.builder import builder
from Runtime._lib.exec_graph import ExecutionGraph, Node, PortRef, OutputSpec
from Runtime._lib.mod_iface import (
    ExecutionPlan,
    InputSlot,
    OutputSlot,
)


class StubDriver:
    """Minimal driver: builder needs only store_output (+optional memory)."""

    def __init__(self):
        self.stored = {}

    def store_output(self, name, node_id, port, raw):
        self.stored[(node_id, port)] = (name, raw)


def _creation_kernel_table(n_out=8):
    """kernel_table for a 0-input creation kernel: out[i] = i, size N."""

    def describe(params):
        n = int(params["N"])
        return ExecutionPlan(
            inputs=[],
            outputs=[OutputSlot(dtype="float", template="out_{i}")],
            workspace=[],
            uniforms={"N": float(n)},
            output_size_fn=lambda input_sizes: [n],
            dispatch=(1, 1, 1),
        )

    return {"Range": {"describe": describe, "abi_version": 1}}


def _identity_kernel_table():
    def describe(params):
        return ExecutionPlan(
            inputs=[InputSlot(name="data", dtype="float")],
            outputs=[OutputSlot(dtype="float", template="copy_{i}")],
            uniforms={"period": int(params["period"])},
            dispatch=(1, 1, 1),
        )

    return {"Copy": {"describe": describe, "abi_version": 1}}


def _graph(nodes, inputs=(), outputs=()):
    return ExecutionGraph(
        nodes=list(nodes), inputs=list(inputs), outputs=list(outputs)
    )


def test_zero_input_plan_valid():
    node = Node(
        id=0,
        kernel_id="Range",
        inputs=[],
        outputs=[OutputSpec(name="out0", dtype="f32")],
        params={"N": 8},
        dispatch=None,
    )
    driver = StubDriver()
    packets = builder(
        _graph([node], outputs=[("out0", 0, 0)]),
        source_data={},
        driver=driver,
        kernel_table=_creation_kernel_table(),
    )
    assert len(packets) == 1
    pkt = packets[0]
    assert pkt.kernel == "Range"
    assert pkt.input_buffers == []
    assert len(pkt.output_buffers) == 1
    assert pkt.output_buffers[0].size == 8
    assert pkt.uniforms["N"] == 8.0
    assert pkt.dispatch == (1, 1, 1)
    assert pkt.bindings and pkt.bindings[-1].size == 8
    assert (0, 0) in driver.stored


def test_zero_input_no_outputs_rejected():
    node = Node(
        id=0, kernel_id="Range", inputs=[], outputs=[], params={"N": 4}
    )
    with pytest.raises(ValueError, match="no inputs"):
        builder(
            _graph([node]), source_data={}, driver=StubDriver(),
            kernel_table=_creation_kernel_table(),
        )


def test_zero_input_no_uniforms_rejected():
    node = Node(
        id=0,
        kernel_id="Range",
        inputs=[],
        outputs=[OutputSpec(name="out0")],
        params={},
    )
    with pytest.raises(ValueError, match="no inputs"):
        builder(
            _graph([node]), source_data={}, driver=StubDriver(),
            kernel_table=_creation_kernel_table(),
        )


def test_existing_inputs_path_unchanged():
    data = np.arange(16, dtype=np.float64)
    node = Node(
        id=7,
        kernel_id="Copy",
        inputs=[PortRef(source="@input", port="data", dtype="f32")],
        outputs=[OutputSpec(name="copy0", dtype="f32")],
        params={"period": 3},
    )
    driver = StubDriver()
    packets = builder(
        _graph([node]),
        source_data={"data": data},
        driver=driver,
        kernel_table=_identity_kernel_table(),
    )
    assert len(packets) == 1
    pkt = packets[0]
    assert len(pkt.input_buffers) == 1
    assert pkt.input_buffers[0].size == 16
    # default path: output sized from dispatch_size (= first input size)
    assert pkt.output_buffers[0].size == 16
    assert pkt.uniforms["period"] == 3
    assert driver.stored[(7, 0)][0] == "copy0"
