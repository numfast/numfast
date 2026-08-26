"""Builder -- pure function: ExecutionGraph -> list[ExecutionPacket].

Builder walks the graph in topological order, resolves buffers,
allocates memory, and produces ExecutionPackets for the Driver.

No optimization. No fusion. Just mechanical graph -> packets.
"""

import time as _time
from .exec_graph import ExecutionGraph, Node
from .packet import ExecutionPacket, BufferView as BV
from .mod_iface import BlockView


def _alloc_array(driver, size: int, dtype="float"):
    """Allocate array through Driver memory manager or fallback."""
    mm = getattr(driver, 'memory', None)
    import numpy as np
    if mm is not None and hasattr(mm, 'alloc_array'):
        if dtype == "uint32":
            return mm.alloc_array(size, np.uint32)
        elif dtype == "int64":
            return mm.alloc_array(size, np.int64)
        return mm.alloc_array(size, np.float64)
    if dtype == "uint32":
        return np.zeros(size, dtype=np.uint32)
    return np.zeros(size, dtype=np.float64 if dtype == "float" else np.int64)


def _describe_kernel(kernel_id: str, params: dict, kernel_table: dict):
    """Call describe() for a kernel, return ExecutionPlan or None."""
    entry = kernel_table.get(kernel_id)
    if entry and entry.get("describe"):
        return entry["describe"](params)
    return None


def builder(graph: ExecutionGraph, source_data: dict,
            driver, kernel_table: dict) -> list[ExecutionPacket]:
    """Convert ExecutionGraph to a list of ExecutionPackets.

    The graph MUST be in topological order (parents before children).

    Args:
        graph: ExecutionGraph DAG
        source_data: dict of column_name -> numpy array (raw inputs)
        driver: Driver instance (for memory allocation + output storage)
        kernel_table: kernel registry

    Returns:
        list[ExecutionPacket] -- ready for Driver.execute() or execute_wave()
    """
    packets: list[ExecutionPacket] = []

    resolved_outputs: dict[tuple[int, int], object] = {}

    for node in graph.nodes:
        t0 = _time.perf_counter_ns()

        # ---- Resolve input buffers ----
        input_views: list[BV] = []
        for pr in node.inputs:
            if pr.source == "@input":
                col_name = str(pr.port)
                arr = source_data.get(col_name)
                if arr is None:
                    raise KeyError(f"Source data missing column '{col_name}'")
                input_views.append(BV(view=BlockView(arr), dtype=pr.dtype, size=len(arr)))
            elif isinstance(pr.source, int):
                key = (pr.source, pr.port if isinstance(pr.port, int) else 0)
                arr = resolved_outputs.get(key)
                if arr is None:
                    raise KeyError(f"Node {node.id}: input from node {key[0]}:{key[1]} not resolved")
                input_views.append(BV(view=BlockView(arr), dtype=pr.dtype, size=len(arr)))

        # S204: 0-input creation plans (outputs+uniforms required) — sign-off Coordinator 2026-08-26
        if not input_views and not (node.outputs and node.params):
            raise ValueError(f"Node {node.id} ({node.kernel_id}): no inputs")

        input_sizes = [v.size for v in input_views]
        dispatch_size = input_sizes[0] if input_sizes else 1

        # ---- Query kernel descriptor for plan ----
        plan = _describe_kernel(node.kernel_id, node.params, kernel_table)

        # Output sizes
        num_outputs = len(node.outputs)
        if plan and plan.output_size_fn:
            output_sizes = plan.output_size_fn(input_sizes)
        else:
            output_sizes = [dispatch_size] * num_outputs

        if len(output_sizes) != num_outputs:
            output_sizes = [dispatch_size] * num_outputs

        # ---- Allocate output buffers ----
        output_views: list[BV] = []
        output_raws: list[object] = []
        for i in range(num_outputs):
            out_size = output_sizes[i]
            out_dtype = plan.outputs[i].dtype if plan and i < len(plan.outputs) else "float"
            raw = _alloc_array(driver, out_size, out_dtype)
            output_raws.append(raw)
            output_views.append(BV(view=BlockView(raw), dtype=out_dtype, size=out_size))

        # ---- Allocate workspace buffers ----
        workspace_views: list[BV] = []
        if plan:
            for ws_spec in plan.workspace:
                ws_size = ws_spec.elements if ws_spec.elements > 0 else dispatch_size
                raw = _alloc_array(driver, ws_size)
                workspace_views.append(BV(view=BlockView(raw), dtype=ws_spec.dtype, size=ws_size))

        # ---- Uniforms ----
        # Start with plan.uniforms order (matches WGSL struct layout),
        # then override with node.params values to preserve field order.
        # FIX: plan.uniforms are typed (f32 vs u32). Node.params may be
        # Python ints (e.g. batched: N=32) that would override float uniforms
        # with ints -> pack_uniforms would pack as <I (0x00000020) instead of
        # <f (0x42000000), WGSL f32 reads 1e-44 FTZ->0, kernel sees N=0.
        # Preserve plan's type: if plan had float, coerce override to float.
        if plan:
            uniforms = dict(plan.uniforms)
            for k, v in node.params.items():
                if k in uniforms and isinstance(uniforms[k], float):
                    # plan expects f32: keep float (np.int32 also -> float via float())
                    try:
                        uniforms[k] = float(v)
                    except Exception:
                        uniforms[k] = v
                else:
                    uniforms[k] = v
        else:
            uniforms = dict(node.params)
        for k, v in uniforms.items():
            if not isinstance(v, (int, float, bool,)) and not hasattr(v, "__float__"):
                try:
                    uniforms[k] = float(v)
                except Exception:
                    pass
            elif hasattr(v, "item") and not isinstance(v, (int, float, bool)):
                # numpy scalar like np.int32: coerce via float if plan was float
                try:
                    if k in uniforms and isinstance(uniforms.get(k), float):
                        uniforms[k] = float(v)
                except Exception:
                    pass

        # ---- Bindings ----
        bindings = list(input_views) + list(workspace_views) + list(output_views)

        # ---- Dispatch ----
        if node.dispatch is not None:
            dispatch = node.dispatch
        elif plan and plan.dispatch is not None:
            dispatch = plan.dispatch
        else:
            wg_size = 64
            dx = (dispatch_size + wg_size - 1) // wg_size
            dispatch = (dx, 1, 1)

        # ---- Profile metadata ----
        input_bytes = sum(v.size * 8 for v in input_views)
        output_bytes = sum(v.size * 8 for v in output_views)
        workspace_bytes = sum(v.size * 8 for v in workspace_views)
        build_ns = _time.perf_counter_ns() - t0

        profile = {
            "kernel": node.kernel_id,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "workspace_bytes": workspace_bytes,
            "total_bytes": input_bytes + output_bytes + workspace_bytes,
            "dispatch_x": dispatch[0],
            "dispatch_y": dispatch[1],
            "dispatch_z": dispatch[2],
            "build_ns": build_ns,
            "exec_start_ns": 0,
            "exec_end_ns": 0,
            "exec_time_ns": 0,
            "readback_ns": 0,
        }

        # ---- Build ExecutionPacket ----
        packet = ExecutionPacket(
            kernel=node.kernel_id,
            input_buffers=input_views,
            output_buffers=output_views,
            workspace_buffers=workspace_views,
            uniforms=uniforms,
            bindings=bindings,
            dispatch=dispatch,
            abi_version=kernel_table.get(node.kernel_id, {}).get("abi_version", 1),
            profile=profile,
        )
        packets.append(packet)

        # ---- Store outputs ----
        for i, raw in enumerate(output_raws):
            key = (node.id, i)
            resolved_outputs[key] = raw
            out_name = node.outputs[i].name if i < len(node.outputs) else f"out_{i}"
            driver.store_output(out_name, node.id, i, raw)

    return packets
