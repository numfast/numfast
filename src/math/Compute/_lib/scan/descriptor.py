"""Scan descriptors — ExecutionPlans for three chained Scan kernels.

Параметр op (оп-параметризация по образцу Reduce, F-046):
  - строка: "sum" / "mul" / "max" / "min" (default "sum");
  - числовой код: 0 = sum, 1 = mul, 2 = max, 3 = min.

Uniform "op" всегда числовой (f32-код): WebGPU pack_uniforms пакует все
uniforms как float32. Строковый op принимается для имён выходов (template)
и CPU-диспетчеризации — числовой код униформится в descriptor'е.

Обратная совместимость: вызов без op == sum, шаблоны промежуточных выходов
не меняются ("scan_local", "block_sum", "block_prefix"), итоговый выход для
sum — прежний "scan"; для остальных ops — "scan_<имя>" ("scan_max" и т.д.).
"""

from Runtime._lib.mod_iface import ExecutionPlan, InputSlot, OutputSlot

BLOCK = 64

_OP_NAMES = {0: "sum", 1: "mul", 2: "max", 3: "min"}
_OP_CODES = {"sum": 0.0, "mul": 1.0, "max": 2.0, "min": 3.0}


def _resolve_op(params: dict):
    """(name, code) из params: строка или числовой код (default sum)."""
    op = params.get("op", "sum")
    if isinstance(op, (int, float)):
        name = _OP_NAMES.get(int(op), "sum")
        code = float(op)
    else:
        name = op if op in _OP_CODES else "sum"
        code = _OP_CODES.get(name, 0.0)
    return name, code


def describe_local(params: dict) -> ExecutionPlan:
    """Local scan per block of 64.

    Inputs: data (float)
    Outputs: scan_local (float) — local scan, same size as input
             block_sum (float) — total per block, size = ceil(N/64)
    """
    _, code = _resolve_op(params)

    def _sizes(input_sizes):
        n = input_sizes[0]
        num_blocks = max(1, (n + BLOCK - 1) // BLOCK)
        return [n, num_blocks]

    return ExecutionPlan(
        inputs=[InputSlot(name="data", dtype="float")],
        outputs=[
            OutputSlot(dtype="float", template="scan_local"),
            OutputSlot(dtype="float", template="block_sum"),
        ],
        workspace=[],
        uniforms={"op": code},
        output_size_fn=_sizes,
    )


def describe_totals(params: dict) -> ExecutionPlan:
    """Exclusive scan of block totals (sp[0] = neutral).

    Inputs: block_sums (float)
    Outputs: block_prefix (float) — same size as input
    """
    _, code = _resolve_op(params)

    def _sizes(input_sizes):
        return [input_sizes[0]]  # same as input

    return ExecutionPlan(
        inputs=[InputSlot(name="block_sums", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="block_prefix")],
        workspace=[],
        uniforms={"op": code},
        output_size_fn=_sizes,
    )


def describe_final(params: dict) -> ExecutionPlan:
    """Combine accumulated block offset with local scans.

    Inputs: data (float) — original input
            local (float) — local scans
            prefix (float) — exclusive block prefix (sp[block_id])
    Outputs: scan (float) — final inclusive scan, same as data
    """
    name, code = _resolve_op(params)
    template = "scan" if name == "sum" else f"scan_{name}"

    def _sizes(input_sizes):
        return [input_sizes[0]]  # same as "data" input

    return ExecutionPlan(
        inputs=[
            InputSlot(name="data", dtype="float"),
            InputSlot(name="local", dtype="float"),
            InputSlot(name="prefix", dtype="float"),
        ],
        outputs=[OutputSlot(dtype="float", template=template)],
        workspace=[],
        uniforms={"op": code},
        output_size_fn=_sizes,
    )
