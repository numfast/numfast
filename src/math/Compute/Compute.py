"""Compute Extension -- GPU compute kernels.

Builder entry point.

Architecture of primitives (ISA Levels v1.0):
---------------------------------------------

L0. Memory             -- memory access, copying
  Gather                 indexed read: out[i] = src[idx[i]]
  Scatter                indexed write: out[idx[i]] = value[i]
  Shift                  offset access: out[i] = src[i + offset]
  Fill (via MapBinary)   out[i] = scalar

L1. Logic              -- conditional operations
  Compare (op)           comparison: out[i] = a[i] op b[i] -> 0.0/1.0
                         ops: gt, ge, lt, le, eq, ne
  Mask                   conditional select: out[i] = cond ? a : b
  Clamp                  bound: out[i] = clamp(x, lo, hi)

L2. Element-wise       -- element-wise functions
  Map (Unary)            out[i] = f(x[i]), f in {sin,cos,exp,sqrt,log,abs,neg,square}
  MapBinary              out[i] = a op b, op in {add,sub,mul,div,max,min}
                         With scalar mode (use_scalar_a/b)

L3. Reduction          -- array reduction
  Reduce                 single value: sum, max, min
  Scan (Local/Totals/Final) prefix sum (3-pass Hillis-Steele)
  ArgMinMax              reduce with index: out_val[i], out_idx[i] = min/max + position

L4. Window             -- rolling window
  RollingSum, RollingMin, RollingMax, RollingStdDev

L5. Stateful           -- recurrent computations
  StateKernel (Single/Dual/SuperTrend) out[i] = a*in[i] + b*state[i-1]
  Recurrent             generic stateful computation
  TrueRange             TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
  EMA = StateKernel_Single(mode=0, a=2/(period+1), b=1-2/(period+1))

L6. Structural         -- permutation, distribution
  Sort, Histogram, Combine

L7. Numerical          -- advanced numerical methods
  FFT (BitReverse + FftStage), MatMul

Backend: AST Compiler (Expression deprecated)
  Expression(formula, A, B, ...) -- kept for backward compatibility.
  New code uses AST directly (Runtime._lib.ast): parse_expr -> optimize -> generate.
  AST supports both WgslGenerator (GPU) and CpuGenerator (CPU).

New WGSL Rule (Principle N1 NumFast):
  No new WGSL shader for a new indicator.
  WGSL written ONLY for a new compute primitive.
  
  To add a new WGSL, the developer MUST:
  1. Show the DAG from existing primitives
  2. Prove the DAG is worse in at least one of:
     - Speed (measured, not assumed)
     - Memory (peak or total)
     - Readability (expression complexity)
     - Expressiveness (cannot be composed)
  
  If no DAG is shown or no criterion is proven worse -> WGSL REJECTED.
  Indicators are always DAG from primitives, never standalone WGSL.

Level dependencies:
  L0 -> L1 -> L2 -> L3 -> L4 -> L5 -> L6 -> L7
  AST Backend can generate code for any level.
  28 primitives total (v1.0).
"""


_CAPABILITIES = {
    "Map":      {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "Reduce":   {"streaming": False, "workspace": True,  "multi_input": False, "multi_output": False},
    "ScanLocal": {"streaming": False, "workspace": False, "multi_input": False, "multi_output": True},
    "ScanTotals": {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "ScanFinal": {"streaming": False, "workspace": False, "multi_input": True,  "multi_output": False},
    "Histogram": {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "Sort":     {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "MatMul":   {"streaming": False, "workspace": False, "multi_input": True,  "multi_output": False},
    "Compare":   {"streaming": False, "workspace": False, "multi_input": True,  "multi_output": False},
    "Scatter":   {"streaming": False, "workspace": False, "multi_input": True,  "multi_output": False},
    "ArgMinMax": {"streaming": False, "workspace": True,  "multi_input": False,  "multi_output": True},
    "Gather":    {"streaming": False, "workspace": False, "multi_input": True,  "multi_output": False},
    "Mask":      {"streaming": False, "workspace": False, "multi_input": True,  "multi_output": False},
    "BitReverse": {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "FftStage":  {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "RollingSum":    {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "RollingMin":    {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "RollingMax":    {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "RollingStdDev": {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "MapBinary": {"streaming": False, "workspace": False, "multi_input": True,  "multi_output": False},
    "Shift":     {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "Combine":   {"streaming": False, "workspace": False, "multi_input": True,  "multi_output": False},
    "Clamp":     {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "Expression": {"streaming": False, "workspace": False, "multi_input": True,  "multi_output": False},
    "Recurrent": {"streaming": False, "workspace": True,  "multi_input": False, "multi_output": False},
    "TrueRange": {"streaming": False, "workspace": False, "multi_input": True,  "multi_output": False},
    "StateKernel_Single": {"streaming": False, "workspace": True, "multi_input": False, "multi_output": False},
    "StateKernel_Dual":   {"streaming": False, "workspace": True, "multi_input": True,  "multi_output": True},
    "StateKernel_ST":     {"streaming": False, "workspace": True, "multi_input": True,  "multi_output": False},
    "LogicalAnd": {"streaming": False, "workspace": False, "multi_input": True,  "multi_output": False},
    "LogicalOr":  {"streaming": False, "workspace": False, "multi_input": True,  "multi_output": False},
}


def _kernels():
    from Compute._lib.map import describe as _describe_map, cpu as _cpu_map, wgsl as _wgsl_map
    from Compute._lib.reduce import describe as _describe_reduce, cpu as _cpu_reduce, wgsl as _wgsl_reduce
    from Compute._lib.scan import describe_local, describe_totals, describe_final
    from Compute._lib.scan import cpu_local, cpu_totals, cpu_final
    from Compute._lib.scan import wgsl_local, wgsl_totals, wgsl_final
    from Compute._lib.histogram import describe as _describe_histogram, cpu_histogram as _cpu_histogram, wgsl_histogram as _wgsl_histogram
    from Compute._lib.sort import describe_sort as _describe_sort, cpu_sort as _cpu_sort, wgsl_sort as _wgsl_sort
    from Compute._lib.matmul import describe as _describe_matmul, cpu as _cpu_matmul, wgsl as _wgsl_matmul
    from Compute._lib.fft import describe_stage, describe_bitreverse
    from Compute._lib.fft import cpu_stage, cpu_bitreverse
    from Compute._lib.fft import wgsl_stage, wgsl_bitreverse
    from Compute._lib.gather import describe as _describe_gather, cpu as _cpu_gather, wgsl as _wgsl_gather
    from Compute._lib.mask import describe as _describe_mask, cpu as _cpu_mask, wgsl as _wgsl_mask
    from Compute._lib.rolling_sum import describe as _describe_rolling_sum, cpu as _cpu_rolling_sum, wgsl as _wgsl_rolling_sum
    from Compute._lib.rolling_min import describe as _describe_rolling_min, cpu as _cpu_rolling_min, wgsl as _wgsl_rolling_min
    from Compute._lib.rolling_max import describe as _describe_rolling_max, cpu as _cpu_rolling_max, wgsl as _wgsl_rolling_max
    from Compute._lib.rolling_stddev import describe as _describe_rolling_stddev, cpu as _cpu_rolling_stddev, wgsl as _wgsl_rolling_stddev
    from Compute._lib.map_binary import describe as _describe_map_binary, cpu as _cpu_map_binary, wgsl as _wgsl_map_binary
    from Compute._lib.shift import describe as _describe_shift, cpu as _cpu_shift, wgsl as _wgsl_shift
    from Compute._lib.combine import describe as _describe_combine, cpu as _cpu_combine, wgsl as _wgsl_combine
    from Compute._lib.clamp import describe as _describe_clamp, cpu as _cpu_clamp, wgsl as _wgsl_clamp
    from Compute._lib.recurrent import describe as _describe_recurrent, cpu as _cpu_recurrent, wgsl as _wgsl_recurrent
    from Compute._lib.true_range import describe as _describe_true_range, cpu as _cpu_true_range, wgsl as _wgsl_true_range
    from Compute._lib.state_kernel import describe as _describe_sk, cpu as _cpu_sk, WGSL_SINGLE, WGSL_DUAL, WGSL_SUPERTREND
    from Compute._lib.expression import describe as _describe_expr, cpu as _cpu_expr, wgsl_generator as _wgsl_expr
    from Compute._lib.compare import describe as _describe_compare, cpu as _cpu_compare
    from Compute._lib.compare.wgsl import wgsl_generator as _wgsl_compare
    from Compute._lib.scatter import describe as _describe_scatter, cpu as _cpu_scatter, wgsl as _wgsl_scatter
    from Compute._lib.argminmax import describe as _describe_argminmax, cpu as _cpu_argminmax
    from Compute._lib.argminmax.wgsl import wgsl_generator as _wgsl_argminmax
    from Compute._lib.logical_and import describe as _describe_logical_and, cpu as _cpu_logical_and, WGSL as _wgsl_logical_and
    from Compute._lib.logical_or import describe as _describe_logical_or, cpu as _cpu_logical_or, WGSL as _wgsl_logical_or

    return [
        ("Map",       _describe_map,       _cpu_map,       _wgsl_map),
        ("Reduce",    _describe_reduce,    _cpu_reduce,    _wgsl_reduce),
        ("ScanLocal",  describe_local,      cpu_local,      wgsl_local),
        ("ScanTotals", describe_totals,     cpu_totals,     wgsl_totals),
        ("ScanFinal",  describe_final,      cpu_final,      wgsl_final),
        ("Histogram", _describe_histogram, _cpu_histogram, _wgsl_histogram),
        ("Sort",      _describe_sort,      _cpu_sort,      _wgsl_sort),
        ("MatMul",    _describe_matmul,    _cpu_matmul,    _wgsl_matmul),
        ("BitReverse", describe_bitreverse, cpu_bitreverse, wgsl_bitreverse),
        ("FftStage",   describe_stage,      cpu_stage,      wgsl_stage),
        ("Gather",   _describe_gather,   _cpu_gather,   _wgsl_gather),
        ("Mask",     _describe_mask,     _cpu_mask,     _wgsl_mask),
        ("RollingSum",    _describe_rolling_sum,    _cpu_rolling_sum,    _wgsl_rolling_sum),
        ("RollingMin",    _describe_rolling_min,    _cpu_rolling_min,    _wgsl_rolling_min),
        ("RollingMax",    _describe_rolling_max,    _cpu_rolling_max,    _wgsl_rolling_max),
        ("RollingStdDev", _describe_rolling_stddev, _cpu_rolling_stddev, _wgsl_rolling_stddev),
        ("MapBinary",  _describe_map_binary,  _cpu_map_binary,  _wgsl_map_binary),
        ("Shift",      _describe_shift,       _cpu_shift,       _wgsl_shift),
        ("Combine",    _describe_combine,      _cpu_combine,     _wgsl_combine),
        ("Clamp",      _describe_clamp,        _cpu_clamp,       _wgsl_clamp),
        ("Expression", _describe_expr,         _cpu_expr,        _wgsl_expr),
        ("Recurrent",  _describe_recurrent,    _cpu_recurrent,   _wgsl_recurrent),
        ("TrueRange",  _describe_true_range,   _cpu_true_range,  _wgsl_true_range),
        ("StateKernel_Single", _describe_sk, _cpu_sk, WGSL_SINGLE),
        ("StateKernel_Dual",   _describe_sk, _cpu_sk, WGSL_DUAL),
        ("StateKernel_ST",     _describe_sk, _cpu_sk, WGSL_SUPERTREND),
        ("Compare",   _describe_compare,   _cpu_compare,   _wgsl_compare),
        ("Scatter",   _describe_scatter,   _cpu_scatter,   _wgsl_scatter),
        ("ArgMinMax", _describe_argminmax, _cpu_argminmax, _wgsl_argminmax),
        ("LogicalAnd", _describe_logical_and, _cpu_logical_and, _wgsl_logical_and),
        ("LogicalOr",  _describe_logical_or,  _cpu_logical_or,  _wgsl_logical_or),
    ]


def register_all(runtime):
    for alias, desc, cpu_fn, wgsl_src in _kernels():
        runtime.register_kernel(
            alias, describe=desc, cpu=cpu_fn, wgsl=wgsl_src, abi_version=1,
            capabilities=_CAPABILITIES.get(alias, {}),
        )


def setup(kernel):
    register_fn = kernel.alias.get("register_kernel")
    if register_fn:
        for alias, desc, cpu_fn, wgsl_src in _kernels():
            register_fn(
                alias, describe=desc, cpu=cpu_fn, wgsl=wgsl_src, abi_version=1,
                capabilities=_CAPABILITIES.get(alias, {}),
            )
