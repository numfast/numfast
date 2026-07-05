"""NumFast Compute — вычислительные ядра.

Architecture:
  Core/         — фундаментальные алгоритмы (Copy, Fill, Map, Reduce, Scan, Sort)
  LinearAlgebra/ — матричные операции (MatMul, Transpose)
  Signal/       — обработка сигналов (FFT, Convolution)
"""

from .Core.Map import describe as _describe_map, cpu as _cpu_map, wgsl as _wgsl_map
from .Core.Reduce import describe as _describe_reduce, cpu as _cpu_reduce, wgsl as _wgsl_reduce
from .Core.Scan import (
    describe_local, cpu_local, wgsl_local,
    describe_totals, cpu_totals, wgsl_totals,
    describe_final, cpu_final, wgsl_final,
)
from .Core.Histogram import describe as _describe_histogram, cpu_histogram as _cpu_histogram, wgsl_histogram as _wgsl_histogram
from .Core.Sort import describe_sort as _describe_sort, cpu_sort as _cpu_sort, wgsl_sort as _wgsl_sort
from .Core.MatMul import describe as _describe_matmul, cpu as _cpu_matmul, wgsl as _wgsl_matmul
from .Core.FFT import (
    describe_stage, describe_bitreverse,
    cpu_stage, cpu_bitreverse,
    wgsl_stage, wgsl_bitreverse,
)


def register_all(runtime):
    """Register all Compute kernels in a Runtime instance."""
    runtime.register_kernel(
        "Map",
        describe=_describe_map,
        cpu=_cpu_map,
        wgsl=_wgsl_map,
        abi_version=1,
        capabilities={
            "streaming": False,
            "workspace": False,
            "multi_input": False,
            "multi_output": False,
        },
    )
    runtime.register_kernel(
        "Reduce",
        describe=_describe_reduce,
        cpu=_cpu_reduce,
        wgsl=_wgsl_reduce,
        abi_version=1,
        capabilities={
            "streaming": False,
            "workspace": True,
            "multi_input": False,
            "multi_output": False,
        },
    )
    runtime.register_kernel(
        "ScanLocal",
        describe=describe_local,
        cpu=cpu_local,
        wgsl=wgsl_local,
        abi_version=1,
        capabilities={
            "streaming": False,
            "workspace": False,
            "multi_input": False,
            "multi_output": True,
        },
    )
    runtime.register_kernel(
        "ScanTotals",
        describe=describe_totals,
        cpu=cpu_totals,
        wgsl=wgsl_totals,
        abi_version=1,
        capabilities={
            "streaming": False,
            "workspace": False,
            "multi_input": False,
            "multi_output": False,
        },
    )
    runtime.register_kernel(
        "ScanFinal",
        describe=describe_final,
        cpu=cpu_final,
        wgsl=wgsl_final,
        abi_version=1,
        capabilities={
            "streaming": False,
            "workspace": False,
            "multi_input": True,
            "multi_output": False,
        },
    )
    runtime.register_kernel(
        "Histogram",
        describe=_describe_histogram,
        cpu=_cpu_histogram,
        wgsl=_wgsl_histogram,
        abi_version=1,
        capabilities={
            "streaming": False,
            "workspace": False,
            "multi_input": False,
            "multi_output": False,
        },
    )
    runtime.register_kernel(
        "Sort",
        describe=_describe_sort,
        cpu=_cpu_sort,
        wgsl=_wgsl_sort,
        abi_version=1,
        capabilities={
            "streaming": False,
            "workspace": False,
            "multi_input": False,
            "multi_output": False,
        },
    )
    runtime.register_kernel(
        "MatMul",
        describe=_describe_matmul,
        cpu=_cpu_matmul,
        wgsl=_wgsl_matmul,
        abi_version=1,
        capabilities={
            "streaming": False,
            "workspace": False,
            "multi_input": True,
            "multi_output": False,
        },
    )
    runtime.register_kernel(
        "BitReverse",
        describe=describe_bitreverse,
        cpu=cpu_bitreverse,
        wgsl=wgsl_bitreverse,
        abi_version=1,
        capabilities={
            "streaming": False,
            "workspace": False,
            "multi_input": False,
            "multi_output": False,
        },
    )
    runtime.register_kernel(
        "FftStage",
        describe=describe_stage,
        cpu=cpu_stage,
        wgsl=wgsl_stage,
        abi_version=1,
        capabilities={
            "streaming": False,
            "workspace": False,
            "multi_input": False,
            "multi_output": False,
        },
    )


__all__ = ["register_all"]
