# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""NumFast — GPU-first numerical computing framework.

High-Level API:
    from numfast import scan, matmul, fft, sort, histogram

Agent SDK:
    from numfast import AgentSDK

    sdk = AgentSDK()
    with sdk.session() as session:
        session.run("examples/moving_average.py")
"""

# Operations — high-level API
from numfast.operations import scan, matmul, fft, sort, histogram

from numfast._core.context import create_context
from numfast._core.series_types import SeriesKind
from numfast._core.series_proxy import SeriesProxy
from numfast.Series._lib import NumericSeries, make_series
from numfast.Series._lib.math_ops import sin, cos, tan, exp, log, sqrt, neg
from numfast.Series._lib.expr import _LazyExpr
from numfast._core.backend import get_xp, set_active, get_active_name
from numfast.Stats.Stats import total, mean, var, std, minimum, maximum, count, range_

# Storage API
from numfast._kernel import (
    Context,
    create_table,
    _kernel as _ctx,
)

Table = _ctx.Table
Column = _ctx.Column
ColumnType = _ctx.ColumnType
PackingPlan = _ctx.PackingPlan
ColumnLayout = _ctx.ColumnLayout

# Pipeline
from numfast.Pipeline import Pipeline, Task

# Trading indicators — pure-function kernels, register via register_all(runtime)
from numfast.Trading import register_all as register_trading_kernels

# Compute — вычислительные ядра
from numfast.Compute import register_all as register_compute_kernels

# Runtime — исполнительный слой
from numfast.Runtime import (
    Runtime, CpuDriver, Task as RuntimeTask, compile as compile_jobs,
    BlockView, BufferView, ExecutionPacket,
    validate_uniforms, KernelValidator,
)

# Agent SDK — программный слой для автоматизации
from numfast.agent import AgentSDK, Result

__all__ = [
    "Context", "create_table",
    "Table", "Column", "ColumnType", "PackingPlan", "ColumnLayout",
    "SeriesProxy",
    "Pipeline", "Task",
    "register_trading_kernels",
    "register_compute_kernels",
    "create_context", "SeriesKind",
    "NumericSeries", "make_series",
    "sin", "cos", "tan", "exp", "log", "sqrt", "neg",
    "get_xp", "set_active", "get_active_name",
    "range_",
    "Runtime", "CpuDriver", "compile_jobs", "RuntimeTask",
    "BlockView", "BufferView", "ExecutionPacket",
    "validate_uniforms", "KernelValidator",
    "scan", "matmul", "fft", "sort", "histogram",
    "AgentSDK", "Result",
]

__version__ = "0.1.0"
