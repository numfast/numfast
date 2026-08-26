"""StateKernel — рекуррентный процессор с state-машиной.

Modes:
  0 = Single EMA:    out[i] = a*in[i] + b*out[i-1]
  1 = Dual EMA:      out0[i] = a*in0[i] + b*out0[i-1]
                     out1[i] = a*in1[i] + b*out1[i-1]
  2 = SuperTrend:    state machine with trailing stops

Parallel 3-pass scan (StateKernelScanLocal/Totals/Final): та же линейная
рекурренция через chunk-декомпозицию (chunk = 256 по умолчанию):
  Local   — локальный ряд с нулевым состоянием + last[tid]
  Totals  — sp[t] = last[t-1] + b^chunk * sp[t-1], sp[0] = 0
  Final   — out[i] = tmp[i] + b^(j+1) * sp[tid]
"""

from .descriptor import describe
from .descriptor import describe_scan_local, describe_scan_totals, describe_scan_final
from .cpu import cpu
from .cpu import cpu_scan_local, cpu_scan_totals, cpu_scan_final
from .wgsl import WGSL_SINGLE, WGSL_DUAL, WGSL_SUPERTREND
from .wgsl import (WGSL_SCAN_LOCAL, WGSL_SCAN_LOCAL_DUAL,
                   WGSL_SCAN_TOTALS, WGSL_SCAN_TOTALS_DUAL,
                   WGSL_SCAN_FINAL, WGSL_SCAN_FINAL_DUAL)

__all__ = [
    "describe", "cpu", "WGSL_SINGLE", "WGSL_DUAL", "WGSL_SUPERTREND",
    "describe_scan_local", "describe_scan_totals", "describe_scan_final",
    "cpu_scan_local", "cpu_scan_totals", "cpu_scan_final",
    "WGSL_SCAN_LOCAL", "WGSL_SCAN_LOCAL_DUAL",
    "WGSL_SCAN_TOTALS", "WGSL_SCAN_TOTALS_DUAL",
    "WGSL_SCAN_FINAL", "WGSL_SCAN_FINAL_DUAL",
]
