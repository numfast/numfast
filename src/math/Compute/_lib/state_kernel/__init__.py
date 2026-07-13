"""StateKernel — рекуррентный процессор с state-машиной.

Modes:
  0 = Single EMA:    out[i] = a*in[i] + b*out[i-1]
  1 = Dual EMA:      out0[i] = a*in0[i] + b*out0[i-1]
                     out1[i] = a*in1[i] + b*out1[i-1]
  2 = SuperTrend:    state machine with trailing stops
"""

from .descriptor import describe
from .cpu import cpu
from .wgsl import WGSL_SINGLE, WGSL_DUAL, WGSL_SUPERTREND

__all__ = ["describe", "cpu", "WGSL_SINGLE", "WGSL_DUAL", "WGSL_SUPERTREND"]