"""Recurrent — рекуррентный фильтр: out[i] = a*in[i] + b*out[i-1].

Modes:
  0 = single:  out[i] = a*in[i] + b*out[i-1]  (EMA: a=alpha, b=1-alpha)
  1 = dual:    out0[i] = a*in0[i] + b*out0[i-1]
               out1[i] = a*in1[i] + b*out1[i-1] (RSI: gain/loss)
"""

from .descriptor import describe
from .cpu import cpu
from .wgsl import WGSL as wgsl, WGSL_DUAL

__all__ = ["describe", "cpu", "wgsl", "WGSL_DUAL"]
