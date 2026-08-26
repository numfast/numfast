"""Keltner Channels.

DAG decomposition (GPU via primitives):
  Middle = StateKernel_Single(close, mode=0, a=2/(period_ema+1), b=1-a)  -- L5 Stateful
  TR = TrueRange(high, low, close)       -- L4 Window
  ATR = RollingSum(TR, period_atr) / period_atr  -- L4 + L2
  Upper = Expression("A+B*C", A=Middle, B=ATR, C=mult)  -- Backend JIT
  Lower = Expression("A-B*C", A=Middle, B=ATR, C=mult)  -- Backend JIT

CPU: native kernel (reference).
"""

from .descriptor import describe
from .cpu import cpu

__all__ = ["describe", "cpu"]
