"""Williams %R.

DAG decomposition (GPU via primitives):
  HH = RollingMax(high, period)       -- L4 Window
  LL = RollingMin(low, period)        -- L4 Window
  %R = Expression("(A-C)/(A-B)*-100.0", A=HH, B=LL, C=close)  -- Backend JIT

CPU: native kernel (reference).
"""

from .descriptor import describe
from .cpu import cpu

__all__ = ["describe", "cpu"]
