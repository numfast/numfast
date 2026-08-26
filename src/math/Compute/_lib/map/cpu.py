"""Map kernel — CPU reference implementation.

Domain semantics (S38): unified NaN propagation. The CPU reference
computes in numpy (not math.*), wrapped in np.errstate(all='ignore') —
it returns NaN/+-Inf per IEEE-754 (like the WGSL f32 shader) and NEVER
raises ValueError or OverflowError inside the domain.

Domain table (func -> input -> result):
  0 sin / 1 cos | |x|<=1e3          | np.sin/np.cos (f64)      | parity rel 1e-6
  0 sin / 1 cos | +-Inf / NaN       | NaN / NaN               | matches GPU
  2 exp         | x<=80             | np.exp(x)               | parity rel 1e-6
  2 exp         | 80<x<=88.72       | finite (f64)            | band (characterization)
  2 exp         | 88.72<x<=709.78   | finite (f64)            | GPU +Inf (band)
  2 exp         | x>709.78          | +Inf                    | matches GPU (+Inf)
  2 exp         | x<-87.34          | finite/0 (f64, to -745) | GPU 0 (FTZ, band)
  2 exp         | x<-745            | 0                       | matches GPU (0)
  2 exp         | +Inf / -Inf / NaN | +Inf / 0 / NaN          | matches GPU
  3 sqrt        | x<0               | NaN                     | matches GPU
  3 sqrt        | x=0 / x>0         | 0 / sqrt(x)             | parity
  3 sqrt        | +Inf / NaN        | +Inf / NaN              | matches GPU
  4 log         | x=0               | -Inf                    | matches GPU
  4 log         | x<0               | NaN                     | matches GPU
  4 log         | x>0               | log(x)                  | parity
  4 log         | +Inf / NaN        | +Inf / NaN              | matches GPU
  5 abs         | any f32           | abs                     | exact
  6 neg         | any f32           | -x                      | exact
  7 square      | |x|<=1e5          | x*x                     | parity rel 1e-6
  7 square      | 1.84e19<|x|<=3.4e38 | finite (f64)          | GPU +Inf (band)
  7 square      | |x|>1.34e154      | +Inf (f64 overflow)     | matches GPU (Inf)

Thresholds: f32 exp overflow ln(3.4028235e38)=88.72, f64 ln(1.8e308)=709.78;
f32 exp underflow ~-87.34 (subnormal -> FTZ -> 0), f64 ~-745;
f32 square overflow sqrt(3.4e38)=1.8447e19, f64 sqrt(1.8e308)=1.34e154.
Bands between the f32/f64 thresholds are characterization: parity is not
defined there (recorded in evidence, NOT a failure).
"""

import numpy as np
from Runtime._lib.mod_iface import ExecutionContext


_FUNCS = {
    0: np.sin,
    1: np.cos,
    2: np.exp,
    3: np.sqrt,
    4: np.log,
    5: np.abs,
    6: np.negative,
    7: np.square,
}


def cpu(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    func_code = int(ctx.uniforms.get("func", 7))
    func = _FUNCS.get(func_code)
    if func is None:
        # Defense-in-depth (S39): the canonical validation lives in
        # descriptor.describe() (single point, both backends, before
        # dispatch); keep this check as well.
        raise ValueError(f"Unknown Map function code: {func_code}")
    with np.errstate(all="ignore"):
        for i in range(src.length()):
            dst.write(i, func(src.read(i)))