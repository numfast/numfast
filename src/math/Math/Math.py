"""Math Extension — element-wise operations on Series.

Re-exports from Series._lib.math_ops as Builder mods.
"""


def sin(x):
    from Series._lib.math_ops import sin as _sin
    return _sin(x)


def cos(x):
    from Series._lib.math_ops import cos as _cos
    return _cos(x)


def tan(x):
    from Series._lib.math_ops import tan as _tan
    return _tan(x)


def exp(x):
    from Series._lib.math_ops import exp as _exp
    return _exp(x)


def log(x):
    from Series._lib.math_ops import log as _log
    return _log(x)


def sqrt(x):
    from Series._lib.math_ops import sqrt as _sqrt
    return _sqrt(x)


def neg(x):
    from Series._lib.math_ops import neg as _neg
    return _neg(x)
