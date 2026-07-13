"""Pipeline Extension -- sequential task execution.
Builder entry point.
"""


def Pipeline(*args, **kwargs):
    from Pipeline._lib.pipeline import Pipeline as _P
    return _P(*args, **kwargs)


def Task(*args, **kwargs):
    from Pipeline._lib.pipeline import Task as _T
    return _T(*args, **kwargs)
