"""Series Extension -- data operations, rolling, normalization.
Builder entry point.
"""


def rolling_window(data, window=3):
    from Series._lib.series_lib import _rolling_series
    return _rolling_series(list(data), window)


def normalize_series(data):
    from Series._lib.series_lib import _normalize_series
    return _normalize_series(list(data))


def series(data, ctx=None, name=None):
    from Series._lib.numeric_series import make_series as _ms
    if ctx is None:
        from _core.context import create_context
        ctx = create_context()
    return _ms(data, ctx, name)


def NumericSeries(data):
    from Series._lib.numeric_series import NumericSeries as _NS
    return _NS(data)


def make_series(data, ctx=None, name=None):
    from Series._lib.numeric_series import make_series as _ms
    if ctx is None:
        from _core.context import create_context
        ctx = create_context()
    return _ms(data, ctx, name)
