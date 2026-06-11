from Series._lib.series_lib import _to_array, _rolling, _normalize

def series_main(data: list | tuple) -> object:
    return _to_array(data)

def rolling_window(data: list | tuple, window: int = 3) -> object:
    arr = _to_array(data)
    return _rolling(arr, window)

def normalize_series(data: list | tuple) -> object:
    arr = _to_array(data)
    return _normalize(arr)
