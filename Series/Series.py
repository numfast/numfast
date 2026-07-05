from Series._lib.series_lib import _to_array, _rolling, _normalize, _rolling_series, _normalize_series
from Series._lib.numeric_series import NumericSeries, make_series, series_add, series_sub
from Series._lib.base_series import BaseSeries
from Series._lib.object_series import ObjectSeries
from Series._lib.text_series import TextSeries
from Series._lib.image_series import ImageSeries
from Series._lib.tensor_series import TensorSeries
from _core.series_types import SeriesKind


def series_main(data: list | tuple) -> object:
    return _to_array(data)


def rolling_window(data: list | tuple, window: int = 3) -> object:
    return _rolling_series(list(data), window)


def normalize_series(data: list | tuple) -> object:
    return _normalize_series(list(data))
