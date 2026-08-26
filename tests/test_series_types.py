# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

import pytest

from _core.context import create_context
from _core.series_types import SeriesKind


class TestSeriesKind:
    def test_numeric_kind(self):
        from Series._lib import NumericSeries
        ctx = create_context("test_type")
        s = NumericSeries([1.0, 2.0, 3.0], ctx)
        assert s.kind == SeriesKind.NUMERIC

    def test_numeric_ndim(self):
        from Series._lib import NumericSeries
        ctx = create_context("test_ndim")
        s = NumericSeries([1.0, 2.0, 3.0], ctx)
        assert s.ndim == 1

    def test_numeric_shape(self):
        from Series._lib import NumericSeries
        ctx = create_context("test_shape")
        s = NumericSeries([1.0, 2.0, 3.0], ctx)
        assert s.shape == (3,)

    def test_numeric_len(self):
        from Series._lib import NumericSeries
        ctx = create_context("test_len")
        s = NumericSeries([10, 20, 30, 40, 50], ctx)
        assert len(s) == 5

    def test_numeric_data(self):
        from Series._lib import NumericSeries
        ctx = create_context("test_data")
        s = NumericSeries([1.5, 2.5, 3.5], ctx)
        d = s.data()
        assert d == [1.5, 2.5, 3.5]

    def test_numeric_add(self):
        from Series._lib import NumericSeries
        ctx = create_context("test_add")
        a = NumericSeries([1.0, 2.0, 3.0], ctx)
        b = NumericSeries([4.0, 5.0, 6.0], ctx)
        c = a + b
        assert c.data() == [5.0, 7.0, 9.0]

    def test_numeric_sub(self):
        from Series._lib import NumericSeries
        ctx = create_context("test_sub")
        a = NumericSeries([10.0, 20.0, 30.0], ctx)
        b = NumericSeries([1.0, 2.0, 3.0], ctx)
        c = a - b
        assert c.data() == [9.0, 18.0, 27.0]

    def test_numeric_proxy_delegation(self):
        from Series._lib import NumericSeries
        ctx = create_context("test_proxy")
        s = NumericSeries([1.0, 2.0], ctx)
        assert s["_table_id"] is not None
        assert s["_col_name"] == "_value"
        assert s["_is_hidden_series"] is True


class TestSeriesStubs:
    def test_object_series(self):
        from Series._lib import ObjectSeries
        os = ObjectSeries()
        assert os.kind == SeriesKind.OBJECT
        assert os.ndim == 1

    def test_text_series(self):
        from Series._lib import TextSeries
        ts = TextSeries()
        assert ts.kind == SeriesKind.TEXT
        assert ts.ndim == 1

    def test_image_series_default_ndim(self):
        from Series._lib import ImageSeries
        img = ImageSeries()
        assert img.kind == SeriesKind.IMAGE
        assert img.ndim == 1

    def test_image_series_custom_ndim(self):
        from Series._lib import ImageSeries
        img = ImageSeries(shape=(224, 224, 3))
        assert img.ndim == 3
        assert img.shape == (224, 224, 3)

    def test_image_series_2d(self):
        from Series._lib import ImageSeries
        img = ImageSeries(shape=(28, 28))
        assert img.ndim == 2
        assert img.shape == (28, 28)

    def test_tensor_series_default(self):
        from Series._lib import TensorSeries
        t = TensorSeries()
        assert t.kind == SeriesKind.TENSOR

    def test_tensor_series_4d(self):
        from Series._lib import TensorSeries
        t = TensorSeries(shape=(32, 3, 224, 224))
        assert t.ndim == 4
        assert t.shape == (32, 3, 224, 224)

    def test_tensor_series_1d(self):
        from Series._lib import TensorSeries
        t = TensorSeries(shape=(100,))
        assert t.ndim == 1
        assert t.shape == (100,)


class TestKindProperties:
    def test_numeric_is_numeric(self):
        assert SeriesKind.NUMERIC.is_numeric is True

    def test_object_not_numeric(self):
        assert SeriesKind.OBJECT.is_numeric is False

    def test_text_not_numeric(self):
        assert SeriesKind.TEXT.is_numeric is False

    def test_image_not_numeric(self):
        assert SeriesKind.IMAGE.is_numeric is False

    def test_tensor_not_numeric(self):
        assert SeriesKind.TENSOR.is_numeric is False


class TestBackwardCompat:
    def test_make_series_still_works(self):
        from Series._lib import make_series
        ctx = create_context("test_compat")
        s = make_series([1.0, 2.0, 3.0], ctx)
        assert s.data() == [1.0, 2.0, 3.0]

    def test_series_add_still_works(self):
        from Series._lib import make_series, series_add
        ctx = create_context("test_add_compat")
        a = make_series([1.0, 2.0], ctx)
        b = make_series([3.0, 4.0], ctx)
        c = series_add(a, b)
        assert c.data() == [4.0, 6.0]

    def test_series_sub_still_works(self):
        from Series._lib import make_series, series_sub
        ctx = create_context("test_sub_compat")
        a = make_series([5.0, 6.0], ctx)
        b = make_series([1.0, 2.0], ctx)
        c = series_sub(a, b)
        assert c.data() == [4.0, 4.0]

    def test_repr(self):
        from Series._lib import NumericSeries
        ctx = create_context("test_repr")
        s = NumericSeries([1.0], ctx)
        r = repr(s)
        assert "NumericSeries" in r
        assert "numeric" in r


class TestStatsWithNumericSeries:
    def test_total_via_numeric(self):
        from Series._lib import NumericSeries
        from Stats.Stats import total
        ctx = create_context("test_stats_n")
        s = NumericSeries([10.0, 20.0, 30.0], ctx)
        assert abs(total(s) - 60.0) < 0.001

    def test_mean_via_numeric(self):
        from Series._lib import NumericSeries
        from Stats.Stats import mean
        ctx = create_context("test_mean_n")
        s = NumericSeries([2.0, 4.0, 6.0], ctx)
        assert abs(mean(s) - 4.0) < 0.001
