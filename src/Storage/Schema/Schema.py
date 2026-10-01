# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.dates import date_decode_impl as _date_decode_impl
from _lib.dates import date_encode_impl as _date_encode_impl
from _lib.resident import resident_prepare_impl as _resident_prepare_impl
from _lib.schema import column_schema_impl as _column_schema_impl
from _lib.schema import pattern_decode_impl as pattern_decode
from _lib.schema import schema_transform_impl as _schema_transform_impl
from _lib.schema import schema_untransform_impl as schema_untransform

_box = {}


def column_schema(name, logical="int32", scale=1, offset=0, auto=True, physical=None):
    a = _box["kernel"].alias
    return _column_schema_impl(
        name, logical, scale, offset, auto, physical,
        a["canonical_dtype"],
    )


def schema_transform(values, scale=1, offset=0):
    a = _box["kernel"].alias
    return _schema_transform_impl(values, scale, offset, a["check_int32_range"])


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("Schema", {})["version"] = "0.1.0"


def resident_prepare(columns):
    a = _box["kernel"].alias
    return _resident_prepare_impl(columns, a["cpu_execute"], a["dictionary_encode"],
                                  date_encode)


def date_encode(values, validity=None):
    a = _box["kernel"].alias
    return _date_encode_impl(values, validity, a["format_error"])


def date_decode(codes, validity=None):
    a = _box["kernel"].alias
    return _date_decode_impl(codes, validity, a["format_error"])


PUBLIC = {
    "column_schema": column_schema,
    "schema_transform": schema_transform,
    "schema_untransform": schema_untransform,
    "pattern_decode": pattern_decode,
    "resident_prepare": resident_prepare,
    "date_encode": date_encode,
    "date_decode": date_decode,
}
