# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.arrow_dense import dict_encode_arrow_impl as _arrow_encode_impl
from _lib.arrow_dense import dict_encode_arrow_metadata_impl as _arrow_meta_impl
from _lib.dictionary import dictionary_decode_impl as _decode_impl
from _lib.dictionary import dictionary_encode_impl as _encode_impl
from _lib.dictionary import dictionary_metadata_impl as _metadata_impl
from _lib.domain import codes_count_allowed_impl as _count_impl
from _lib.domain import codes_member_mask_impl as _member_impl
from _lib.domain import dict_contains_impl as _contains_impl
from _lib.domain import dict_equal_codes_impl as _equal_impl
from _lib.domain import dict_len_lut_impl as _len_lut_impl
from _lib.domain import dict_lookup_impl as _lookup_impl
from _lib.domain import dict_min_max_impl as _min_max_impl
from _lib.domain import dict_not_empty_codes_impl as _not_empty_impl
from _lib.domain import dict_ordering_impl as _ordering_impl
from _lib.domain import dict_range_codes_impl as _range_impl
from _lib.domain import dict_startswith_impl as _startswith_impl

_box = {}


def dictionary_encode(values, validity=None):
    a = _box["kernel"].alias
    return _encode_impl(values, validity, a["format_error"])


def dictionary_decode(codes, values, validity=None):
    a = _box["kernel"].alias
    return _decode_impl(codes, values, validity, a["format_error"])


def dictionary_metadata(values):
    a = _box["kernel"].alias
    return _metadata_impl(values, a["format_error"])


def dict_encode_arrow(values, validity=None, order="sorted"):
    a = _box["kernel"].alias
    return _arrow_encode_impl(values, validity, order, a["format_error"])


def dict_encode_arrow_metadata(enc):
    a = _box["kernel"].alias
    return _arrow_meta_impl(enc, a["format_error"])


def _domain(fn_name):
    def call(*args):
        a = _box["kernel"].alias
        impl = {"dict_lookup": _lookup_impl, "dict_equal_codes": _equal_impl,
                "dict_not_empty_codes": _not_empty_impl, "dict_contains": _contains_impl,
                "dict_startswith": _startswith_impl, "dict_len_lut": _len_lut_impl,
                "dict_ordering": _ordering_impl, "dict_min_max": _min_max_impl,
                "dict_range_codes": _range_impl,
                "codes_member_mask": _member_impl,
                "codes_count_allowed": _count_impl}[fn_name]
        return impl(*args, format_error=a["format_error"])
    call.__name__ = fn_name
    return call


dict_lookup = _domain("dict_lookup")
dict_equal_codes = _domain("dict_equal_codes")
dict_not_empty_codes = _domain("dict_not_empty_codes")
dict_contains = _domain("dict_contains")
dict_startswith = _domain("dict_startswith")
dict_len_lut = _domain("dict_len_lut")
dict_ordering = _domain("dict_ordering")
dict_min_max = _domain("dict_min_max")
dict_range_codes = _domain("dict_range_codes")
codes_member_mask = _domain("codes_member_mask")
codes_count_allowed = _domain("codes_count_allowed")


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("Dictionary", {})["version"] = "0.1.0"


PUBLIC = {
    "dictionary_encode": dictionary_encode,
    "dictionary_decode": dictionary_decode,
    "dictionary_metadata": dictionary_metadata,
    "dict_lookup": dict_lookup,
    "dict_equal_codes": dict_equal_codes,
    "dict_not_empty_codes": dict_not_empty_codes,
    "dict_contains": dict_contains,
    "dict_startswith": dict_startswith,
    "dict_len_lut": dict_len_lut,
    "dict_ordering": dict_ordering,
    "dict_min_max": dict_min_max,
    "dict_range_codes": dict_range_codes,
    "codes_member_mask": codes_member_mask,
    "codes_count_allowed": codes_count_allowed,
    "dict_encode_arrow": dict_encode_arrow,
    "dict_encode_arrow_metadata": dict_encode_arrow_metadata,
}
