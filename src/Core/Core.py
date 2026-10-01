# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.dtypes import accum_dtype as accum_dtype
from _lib.dtypes import canonical_dtype as canonical_dtype
from _lib.dtypes import check_int32_range as check_int32_range
from _lib.dtypes import check_overflow as check_overflow
from _lib.dtypes import is_scaled as is_scaled
from _lib.errors import format_error as format_error


def setup(kernel):
    kernel.metadata.setdefault("Core", {})["version"] = "0.1.0"


PUBLIC = {
    "canonical_dtype": canonical_dtype,
    "accum_dtype": accum_dtype,
    "is_scaled": is_scaled,
    "check_overflow": check_overflow,
    "check_int32_range": check_int32_range,
    "format_error": format_error,
}
