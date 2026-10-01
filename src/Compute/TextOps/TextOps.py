# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.regex_dict import regex_replace_dict_impl as _regex_replace_dict
from _lib.text_length import text_length as text_length

_box = {}


def regex_replace_dict(codes, values, validity=None, pattern=None, repl=None):
    a = _box["kernel"].alias
    return _regex_replace_dict(codes, values, validity, pattern, repl,
                               format_error=a["format_error"])


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("TextOps", {})["version"] = "0.1.0"
    kernel.metadata["TextOps"]["types"] = ["text_length", "regex_replace_dict"]


PUBLIC = {"text_length": text_length, "regex_replace_dict": regex_replace_dict}