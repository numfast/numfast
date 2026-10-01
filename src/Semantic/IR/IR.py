# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.nodes import ir_compare as ir_compare
from _lib.nodes import ir_encode_pattern as ir_encode_pattern
from _lib.nodes import ir_filter as ir_filter
from _lib.nodes import ir_gather as ir_gather
from _lib.nodes import ir_groupby as ir_groupby
from _lib.nodes import ir_groupby_multi as ir_groupby_multi
from _lib.nodes import ir_map as ir_map
from _lib.nodes import ir_mask as ir_mask
from _lib.nodes import ir_where as ir_where
from _lib.nodes import ir_pack_keys as ir_pack_keys
from _lib.nodes import ir_reduce as ir_reduce
from _lib.nodes import ir_rolling_sum as ir_rolling_sum
from _lib.nodes import ir_series as ir_series
from _lib.nodes import ir_slice as ir_slice
from _lib.nodes import ir_shift as ir_shift
from _lib.nodes import ir_cumsum as ir_cumsum
from _lib.nodes import ir_sort as ir_sort


from _lib.nodes import ir_map_round as ir_map_round
from _lib.nodes import ir_rng_compat as ir_rng_compat
from _lib.nodes import ir_rng_fill_f64 as ir_rng_fill_f64
from _lib.nodes import ir_rng_fill_i32 as ir_rng_fill_i32
from _lib.nodes import ir_rng_permutation as ir_rng_permutation
from _lib.nodes import ir_rng_sample_no_replace as ir_rng_sample_no_replace
from _lib.nodes import ir_lookup as ir_lookup
from _lib.nodes import ir_unique_inverse as ir_unique_inverse
from _lib.nodes import ir_unique as ir_unique
from _lib.nodes import ir_count_distinct as ir_count_distinct
from _lib.nodes import ir_text_contains as ir_text_contains
from _lib.nodes import ir_text_endswith as ir_text_endswith
from _lib.nodes import ir_text_equals as ir_text_equals
from _lib.nodes import ir_text_length as ir_text_length
from _lib.nodes import ir_text_startswith as ir_text_startswith
from _lib.nodes import ir_text_regex_replace as ir_text_regex_replace


def setup(kernel):
    kernel.metadata.setdefault("IR", {})["version"] = "0.1.0"


PUBLIC = {"ir_series": ir_series, "ir_map": ir_map, "ir_compare": ir_compare, "ir_filter": ir_filter, "ir_mask": ir_mask, "ir_where": ir_where, "ir_gather": ir_gather, "ir_sort": ir_sort, "ir_slice": ir_slice, "ir_reduce": ir_reduce, "ir_rolling_sum": ir_rolling_sum, "ir_shift": ir_shift, "ir_cumsum": ir_cumsum, "ir_groupby": ir_groupby, "ir_groupby_multi": ir_groupby_multi, "ir_pack_keys": ir_pack_keys, "ir_encode_pattern": ir_encode_pattern, "ir_rng_fill_i32": ir_rng_fill_i32, "ir_rng_fill_f64": ir_rng_fill_f64, "ir_rng_sample_no_replace": ir_rng_sample_no_replace, "ir_rng_permutation": ir_rng_permutation, "ir_rng_compat": ir_rng_compat, "ir_map_round": ir_map_round, "ir_unique_inverse": ir_unique_inverse, "ir_unique": ir_unique, "ir_count_distinct": ir_count_distinct, "ir_lookup": ir_lookup, "ir_text_length": ir_text_length, "ir_text_contains": ir_text_contains, "ir_text_startswith": ir_text_startswith, "ir_text_endswith": ir_text_endswith, "ir_text_equals": ir_text_equals, "ir_text_regex_replace": ir_text_regex_replace}
