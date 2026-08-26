import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import pytest
from _core.backend import set_active, get_active_name, get_xp
from _core.container import compute_layout, pack_rows, extract_column
from _core import kernel
from _core.backend._wgpu import generate_i64_moments_shader, compute_column_moments_wgpu

INT64_TEST_VALUES = [
    ([0], 0, 0, 0),
    ([1], 1, 1, 1),
    ([-1], -1, -1, -1),
    ([42, 100, -5, 7], 144, -5, 100),
    ([1 << 60, (1 << 60) + 1000], (1 << 60) + (1 << 60) + 1000, 1 << 60, (1 << 60) + 1000),
    ([-9223372036854775808], -9223372036854775808, -9223372036854775808, -9223372036854775808),
    ([9223372036854775807], 9223372036854775807, 9223372036854775807, 9223372036854775807),
    ([999999999999999999, -999999999999999999], 0, -999999999999999999, 999999999999999999),
]


def _register_int64_table(values):
    schema = [{'name': 'big', 'dtype': 'int64', 'bits': 64}]
    layout = compute_layout(schema)
    data = {'big': values}
    packed = pack_rows(schema, data)
    tid, _ = kernel._register_table(packed['rows'], packed['num_parts'],
                                     packed['schema'], packed['layout'])
    entry = kernel._get_table_entry(tid)
    return entry, layout


def test_i64_extract():
    for vals, expected_sum, expected_min, expected_max in INT64_TEST_VALUES:
        schema = [{'name': 'big', 'dtype': 'int64', 'bits': 64}]
        layout = compute_layout(schema)
        data = {'big': vals}
        packed = pack_rows(schema, data)
        extracted = extract_column(packed['rows'], 'big', packed['layout'])
        assert extracted == vals, f"extract({vals}): got {extracted}"


def test_i64_cpu_stats():
    for vals, expected_sum, expected_min, expected_max in INT64_TEST_VALUES:
        entry, layout = _register_int64_table(vals)
        from _core.container import extract_column
        extracted = extract_column(entry['rows'], 'big', layout)
        total_cpu = sum(extracted)
        assert total_cpu == expected_sum, \
            f"CPU sum({vals}): expected {expected_sum}, got {total_cpu}"
        assert min(extracted) == expected_min, \
            f"CPU min({vals}): expected {expected_min}, got {min(extracted)}"
        assert max(extracted) == expected_max, \
            f"CPU max({vals}): expected {expected_max}, got {max(extracted)}"


@pytest.mark.skipif(get_active_name() != "wgpu",
                    reason="requires WebGPU backend")
def test_i64_wgpu_stats():
    for vals, expected_sum, expected_min, expected_max in INT64_TEST_VALUES:
        entry, layout = _register_int64_table(vals)
        result = compute_column_moments_wgpu(
            entry['rows'], entry['num_parts'],
            layout, 'big', entry['num_rows']
        )
        assert result['count'] == len(vals)
        assert result['sum'] == expected_sum, \
            f"WGSL sum({vals}): expected {expected_sum}, got {result['sum']}"
        assert result['min'] == expected_min, \
            f"WGSL min({vals}): expected {expected_min}, got {result['min']}"
        assert result['max'] == expected_max, \
            f"WGSL max({vals}): expected {expected_max}, got {result['max']}"


@pytest.mark.skipif(get_active_name() != "wgpu",
                    reason="requires WebGPU backend")
def test_i64_wgpu_single_workgroup():
    """256 rows, exact 1 workgroup. Symmetric around zero, sum=0."""
    vals = [(i - 128) * (1 << 50) for i in range(256)]
    entry, layout = _register_int64_table(vals)
    result = compute_column_moments_wgpu(
        entry['rows'], entry['num_parts'],
        layout, 'big', entry['num_rows']
    )
    expected_sum = sum(vals)
    expected_min = min(vals)
    expected_max = max(vals)
    assert result['count'] == len(vals)
    assert result['sum'] == expected_sum, \
        f"sum: expected {expected_sum}, got {result['sum']}"
    assert result['min'] == expected_min, \
        f"min: expected {expected_min}, got {result['min']}"
    assert result['max'] == expected_max, \
        f"max: expected {expected_max}, got {result['max']}"


@pytest.mark.skipif(get_active_name() != "wgpu",
                    reason="requires WebGPU backend")
def test_i64_wgpu_multi_workgroup():
    """500 rows, spans 2 workgroups, tests reduction."""
    vals = [(i - 250) * (1 << 52) for i in range(500)]
    entry, layout = _register_int64_table(vals)
    result = compute_column_moments_wgpu(
        entry['rows'], entry['num_parts'],
        layout, 'big', entry['num_rows']
    )
    expected_sum = sum(vals)
    expected_min = min(vals)
    expected_max = max(vals)
    assert result['count'] == len(vals)
    assert result['sum'] == expected_sum, \
        f"sum: expected {expected_sum}, got {result['sum']}"
    assert result['min'] == expected_min, \
        f"min: expected {expected_min}, got {result['min']}"
    assert result['max'] == expected_max, \
        f"max: expected {expected_max}, got {result['max']}"


@pytest.mark.skipif(get_active_name() != "wgpu",
                    reason="requires WebGPU backend")
def test_i64_wgpu_negative_only():
    """All negative values with sum staying within i64 range."""
    vals = [-i * (1 << 50) for i in range(1, 11)]  # sum ≈ -11*6*2^49 ≈ -3.7e15, safely in i64 range
    entry, layout = _register_int64_table(vals)
    result = compute_column_moments_wgpu(
        entry['rows'], entry['num_parts'],
        layout, 'big', entry['num_rows']
    )
    expected_sum = sum(vals)
    expected_min = min(vals)
    expected_max = max(vals)
    assert result['sum'] == expected_sum
    assert result['min'] == expected_min
    assert result['max'] == expected_max
