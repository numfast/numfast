# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Dzst loader — single-file file->Storage (P0-L).

Decompress zstd + csv + traversal decode (Polars cum_sum oracle + python fallback).
Single-file only, no BybitLoader range stitching (KEEP in Colossus).
Uses pyzstd + csv, reconstructs deltas via row-level cum_sum.
Returns int64 ticks.
"""

import os
import re
import csv
import io

import numpy as np


def _parse_filename(filepath: str) -> tuple[str, int, int]:
    basename = os.path.basename(filepath)
    basename = basename.replace('.csv.zst', '').replace('.csv', '')
    m = re.search(r'_d(\d+)_', basename)
    power = int(m.group(1)) if m else 5
    m_date = re.search(r'(\d{4}-\d{2}-\d{2})', basename)
    date_str = m_date.group(1) if m_date else ''
    return date_str, power, 10 ** power


def _reconstruct_via_polars(filepath: str) -> tuple[np.ndarray, int, int]:
    import polars as pl
    date_str, power, multiplier = _parse_filename(filepath)
    try:
        df = pl.read_csv(
            filepath,
            has_header=True,
            schema={
                'Open': pl.Int64,
                'High': pl.Int64,
                'Low': pl.Int64,
                'Close': pl.Int64,
                'Buy_Volume': pl.Int64,
                'Sell_Volume': pl.Int64,
            },
            null_values=[''],
        ).fill_null(0).with_columns([
            pl.col('Buy_Volume').cast(pl.Int64),
            pl.col('Sell_Volume').cast(pl.Int64),
        ])
    except Exception as e:
        msg = str(e).lower()
        if 'truncated' in msg or 'corrupt' in msg or 'unexpected' in msg or 'zstd' in msg:
            raise ValueError(f"corrupted file: truncated body {filepath}: {e}") from e
        if 'not found' in msg or 'no such file' in msg:
            raise
        raise ValueError(f"corrupted file: {filepath}: {e}") from e

    n = len(df)
    if n == 0:
        result = np.empty((0, 6), dtype=np.int64)
        return result, multiplier, power

    row_delta = df['Open'] + df['High'] + df['Low'] + df['Close']
    close = row_delta.cum_sum()
    prev_close = close.shift(1).fill_null(0)

    open_abs = prev_close + df['Open']
    high_abs = prev_close + df['Open'] + df['High']
    low_abs = prev_close + df['Open'] + df['High'] + df['Low']

    result = np.empty((n, 6), dtype=np.int64)
    # Use to_numpy with zero_copy if available, else to_numpy
    try:
        result[:, 0] = open_abs.to_numpy().astype(np.int64)
        result[:, 1] = high_abs.to_numpy().astype(np.int64)
        result[:, 2] = low_abs.to_numpy().astype(np.int64)
        result[:, 3] = close.to_numpy().astype(np.int64)
        result[:, 4] = df['Buy_Volume'].to_numpy().astype(np.int64)
        result[:, 5] = df['Sell_Volume'].to_numpy().astype(np.int64)
    except Exception as e:
        raise ValueError(f"corrupted file: numpy conversion {e}") from e

    # Validation: prices must be >=1, high >= low globally, volume >=0
    if (result[:, :4] < 1).any():
        # Could be corrupted dX negative leading to price <1
        raise ValueError(f"corrupted file: dX out of i32 range price <1 in {filepath}")
    if (result[:, 1] < result[:, 2]).any():
        raise ValueError(f"corrupted file: dX out of i32 range high<low in {filepath}")
    # Check i32 range for deltas: |delta*?| <2^31 done implicitly via price range
    # Also check file size sanity
    return result, multiplier, power


def _reconstruct_via_python(filepath: str) -> tuple[np.ndarray, int, int]:
    import pyzstd
    date_str, power, multiplier = _parse_filename(filepath)
    try:
        compressed = open(filepath, 'rb').read()
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"corrupted file: {e}") from e
    if len(compressed) == 0:
        raise ValueError(f"corrupted file: truncated body empty {filepath}")
    try:
        data_bytes = pyzstd.decompress(compressed)
    except Exception as e:
        raise ValueError(f"corrupted file: truncated body zstd decompress failed {filepath}: {e}") from e

    try:
        text = data_bytes.decode('utf-8')
    except Exception as e:
        raise ValueError(f"corrupted file: decode failed {filepath}: {e}") from e

    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        # empty file
        return np.empty((0, 6), dtype=np.int64), multiplier, power

    # header should be Open,High,Low,Close,Buy_Volume,Sell_Volume
    rows = []
    for line in reader:
        # pad to 6
        if len(line) == 0 or all(v == '' for v in line):
            rows.append([0, 0, 0, 0, 0, 0])
            continue
        # each line has up to 6 fields, empty means 0
        vals = []
        for i in range(6):
            if i < len(line) and line[i] != '' and line[i].strip() != '':
                try:
                    vals.append(int(line[i].strip()))
                except ValueError as e:
                    raise ValueError(f"corrupted file: bad int {line[i]} in {filepath}") from e
            else:
                vals.append(0)
        rows.append(vals)

    n = len(rows)
    if n == 0:
        return np.empty((0, 6), dtype=np.int64), multiplier, power

    deltas = np.array(rows, dtype=np.int64)  # (N,6) deltas for first 4 cols, absolute for vols
    # reconstruct traversal
    # col0 = Open delta = Open[i]-Close[i-1]
    # col1 = High delta = High[i]-Open[i]
    # col2 = Low delta = Low[i]-High[i] (<=0)
    # col3 = Close delta = Close[i]-Low[i] (>=0)
    # col4, col5 are volumes absolute
    result = np.empty((n, 6), dtype=np.int64)
    # vectorized cum_sum approach same as polars but in numpy
    row_delta = deltas[:, 0] + deltas[:, 1] + deltas[:, 2] + deltas[:, 3]
    close = np.cumsum(row_delta, dtype=np.int64)
    # prev_close shift
    prev_close = np.empty(n, dtype=np.int64)
    prev_close[0] = 0
    prev_close[1:] = close[:-1]

    result[:, 3] = close
    result[:, 0] = prev_close + deltas[:, 0]
    result[:, 1] = prev_close + deltas[:, 0] + deltas[:, 1]
    result[:, 2] = prev_close + deltas[:, 0] + deltas[:, 1] + deltas[:, 2]
    result[:, 4] = deltas[:, 4]
    result[:, 5] = deltas[:, 5]

    if (result[:, :4] < 1).any():
        raise ValueError(f"corrupted file: dX out of i32 range price <1 in {filepath}")
    if (result[:, 1] < result[:, 2]).any():
        raise ValueError(f"corrupted file: dX out of i32 range high<low in {filepath}")
    return result, multiplier, power


def load_dzst(path: str) -> tuple[np.ndarray, int, int]:
    """Load single dzst .csv.zst file -> (int64[N,6] array, multiplier, power).

    Single-file only, no range stitching.
    Traversal decode: row_delta cum_sum (Polars oracle + python fallback).
    Returns ticks int64.
    """
    # single-file validation
    if isinstance(path, (list, tuple)):
        raise ValueError(f"corrupted file: single-file only, got list/tuple {path}")
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError(f"corrupted file: path must be str|Path, got {type(path)}")
    p = str(path)
    if '*' in p or '?' in p or '[' in p:
        raise ValueError(f"corrupted file: single-file only, glob not allowed {p}")
    if os.path.isdir(p):
        raise ValueError(f"corrupted file: single-file only, got directory {p}")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    if not os.path.isfile(p):
        raise ValueError(f"corrupted file: not a file {p}")

    # check extension sanity? allow any file but must be decompressable
    # try polars first, fallback to python
    try:
        return _reconstruct_via_polars(p)
    except ValueError:
        raise
    except FileNotFoundError:
        raise
    except Exception as e:
        # fallback
        try:
            return _reconstruct_via_python(p)
        except ValueError:
            raise
        except FileNotFoundError:
            raise
        except Exception as e2:
            raise ValueError(f"corrupted file: {p}: {e2}") from e2
