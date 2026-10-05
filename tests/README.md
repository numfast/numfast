# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# tests/ — the suite

## Layout

```
tests/
  conftest.py      gate markers, and the installed-package mode
  fast/            the suite; `pytest tests/` runs it
  oracles/         independent reference implementations the tests check against
```

There is one tier. `tests/fast/` is seconds-scale and is what the default
selection runs.

## Rules

- Markers: `fast`, registered in `conftest.py`. Unmarked is treated as `fast`.
- `heavy` exists for **one** test — `test_ops_null_pattern.py`'s wall-clock ratio
  assertion, which is a property of the machine rather than of the engine. It is
  skipped by default and opt-in:
  `pytest -m heavy tests/fast/test_ops_null_pattern.py -s`. There is no
  `tests/heavy/` directory.
- Fixed seed `42` for every reproducible test.
- Numeric thresholds come **only** from
  `../specs-rebuilt/conformance-profile.toml` (atol/rtol/ULP,
  `max_dict_entries`) — `fast/harness.py` reads it, nothing hardcodes a
  tolerance.
- A test that documents a cost must say in its docstring how much memory and
  time it uses and what it is proving.

## Commands

```bash
pytest tests/            # the whole suite
pytest tests/fast -q     # the same thing, named explicitly
```

## Two modes: checkout and installed package

The same suite runs in two modes, and `conftest.py` picks the mode itself, from
where `numfast` resolves.

**Checkout** (developer mode): `numfast` resolves in this repository's `src/`,
`builder` is the real one from app-builder. Everything is collected, nothing is
skipped.

```bash
export PYTHONPATH="C:/App/numfast/numfast/src;C:/App/numfast/app-builder"
pytest tests/
```

**Installed package** (what ships): `numfast` resolves in site-packages and
there is no engine tree. The kernel comes from `numfast.get_kernel()` — an
installed package finds its own fork root from `numfast/full.toml`, so
`APP_DIR`/`FORK` are not inputs. `tests/` and `specs-rebuilt/` are needed (the
sdist ships both, via `MANIFEST.in`) but `src/`, the repository-root `full.toml`
and `numfast-native/` are not:

```bash
python -m venv .venv-wheel
.\.venv-wheel\Scripts\python.exe -m pip install numfast-0.2.1-py3-none-win_amd64.whl[pandas,test]
# unpack the sdist (or copy tests/ + specs-rebuilt/) OUTSIDE the repository tree
.\.venv-wheel\Scripts\python.exe -m pytest <that-dir>/tests -q
```

Modules that need the engine source are not collected in this mode, and
`conftest.py` prints the name and the reason for each at the end of the run
(the list is `conftest.py:SOURCE_ONLY`). Dropping them silently is not allowed:
they are where the text guards over the engine sources live, and without the tree
there is nothing to guard.
