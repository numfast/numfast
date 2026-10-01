# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# numfast wheel — packaging notes

`pip install dist/numfast-*.whl` gives a self-contained `import numfast`:

- Engine Extension sources are vendored into the wheel (`numfast/_ext/*`)
  verbatim at build time by `setup.py`; the repo tree stays clean.
- The Rust DLL ships as `numfast/_native/numfast_native.dll` and resolves
  via package-relative path (`NUMFAST_NATIVE_DLL` env overrides,
  `NUMFAST_NATIVE_DISABLE=1` forces the NumPy fallback).
- The wheel builder (`src/numfast/_builder/`) is a fresh minimal
  implementation: no import from the dev-tree `app-builder-ponytail`,
  no hardcoded filesystem roots.
- Optional deps: `pip install numfast[pandas]`, `numfast[arrow]`.
  Without pyarrow the CPU path keeps working (NumPy fallback).

Boundary checks after install:

```bash
python -I -c "import numfast as nf; s = nf.from_numpy([1,2,3]); print(nf.to_numpy(s * 2))"
python -I -c "import numfast as nf; print(nf.native_info())"
NUMFAST_NATIVE_DISABLE=1 python -I -c "import numfast as nf; print(nf.to_numpy(nf.from_numpy([1.5,2.5]).reduce('sum')))"
```
