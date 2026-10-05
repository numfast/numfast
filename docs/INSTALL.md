# Installing NumFast — all five channels in one place

**NumFast 0.2.1.** AGPL-3.0-only. This page is the single answer to "how do I
get this, and what will I actually get". Every statement on it was measured
against the built artefacts, not against a checkout.

The five channels are **not** interchangeable and they are **not** equally
supported. The table is the honest summary:

| Channel | Install | Native kernels | Status |
|---|---|---|---|
| [Python / PyPI](#1-python--pypi) | `pip install numfast` | on Windows x86-64 only | **supported** |
| [JavaScript / npm](#2-javascript--npm) | `npm install @numfast/kernels` | WebAssembly, Node host | **supported (Node only)** |
| [Browser / WASM](#3-browser--wasm) | — | — | **not supported** |
| [Linux native](#4-linux-native) | no artefact | buildable by hand, not shipped | **not supported** |
| [Windows native](#5-windows-native) | arrives with the Windows wheel | yes | **supported** |

---

## 1. Python / PyPI

### Install

```bash
pip install numfast            # the engine
pip install "numfast[pandas]"  # + pandas adapters
pip install "numfast[arrow]"   # + pyarrow adapters
pip install "numfast[gpu]"     # + wgpu, required to execute on the GPU path
pip install "numfast[mem]"     # + psutil, for process/memory reporting
```

### Requirements

- Python **3.11 or newer**. (Only 3.14 has been exercised; see
  [What is verified](#what-is-verified) below.)
- `numpy>=1.24` — the only hard dependency.
- Everything else is an extra. Without `pandas` the engine still works from
  NumPy; without `pyarrow` the CPU path keeps a NumPy fallback.

### What you get

Two wheels are published at the same version. `pip` picks the right one:

| Wheel | Tag | Carries the native library | Who gets it |
|---|---|---|---|
| `numfast-0.2.1-py3-none-win_amd64.whl` | `py3-none-win_amd64` | **yes** | Windows x86-64: full engine, native path live |
| `numfast-0.2.1-py3-none-any.whl` | `py3-none-any` | **no** | Linux, macOS: full engine, native path absent |

The tag is `py3`, not `cp3xx`, on purpose: nothing in the distribution is a
CPython extension module — the native library is loaded through `ctypes` — so
neither wheel is pinned to one interpreter version.

### Minimal example

```python
import pandas as pd
import numfast as nf

orders = pd.DataFrame({
    "region":  ["emea", "apac", "emea", "amer", "apac", "emea"],
    "revenue": [120.0, 80.0, 240.5, 60.0, 95.5, 310.0],
})

result = (nf.from_pandas(orders)
          .query()
          .group("region", {"revenue": ("sum", "count")})
          .sort("revenue.sum", desc=True)
          .compile())

print(result.to_pandas())
```

```
  region  revenue.sum  revenue.count
0   emea        670.5              3
1   apac         175.5              2
2   amer          60.0              1
```

### How to verify it works

```bash
python -c "import numfast as nf; print(nf.__version__); print(nf.native_info())"
```

```
0.2.1
{'disabled': False, 'dll': '.../site-packages/numfast/_native/numfast_native.dll', 'dll_exists': True}
```

On Windows x86-64 `dll_exists` is `True`. **On Linux and macOS it is `False` and
that is the expected result, not a failure**: `numfast/_native/` is simply not
there, and every Rust-backed call falls back to the NumPy CPU path. It is not a
binary that fails to load at run time. `NUMFAST_NATIVE_DISABLE=1` forces the
same path off on Windows; `NUMFAST_NATIVE_DLL` points at a locally built binary.

Full walkthrough with output: **[EXAMPLES.md](EXAMPLES.md)**.

### Running from a clone instead

```bash
export PYTHONPATH="<repo>/src;<path-to-app-builder>"
python examples/quickstart.py
```

The Builder (`app-builder`) is needed only to run the test suite from a
checkout; the installed package carries its own builder and needs nothing else.

---

## 2. JavaScript / npm

### Install

```bash
npm install @numfast/kernels
```

### Requirements

- **Node 22.18 or newer.**
- Nothing else. No native addon, no postinstall step, no compiler.

### What you get

`@numfast/kernels` is a **portable kernel library, not a compute core**: 86
WebAssembly function exports, one memory, zero imports, 17 typed wrappers. There
is no executor, no graph runtime, no IR dispatch and no buffer allocator inside
the `.wasm`. Orchestration, buffer ownership and the type layer stay in your
host.

It is a different artefact from the Python package, not a wrapper around it. If
you want a compute engine, use the Python package.

### Minimal example

```js
import { loadKernels, ssspCsr } from "@numfast/kernels";

const k = await loadKernels();
const dist = ssspCsr(k,
  new Uint32Array([0, 1, 2, 3]),      // indptr
  new Uint32Array([1, 2, 0]),         // indices
  new Uint32Array([10, 20, 30]),      // weights
  0);                                 // source
console.log(dist[1], dist[2]);        // 10 30
```

### How to verify it works

```bash
npm install @numfast/kernels
node --input-type=module -e "
import { loadKernels, ssspCsr, TOTAL_EXPORTS } from '@numfast/kernels';
const k = await loadKernels();
const d = ssspCsr(k, new Uint32Array([0,1,2,3]), new Uint32Array([1,2,0]), new Uint32Array([10,20,30]), 0);
console.log(TOTAL_EXPORTS, d[1], d[2]);   // 86 10 30
"
```

The engine's Python side, and everything else:
**[numfast on GitHub](https://github.com/numfast/numfast)**.

---

## 3. Browser / WASM

### **Not supported.**

There is no browser entry point in the published npm package, and this release
does not add one. Stated precisely, so it is not mistaken for a gap in the
documentation:

- The `.wasm` itself has **zero imports** and does instantiate in a browser.
- **`import "@numfast/kernels"` does not work in a browser.** The package's only
  `"."` export, `dist/index.js`, statically imports `node:module`, `node:fs` and
  `node:crypto`. Bundling the packed tarball with `esbuild --platform=browser`
  fails on all three.
- The browser-capable module, `dist/bridge.js`, bundles cleanly for the browser
  but is **not importable**: the package's `exports` map offers only `.`,
  `./wasm` and `./package.json`, so a subpath import fails with
  `ERR_PACKAGE_PATH_NOT_EXPORTED`.
- The repository ships `numfast-native/ts/demo.html`, which demonstrates the
  pattern against the build tree. It is not in the npm tarball and it is not
  an install path.

Reaching the kernels from a browser today means vendoring `dist/bridge.js` and
the `.wasm` yourself and calling `loadBridge(bytes)`. That is unsupported and
unversioned as a public API.

---

## 4. Linux native

### **No artefact is published. Not supported as an install.**

- There is no `py3-none-linux_x86_64` wheel, no `.so` in the repository, no
  release job and no Linux CI job. Nothing to install.
- **The native library is nevertheless buildable on Linux today**, and this was
  verified rather than assumed. `numfast-native/.cargo/config.toml` is scoped to
  `[target.x86_64-pc-windows-gnu]`, so a plain `cargo build --release` needs no
  `zig` and no other host-specific tool. Measured on Ubuntu 24.04 x86-64 with
  rustc 1.98.1, from the Corresponding Source the wheel already ships:

  ```bash
  cd "$(python -c 'import numfast,os;print(os.path.dirname(numfast.__file__))')/_corresp_src/numfast-native"
  cargo build --release          # -> target/release/libnumfast_native.so
  cp target/release/libnumfast_native.so ../../_native/numfast_native.so
  ```

  After that, `nf.native_info()` reports the library live and the native path is
  used. The crate has no third-party dependencies and no build script, which is
  why this is a single command.

- **Building it yourself is not the same as it being supported.** The Python
  suite on Linux is red: with the library in place and `wgpu` installed, 10 tests
  fail (8 of them native-lane tests in `test_ops_m7b_native_lanes`,
  `test_ops_pairinsert_bitmasksweep` and `test_ops_text_affix`). Do not depend
  on the Linux native path.

**What a Linux user should do instead:** install the `py3-none-any` wheel. It is
the complete engine; only the native kernels are absent.

---

## 5. Windows native

### **Supported. Nothing separate to install.**

The native library ships inside the Windows wheel. There is no separate
download, no installer and no build step:

```bash
pip install numfast     # on Windows x86-64 this is the wheel that carries it
```

### Requirements

- Windows **x86-64**.
- Python 3.11+. No Visual Studio, no Rust toolchain, nothing to compile.

### How to verify it works

```bash
python -c "import numfast as nf; print(nf.native_info())"
```

```
{'disabled': False, 'dll': '...\\numfast\\_native\\numfast_native.dll', 'dll_exists': True}
```

`dll_exists: True` is the whole check. Then:

```python
import numfast as nf
print(nf.to_numpy(nf.from_numpy([1, 2, 3]) * 2))   # [2 4 6]
print(nf.from_numpy([1.5, 2.5]).reduce("sum"))       # 4.0
```

### To disable or relocate it

| variable | effect |
|---|---|
| `NUMFAST_NATIVE_DISABLE=1` | force the NumPy CPU path, even with the library present |
| `NUMFAST_NATIVE_DLL=<path>` | load a specific binary instead of the packaged one |

---

## What is verified, and what is not

The point of this section is that the claims above are bounded by what was
actually run.

| Claim | How it was checked |
|---|---|
| Windows wheel installs and imports | built from a `git archive HEAD` extraction into an empty directory, installed into a clean venv, imported |
| `native_info()` shape on Linux | the `py3-none-any` wheel installed on Ubuntu 24.04 / CPython 3.12 |
| The quick-start output above | the README example run verbatim in both wheel flavours |
| Corresponding Source rebuilds | `cargo build --release` in the wheel's own `_corresp_src/`, on Linux, no `zig` |
| npm tarball works | `npm pack`, installed into a clean project, `loadKernels()` + `ssspCsr()` run against the packed tarball |
| npm browser claim | `esbuild --bundle --platform=browser` against the installed tarball — **fails**, and is documented as unsupported above |
| Windows suite | installed-wheel run: 503 passed, 5 skipped, 0 failed |
| Linux suite | WSL2 Ubuntu 24.04: 10 failed, 729 passed, 37 skipped with the library built |

**Not verified, and therefore not claimed anywhere:**

- **Python 3.11, 3.12 and 3.13.** `pyproject.toml` says `>=3.11` and the wheels
  carry `py3`, so pip will install them on those versions — but the suite has
  only ever been run on 3.14.
- **macOS.** No macOS artefact is built or tested. The `py3-none-any` wheel is
  platform-independent and will install; nothing beyond that is known.
- **Any browser.**
- **A Linux native wheel**, because none exists.

---

## Licence

**AGPL-3.0-only.** The wheel conveys `numfast_native.dll` — object code — so
AGPL-3.0 §6 is discharged by construction: the crate's 47 `.rs` files, its
manifest, lockfile, cargo config and link shims travel inside the wheel under
`numfast/_corresp_src/`, with a `SHA256SUMS` over exactly those bytes. See
**[CORRESPONDING-SOURCE.md](../CORRESPONDING-SOURCE.md)**.

The npm package conveys the compiled `.wasm` and, as of this release, **not** the
Rust source. See the licence section of
[`numfast-native/ts/README.md`](../numfast-native/ts/README.md).

There is no "paid AGPL". See
**[COMMERCIAL-LICENCE.md](../COMMERCIAL-LICENCE.md)**.