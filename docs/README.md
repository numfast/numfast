# `docs/` — what is here, and what to believe

This directory holds user-facing documentation for the tree as it stands. It
describes the API that exists; where it disagrees with the code, the code is
the truth and the document is the bug.

| Document | What it covers |
|---|---|
| [INSTALL.md](INSTALL.md) | how to install, all five channels: Python/PyPI, JavaScript/npm, Browser/WASM, Linux native, Windows native — what each gives you, and how to verify it |
| [API.md](API.md) | the 44-name consumer surface, the package boundary, the kernel-level names, and every guard |
| [ARCHITECTURE.md](ARCHITECTURE.md) | the IR, the Builder Extension model, the planner, the CPU driver, the native Rust kernels, the GPU split, the WASM path, and where the semantic contract lives |
| [EXAMPLES.md](EXAMPLES.md) | runnable programs with their real output |
| [`../showcase/README.md`](../showcase/README.md) | five executable notebooks: `Table`/`Series`, the GPU path, the WASM package, the comparative matrix, and a full task cross-checked against pandas |

The repository root also carries [README.md](../README.md),
[KNOWN_LIMITATIONS.md](../KNOWN_LIMITATIONS.md),
[README.packaging.md](../README.packaging.md),
[README_PYPI.md](../README_PYPI.md),
[CHANGELOG.md](../CHANGELOG.md),
[CORRESPONDING-SOURCE.md](../CORRESPONDING-SOURCE.md),
[COMMERCIAL-LICENCE.md](../COMMERCIAL-LICENCE.md), [SECURITY.md](../SECURITY.md)
and [LICENSE](../LICENSE).

Nothing in `docs/` is historical. A document that described a superseded design
generation does not belong here: it reads as current, it disagrees with the API,
and a reader has no way to tell which is which.
