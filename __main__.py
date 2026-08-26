"""NumFast CLI entry point.

Usage:
    python -m numfast [command]
"""
import sys
import pathlib

# Add src/ and ALL group subdirectories to sys.path
_base = pathlib.Path(__file__).resolve().parent / 'src'
for _sub in ['', 'core', 'math', 'trading']:
    _p = str(_base / _sub) if _sub else str(_base)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _main._main import diagnostics

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("info", "status", "diagnostics"):
        diagnostics()
    else:
        print("NumFast v0.1.0")
        print("Usage: python -m numfast diagnostics")
