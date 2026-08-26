"""pytest conftest: ensure all extension directories are on sys.path."""

import sys
import pathlib

_base = pathlib.Path(__file__).resolve().parent.parent
_src = _base / 'src'

# Group directories
for _p in [_src, _src / 'core', _src / 'math']:
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

# Trading dir is lowercase 'trading' but tests use `import Trading`.
# Pre-import with correct case and alias.
_trading_init = _src / 'trading' / '__init__.py'
if _trading_init.exists():
    import importlib.util
    _spec = importlib.util.spec_from_file_location('trading', _trading_init)
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules['trading'] = _mod
        sys.modules['Trading'] = _mod
        _spec.loader.exec_module(_mod)
