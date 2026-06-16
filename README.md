# NumFast — Experimental GPU-oriented Columnar Data Engine

NumFast is an experimental columnar data engine where every table is physically stored as a tightly packed Row-Struct container in VRAM. Columns are laid out as contiguous bit sequences — 11, 14, 17 bits, whatever fits — without padding or per-column arrays. JIT WGSL compute shaders extract and reduce data directly from the packed representation, no host-side decompression required.

The public API exposes `NumericSeries` and `Tables` interfaces backed by lightweight proxy dicts carrying only metadata.

## Project structure

```
numfast/
  _core/          — kernel, layout engine, visualizer, compression, series types
  Tables/         — table creation, column proxy, dynamic compression
  Series/         — series extension (Builder)
  Stats/          — statistical functions (total, mean, min, max, var)
  tests/          — pytest suite
```

## Series Type System

| Тип | Статус | Описание |
|-----|--------|----------|
| `NumericSeries` | ✅ Реализовано | Числовые данные, bit packing, GPU-статистика |
| `ObjectSeries` | 🚧 Заглушка | Python-объекты, произвольные структуры |
| `TextSeries` | 🚧 Заглушка | Строки, токены, NLP |
| `ImageSeries` | 🚧 Заглушка | Изображения (JPEG, PNG, raw) |
| `TensorSeries` | 🚧 Заглушка | Многомерные тензоры для AI/ML |

## Quick start

```python
from _core.context import create_context
from _core.series import NumericSeries
from Tables._lib.tables_lib import _container_table_create, _get_column
from Stats.Stats import total, mean

ctx = create_context("demo")

# Numeric series
s = NumericSeries([1.0, 2.0, 3.0, 4.0, 5.0], ctx)
s.info()                  # dual-mode dashboard

# Table with adaptive bit packing
tbl = _container_table_create(
    [{"name": "price", "dtype": "float32",
      "compression": {"scaled": True, "bits": 12, "scale": 0.1, "offset": 50.0}},
     {"name": "volume", "dtype": "float32"}],
    {"price": [150.0, 200.0, 175.0],
     "volume": [1000, 2000, 1500]},
)
tbl.info()

col = _get_column(tbl, "price")
print(total(col), mean(col))
```

## Requirements

- Python ≥ 3.11
- Windows (WebGPU via `wgpu-py`) or CPU fallback
- NumPy (CPU fallback)

## Development

```bash
pytest              # run all tests
pytest -x           # stop on first failure
```

## License

AGPL-3.0-only
