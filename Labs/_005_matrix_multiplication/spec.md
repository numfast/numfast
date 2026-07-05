# Spec: Tiled Matrix Multiplication

## Интерфейс

### Job

```python
{"op": "MatMul", "inputs": ["A", "B"],
 "params": {"M": 64, "N": 64, "K": 64},
 "out": "C"}
```

### Inputs / Outputs

- Input A: float array, M×K элементов (row-major)
- Input B: float array, K×N элементов (row-major)
- Output C: float array, M×N элементов (row-major)
- Uniforms: M, N, K (int)

## Диспетчеризация

- Workgroup size: (16, 16, 1)
- Dispatch: (ceil(N/16), ceil(M/16), 1)
- TILE = 16

## Требования

1. CPU: результат совпадает с numpy @ (double precision)
2. GPU: ошибка ≤ 1e-4 для размеров ≤ 1024
3. Граничные условия: некратные 16 размеры корректны
