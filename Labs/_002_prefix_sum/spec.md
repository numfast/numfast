# Spec: Inclusive Prefix Sum

## Интерфейс

### Jobs

```python
# Single block (≤64 elements)
{"op": "ScanLocal", "inputs": ["data"], "params": {},
 "out": "scan"}

# Multi-block — three-phase chain
{"op": "ScanLocal", "inputs": ["data"], "params": {},
 "out": ["scan_local", "block_sum"]},
{"op": "ScanTotals", "inputs": ["block_sum"], "params": {},
 "out": "block_prefix"},
{"op": "ScanFinal", "inputs": ["data", "scan_local", "block_prefix"], "params": {},
 "out": "scan"}
```

### Inputs / Outputs

#### ScanLocal
- Input: `data (float, N)` — произвольный массив
- Output 0: `scan_local (float, N)` — локальный prefix sum
- Output 1: `block_sum (float, ceil(N/64))` — сумма каждого блока

#### ScanTotals
- Input: `block_sums (float, M)` — массив блочных сумм
- Output: `block_prefix (float, M)` — prefix sum блочных сумм

#### ScanFinal
- Input: `data (float, N)`, `local (float, N)`, `prefix (float, M)`
- Output: `scan (float, N)` — итоговый inclusive prefix sum

## Требования

1. CPU и GPU дают одинаковые результаты
2. cpu: результат совпадает с `np.cumsum(data)` (float64 точность)
3. gpu: относительная ошибка ≤ 0.1% для N ≤ 100000
4. Все выходы имеют корректный размер (output_size_fn)

## Диспетчеризация

- workgroup_size = 64
- dispatch = ceil(N / 64)
- ScanTotals: @workgroup_size(1), dispatch = 1
