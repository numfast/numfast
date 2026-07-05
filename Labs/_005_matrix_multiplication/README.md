# Lab 005 — Matrix Multiplication (Tiled)

## Что это

Умножение матриц C = A × B, реализованное как tiled GPU-алгоритм
с использованием shared memory.

## Почему это важно

MatMul — центральный алгоритм линейной алгебры и нейросетей.
Tiled-реализация демонстрирует ключевые техники GPU-программирования:

- 2D диспетчеризация workgroup
- Tile-загрузка в shared memory (кооперативный доступ)
- Переиспользование данных из shared memory (уменьшение global reads)
- Граничные проверки для некратных размеров

## Архитектура

- Workgroup: 16×16 = 256 threads
- Tile: 16×16 элементов A и B в shared memory
- Каждый thread вычисляет один элемент C
- Внешний цикл по K с шагом 16
- Два `workgroupBarrier` на итерацию (загрузка + вычисление)

## Формат данных

Матрицы хранятся в row-major flatten:
```
A[i][k] → A[i * K + k]
B[k][j] → B[k * N + j]
C[i][j] → C[i * N + j]
```

## Запуск

```bash
cd numfast
python -m Labs._005_matrix_multiplication.jobs
```

## Сравнение с эталоном

```bash
python -m pytest Labs/_005_matrix_multiplication/golden/
```

## Точность

- CPU: double precision (0 ошибка относительно numpy)
- GPU: float32 (ошибка ~1e-6 для типичных размеров)
