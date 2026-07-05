# Moving Average (SMA)

**Задача:** Вычислить простое скользящее среднее (SMA) для временного ряда.

**Метод:** Prefix sum (cumulative sum) → разность окон.

**Почему NumFast:** `scan()` выполняется за O(n) на GPU, что даёт
ускорение на больших окнах.

**Запуск:**
```bash
cd apps/moving_average
python run.py
python benchmark.py
```

**Ожидаемый результат:** SMA для ценового ряда, совпадающий с numpy.
