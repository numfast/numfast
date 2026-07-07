# Moving Average (SMA)

**NumFast operations:** `scan()`

**Difficulty:** ★☆☆☆☆

**Speedup:** CPU O(n) · GPU O(log n)

---

**Задача:** Вычислить простое скользящее среднее (SMA) для временного ряда.

**Метод:** Prefix sum (cumulative sum) → разность окон.

**Почему NumFast:** `scan()` выполняется за O(log n) на GPU через параллельный
префиксный сумматор — быстрее последовательного O(n) на CPU для больших массивов.

**Запуск:**
```bash
cd apps/moving_average
python run.py
python benchmark.py
```

**Ожидаемый результат:** SMA для ценового ряда, совпадающий с numpy.
