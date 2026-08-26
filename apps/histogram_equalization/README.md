# Histogram Equalization

**NumFast operations:** `histogram()` · `scan()`

**Difficulty:** ★★☆☆☆

**Speedup:** CPU O(n) · GPU O(n) with atomics

---

**Задача:** Улучшить контраст изображения через выравнивание гистограммы.

**Метод:** histogram() → CDF via scan() → пиксельная карта.

**Почему NumFast:** histogram() использует GPU-атомики в shared memory,
scan() — параллельный префиксный сумматор.

**Запуск:**
```bash
cd apps/histogram_equalization
python run.py
```

**Ожидаемый результат:** Контраст изображения увеличен, гистограмма равномерна.
