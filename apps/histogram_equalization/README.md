# Histogram Equalization

**Задача:** Улучшить контраст изображения через выравнивание гистограммы.

**Метод:** histogram() + scan() → CDF → пиксельная карта.

**Почему NumFast:** histogram() использует GPU-атомики, scan() — параллельный
prefix sum. Оба работают за O(n).

**Запуск:**
```bash
cd apps/histogram_equalization
python run.py
```

**Ожидаемый результат:** Контраст изображения увеличен, гистограмма равномерна.
