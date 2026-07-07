# FFT Audio Analysis

**NumFast operations:** `fft()`

**Difficulty:** ★★☆☆☆

**Speedup:** CPU O(n log n) · GPU O(n log n) with parallel butterflies

---

**Задача:** Найти доминирующие частоты в аудиосигнале.

**Метод:** FFT → magnitude spectrum → top-k peaks.

**Ограничение:** Только степени двойки (n = 1024, 2048, 8192...).

**Запуск:**
```bash
cd apps/fft_audio
python run.py
```

**Ожидаемый результат:** Определение частот 440 Гц (A4) и 880 Гц (A5).
