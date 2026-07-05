# FFT Audio Analysis

**Задача:** Найти доминирующие частоты в аудиосигнале.

**Метод:** FFT → magnitude spectrum → top-k peaks.

**Почему NumFast:** FFT на GPU работает за O(n log n) с параллельными
butterfly-стадиями.

**Запуск:**
```bash
cd apps/fft_audio
python run.py
```

**Ожидаемый результат:** Определение частот 440 Гц и 880 Гц.
