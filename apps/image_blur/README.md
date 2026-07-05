# Image Box Blur

**Задача:** Размытие изображения (box blur) через матричное умножение.

**Метод:** Разделимая свёртка: два последовательных matmul (строки → столбцы).

**Почему NumFast:** matmul() на GPU — tiled, с shared memory.

**Запуск:**
```bash
cd apps/image_blur
python run.py
```
