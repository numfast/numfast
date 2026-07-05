"""Kernel Conformance Suite.

Обязательный набор тестов, который должен проходить ЛЮБОЙ драйвер.
При появлении нового драйвера (CUDA, Metal, Vulkan):
  1. Подключить driver к Runtime
  2. Запустить conformance suite
  3. Все тесты должны пройти

Покрытие:
  Core:  Map, Reduce
  Core:  Scan (future)
  Core:  Sort (future)
  LA:    MatMul (future)
  Signal: FFT, Conv (future)
"""
