"""Expression — JIT WGSL из формулы.

Expression принимает формулу вида "(A-B)/(C+1.0)" и динамически генерирует
WGSL-шейдер. Переменные: A-Z (одна буква, заглавная).
Поддерживаемые функции: abs, max, min.
Операторы: + - * / (со стандартным приоритетом).
Числа: целые автоматически конвертируются в float (1 -> 1.0).

GPU: WGSL генерируется на лету по формуле (JIT), кешируется по хешу.
CPU: Python eval с ограниченным namespace (только abs, max, min).
"""

from .descriptor import describe
from .cpu import cpu
from .wgsl import wgsl_generator


