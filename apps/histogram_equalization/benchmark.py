import time
import numpy as np
from run import equalize

for n in [128, 256, 512]:
    img = np.random.random((n, n)).astype(np.float64)
    t0 = time.perf_counter()
    for _ in range(10):
        equalize(img)
    t = (time.perf_counter() - t0) / 10
    pixels = n * n
    print(f"n={n:>4} ({pixels:>8.0f} px): {t*1000:.3f} ms  ({pixels/t/1e6:.2f} Mpx/s)")
