import time
import numpy as np
from numfast import fft

for n in [1024, 4096, 16384]:
    data = np.random.random(n).astype(np.float64)
    t0 = time.perf_counter()
    for _ in range(50):
        fft(data)
    t = (time.perf_counter() - t0) / 50
    print(f"n={n:>6}: {t*1000:.3f} ms  ({n/t/1e6:.2f} M/s)")
