import time
import numpy as np
from run import moving_average

for n in [1000, 10000, 100000]:
    data = np.random.random(n).astype(np.float64)
    t0 = time.perf_counter()
    for _ in range(10):
        moving_average(data, window=100)
    t = (time.perf_counter() - t0) / 10
    print(f"n={n:>8}: {t*1000:.3f} ms  ({n/t/1e6:.2f} M/s)")
