import time
import numpy as np
from run import kmeans

for n in [100, 500, 2000]:
    data = np.random.random((n, 2)).astype(np.float64)
    t0 = time.perf_counter()
    kmeans(data, k=5)
    t = time.perf_counter() - t0
    print(f"n={n:>5}: {t*1000:.3f} ms  ({n/t/1e6:.2f} M pts/s)")
