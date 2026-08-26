import numpy as np
from run import moving_average

def expected(data, window=3):
    return np.convolve(data, np.ones(window)/window, 'same')

if __name__ == "__main__":
    data = np.array([1., 2., 3., 4., 5.])
    result = moving_average(data, window=3)
    ref = expected(data, window=3)
    print(f"Result: {result.round(4)}")
    print(f"Ref:    {ref.round(4)}")
    print(f"Match: {np.allclose(result, ref, atol=1e-10)}")
