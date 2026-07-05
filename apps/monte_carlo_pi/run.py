"""Monte Carlo estimation of π."""
import numpy as np
from numfast import scan

def estimate_pi(n_points=1000000):
    """Estimate π using Monte Carlo method.
    
    Points in unit square, count those inside unit circle.
    π ≈ 4 × (points inside circle) / (total points)
    
    Uses scan() for cumulative count.
    """
    np.random.seed(42)
    x = np.random.random(n_points).astype(np.float64)
    y = np.random.random(n_points).astype(np.float64)
    
    inside = (x**2 + y**2) <= 1.0
    # Use scan on the binary "inside" array for cumulative count
    cumsum = scan(inside.astype(np.float64))
    total_inside = int(cumsum[-1])
    pi_estimate = 4.0 * total_inside / n_points
    return pi_estimate

if __name__ == "__main__":
    for n in [10000, 100000, 1000000]:
        pi = estimate_pi(n)
        error = abs(pi - np.pi)
        print(f"n={n:>8}: pi ~ {pi:.6f}  (error={error:.2e})")
