import numpy as np

if __name__ == "__main__":
    from run import estimate_pi
    pi = estimate_pi(100000)
    print(f"pi ~ {pi:.6f}, error={abs(pi - np.pi):.2e}")
