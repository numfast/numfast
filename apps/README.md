# NumFast Applications

> Every example in this directory is executable.
> It is both documentation and an integration test.

| Example                | Operations       | Difficulty |  Status  | Description                     |
| ---------------------- | ---------------- | :--------: | :------: | ------------------------------- |
| [Moving Average](moving_average/) | Scan | ★ | ✅ Stable | Fast SMA using prefix sums |
| [Histogram Equalization](histogram_equalization/) | Histogram + Scan | ★★ | ✅ Stable | Image contrast enhancement |
| [FFT Audio](fft_audio/) | FFT | ★★ | ✅ Stable | Frequency analysis of audio |
| [Image Blur](applications/image_blur/) | MatMul | ★★★ | ✅ Stable | Box blur using tiled matrix ops |
| [Linear Regression](applications/linear_regression/) | MatMul | ★★★ | ✅ Stable | Least squares fitting |
| [Sobel Edge](applications/sobel_edge/) | MatMul | ★★★ | ✅ Stable | Edge detection |

---

Run any example:

```bash
python run.py
```

Verify:

```bash
python expected.py
```

Benchmark:

```bash
python benchmark.py
```

All examples are intentionally minimal.

Their goal is to demonstrate one NumFast concept at a time.
