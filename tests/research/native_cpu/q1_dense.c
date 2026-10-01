/* Q1 direct dense fused aggregate (native CPU prototype, research only).
 *
 * ONE primitive (STEP 3): fused sum+count over resident int32 codes.
 * Native API: contiguous typed buffers in -> canonical arrays out.
 * Caller allocates + owns ALL outputs (no malloc/free inside, no ownership
 * questions); caller guarantees 0 <= codes[i] < m (same Planner-gate trust
 * as dense_by_code_present). int32 in, int64 accumulators (dtype discipline).
 * Python API / IR / Planner untouched; loaded via ctypes (no new runtime dep).
 *
 * Build (Windows, zig cc from pip `ziglang`, no MSVC needed):
 *   python-zig cc -shared -O3 -o q1_dense.dll q1_dense.c
 */
#include <stdint.h>

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API
#endif

API void q1_dense_fused(const int32_t *codes, const int32_t *vals, int64_t n,
                        int32_t m, int64_t *sums /*[m]*/, int64_t *counts /*[m]*/) {
    (void)m; /* range guaranteed by caller (Planner gate) */
    for (int64_t i = 0; i < n; i++) {
        int32_t k = codes[i];
        sums[k] += (int64_t)vals[i];
        counts[k] += 1;
    }
}
