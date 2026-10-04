# examples/

Runnable examples and microbenchmarks that live outside the engine.

Run them from this directory with `PYTHONPATH=<repo>/src;<repo>/..`
(the app-builder checkout, if you have one, is what supplies `builder`):

```bash
export PYTHONPATH="<repo>/src;<path-to-app-builder>"
python examples/quickstart.py
```

`quickstart.py` additionally works with `PYTHONPATH=<repo>/src` alone and
nothing else — it is the example that must run on a clean clone.

The state column below was **measured** by running each file in this directory
against the current tree (Windows, Python 3.14.6).

| File | What it is | State |
|---|---|---|
| [`quickstart.py`](quickstart.py) | the smallest NumFast program that produces a real result: group, sort, and assert agreement with pandas | **runs clean — start here** |
| `bench_composite.py` | microbenchmark of the composite tuple index, before/after, seed 42 | runs clean |
| `bench_stages.py` | stage-by-stage breakdown of the same benchmark | runs clean |
| `bench_unique.py` | `np.unique` on a structured array vs lexsort + new_group | runs clean |
| `bench_lexsort.py` | packing 4×int32 into 2×int64 for a faster lexsort | runs clean, but has no fixed seed, so its numbers are not quotable |
| `bench_newgroup.py` | stacked `np.any` vs a loop for `new_group` | runs clean, no seed contract |
| `bench_ukeys.py` | alternative unique-key construction strategies | runs clean, no seed contract |
| `bench_verify.py` | one-off check that a packed lexsort produces the original order | runs clean; a one-off check, not a benchmark |
| `bench_segmented.py` | segmented reduce / adjacency slice against a naive chain, seed 42 | **fails its own assertion**: the f32 sum differs from the naive chain by 0.035 and the script requires 0. The int32 lane matches exactly (max_diff 0.0). |
| `bench_join_i64.py` | benchmarks and validates the int64 join lane (the int32 lane is untouched); writes `bench_join_i64_results.json` | **cannot run as committed**: it resolves `src/Relational/Join/_lib/native_i64.py` relative to `examples/`, i.e. to `examples/src/…`, which does not exist |
| `_measure_text_encode.py` | phase breakdown for text encoding with native dedup detection | cannot run: hardcodes `C:/App` paths and needs an external parquet file |

Nothing in the README quotes a number from any file in this directory, and
[BENCHMARKS.md](../BENCHMARKS.md) does not list any: these are development
microbenchmarks, not the record of a measurement anyone may rely on.

Rule of the directory: new demos and benchmarks go here, not into the repository
root and not into `src/`. A benchmark whose numbers may be quoted must state its
seed and must be listed in `BENCHMARKS.md` with the command that produces it.