# examples/

Runnable examples.

Run them from the repository root with
`PYTHONPATH=<repo>/src;<path-to-app-builder>` (the app-builder checkout, if you
have one, is what supplies `builder`):

```bash
export PYTHONPATH="<repo>/src;<path-to-app-builder>"
python examples/quickstart.py
```

`quickstart.py` additionally works with `PYTHONPATH=<repo>/src` alone and
nothing else — it is the example that must run on a clean clone.

| File | What it is | State |
|---|---|---|
| [`quickstart.py`](quickstart.py) | the smallest NumFast program that produces a real result: group, sort, and assert agreement with pandas | **runs clean — start here** |

That is the directory's whole contents. The development microbenchmarks that
used to sit here are gone; none of them was quotable, and three of them could not
even run as committed.

Rule of the directory: new demos go here, not into the repository root and not
into `src/`. A file whose numbers may be quoted belongs in an organisation
benchmark repository, published alongside its input data and the command that
produced it — not here, where neither can travel.
