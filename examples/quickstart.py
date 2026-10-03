# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""quickstart.py -- the smallest NumFast program that produces a real result.

    export PYTHONPATH="C:/App/numfast/numfast/src;C:/App/numfast/app-builder"
    python examples/quickstart.py

Group orders by region, sort by revenue descending, and check the answer
against pandas. It prints both tables and asserts they agree, so a run that
succeeds has proved three things at once: the engine works, the numbers are
right, and the consumer API is the shape you would actually write.

The committed CI job `.github/workflows/ci.yml` does not run this file; the
quickstart in README.md is the same program inline.
"""

import sys

import pandas as pd

import numfast as nf


def main():
    orders = pd.DataFrame({
        "region":  ["emea", "apac", "emea", "amer", "apac", "emea"],
        "revenue": [120.0, 80.0, 240.5, 60.0, 95.5, 310.0],
    })

    # from_pandas -> query() -> group() -> sort() -> compile()
    result = (nf.from_pandas(orders)
              .query()
              .group("region", {"revenue": ("sum", "count")})
              .sort("revenue.sum", desc=True)
              .compile())

    got = result.to_pandas()

    want = (orders.groupby("region", as_index=False)["revenue"]
            .agg(["sum", "count"])
            .sort_values("sum", ascending=False)
            .rename(columns={"sum": "revenue.sum", "count": "revenue.count"}))

    print("NumFast:")
    print(got.to_string(index=False))
    print()
    print("pandas:")
    print(want.to_string(index=False))
    print()

    # dtype labels differ (NumFast is nullable where pandas is not); the
    # values and the order must not. See KNOWN_LIMITATIONS.md.
    pd.testing.assert_frame_equal(
        got[["region", "revenue.sum", "revenue.count"]].reset_index(drop=True),
        want[["region", "revenue.sum", "revenue.count"]].reset_index(drop=True),
        check_dtype=False)
    print("OK: NumFast and pandas agree on values and order.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()