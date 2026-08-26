"""Executable versions of documentation examples E1-E8 + quickstart blocks."""

import numfast as nf


def test_e1_million_arithmetic():
    s = nf.series(list(range(1_000_000)))
    y = (s * 2 + 1).compute()
    assert len(y) == 1_000_000 and y.data()[:3] == [1.0, 3.0, 5.0]


def test_e2_index_cycle():
    pat = nf.tile([1, 2, 3], 9); assert pat.to_numpy().tolist() == [1, 2, 3] * 3
    i = nf.index(9)
    s = nf.series((i.to_numpy() % 3 + 1).tolist())
    assert s.data() == [1.0, 2.0, 3.0] * 3  # concrete series: .data(), .compute() is expr-only


def test_e3_deterministic_random():
    a = nf.random.uniform(shape=5, seed=42).to_numpy()
    b = nf.random.uniform(shape=5, seed=42).to_numpy()
    assert (a == b).all()


def test_e4_harmonics():
    s = nf.series([0.0, 1.0, 2.0])
    y = (nf.math.sin(s) + 0.5 * nf.math.cos(s * 2)).compute()
    assert abs(y.data()[0] - 0.5) < 1e-6  # sin(0) + 0.5*cos(0) = 0.5


def test_e5_filter():
    s = nf.series([1.0, 5.0, 2.0, 8.0, 3.0])
    assert (s > 4).filter(s).data() == [5.0, 8.0]


def test_e6_topk():
    r = nf.topk(nf.series([5.0, 1.0, 8.0]), 2)
    assert list(r) == [8.0, 5.0]


def test_e7_statistics():
    s = nf.series([1.0, 2.0, 3.0])
    assert nf.mean(s) == 2.0 and nf.total(s) == 6.0 and nf.count(s) == 3


def test_e8_pipeline():
    x = nf.ones(1_000_000)  # zeros -> y=2 -> mask y>2 empty; ones keeps pipeline non-trivial
    s = nf.series(x.to_numpy().tolist())
    y = ((s + 1) * 2).compute(); mask = y > 2; kept = mask.filter(y)
    assert len(kept) == 1_000_000


def test_quickstart_creation():
    g = nf.arange(0, 10, 2); assert g.to_numpy().tolist() == [0, 2, 4, 6, 8]
    l = nf.linspace(0, 1, 5); assert abs(l.to_numpy()[1] - 0.25) < 1e-6
    t = nf.tile([1, 2, 3], 9); r = nf.repeat([1, 2], 4)
    assert len(t) == 9 and len(r) == 8


def test_quickstart_groupby():
    keys = nf.series([0.0, 1.0, 0.0, 1.0]); vals = nf.series([1.0, 3.0, 2.0, 4.0])
    res = nf.groupby(keys.data(), vals.data())  # NumericSeries has .data(), no .to_numpy()
    assert list(res["mean"]) == [1.5, 3.5]


def test_quickstart_random():
    d = nf.random.integers(low=1, high=7, shape=100, seed=42, dtype="int32")
    assert len(d) == 100


def test_profile_ctx():
    s = nf.series([1.0, 2.0])
    with nf.profile(): y = (s * 2 + 1).compute()
    assert y.data() == [3.0, 5.0]
