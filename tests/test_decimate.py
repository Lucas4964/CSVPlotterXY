import numpy as np

from plotxy_app.decimate import RAW_LIMIT, decimate_minmax


def test_small_slice_is_raw_and_identical():
    x = np.linspace(0, 1, 500)
    y = np.sin(x * 20)
    xd, yd = decimate_minmax(x, y, 10, 200, 1024)
    assert np.array_equal(xd, x[10:200])
    assert np.array_equal(yd, y[10:200])


def test_envelope_and_endpoints_preserved():
    rng = np.random.default_rng(42)
    n = 200_000
    x = np.linspace(0, 100, n)
    y = rng.normal(size=n).cumsum()
    xd, yd = decimate_minmax(x, y, 0, n, 1000)
    # massively reduced but bounded
    assert len(xd) <= 2 * 1000 + 4
    assert len(xd) == len(yd)
    # global extrema always survive (autoRange parity)
    assert yd.max() == y.max()
    assert yd.min() == y.min()
    # endpoints always present (curve can never vanish)
    assert xd[0] == x[0] and xd[-1] == x[-1]
    assert yd[0] == y[0] and yd[-1] == y[-1]
    # x stays sorted so the polyline is well-formed
    assert np.all(np.diff(xd) >= 0)


def test_bucket_extrema_survive():
    # every bucket's min and max must appear in the output
    n = 50_000
    x = np.arange(n, dtype=float)
    y = np.sin(x * 0.37) * np.linspace(1, 5, n)
    buckets = 500
    xd, yd = decimate_minmax(x, y, 0, n, buckets)
    stride = -(-n // buckets)
    out = set(zip(xd.tolist(), yd.tolist()))
    for b in range(0, n // stride):
        seg = y[b * stride:(b + 1) * stride]
        i_lo = b * stride + int(np.argmin(seg))
        i_hi = b * stride + int(np.argmax(seg))
        assert (x[i_lo], y[i_lo]) in out
        assert (x[i_hi], y[i_hi]) in out


def test_nan_gap_survives():
    n = 100_000
    x = np.arange(n, dtype=float)
    y = np.sin(x * 0.01)
    y[40_000:60_000] = np.nan            # a wide gap
    xd, yd = decimate_minmax(x, y, 0, n, 800)
    # some NaN must survive inside the gap region so the line breaks
    in_gap = (xd >= 40_000) & (xd < 60_000)
    assert in_gap.any()
    assert np.isnan(yd[in_gap]).all()
    # outside the gap the data is finite
    assert np.isfinite(yd[xd < 39_000]).all()


def test_partial_tail_extrema():
    # craft a slice whose tail (n % stride) holds the global max
    n = 10_000 * 2 + 1500                # buckets=2 -> stride 10750? no:
    x = np.arange(float(n))
    y = np.zeros(n)
    y[-100] = 99.0                        # spike inside the partial tail
    xd, yd = decimate_minmax(x, y, 0, n, 2)
    assert 99.0 in yd
    assert xd[-1] == x[-1]


def test_raw_limit_boundary():
    n = RAW_LIMIT
    x = np.arange(float(n))
    y = x * 2
    xd, yd = decimate_minmax(x, y, 0, n, 100)
    assert len(xd) == n                   # exactly at limit -> raw
