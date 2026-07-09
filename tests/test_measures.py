import numpy as np

from plotxy_app.measures import compute_measures

_INC, _DEC, _NM = 0, 1, 2


def test_basic_increasing():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 10.0, 20.0, 10.0, 0.0])
    m = compute_measures(x, y, 1.0, 3.0, _INC)
    assert m["n"] == 3
    assert m["max"] == 20.0 and m["min"] == 10.0
    assert m["max_x"] == 2.0                    # Y peaks at x=2
    assert m["min_x"] == 1.0                    # first of the two 10.0 (x=1)
    assert np.isclose(m["mean"], (10 + 20 + 10) / 3)
    assert m["dx"] == 2.0                      # 3 - 1
    assert m["dy"] == 0.0                      # 10 - 10
    assert np.isclose(m["area"], np.trapezoid(y[1:4], x[1:4]))


def test_argmax_argmin_locations():
    x = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    y = np.array([3.0, -5.0, 8.0, 1.0, 8.0])
    m = compute_measures(x, y, 10.0, 14.0, _INC)
    assert m["max"] == 8.0 and m["max_x"] == 12.0   # first max at x=12
    assert m["min"] == -5.0 and m["min_x"] == 11.0
    # decreasing X: locations still in original coordinates
    xd = x[::-1].copy()
    yd = np.array([8.0, 1.0, 8.0, -5.0, 3.0])       # mirror of y
    md = compute_measures(xd, yd, 10.0, 14.0, _DEC)
    assert md["min"] == -5.0 and md["min_x"] == 11.0


def test_area_known_ramp():
    # y = x over [0, 1] -> area 0.5
    x = np.linspace(0.0, 1.0, 101)
    m = compute_measures(x, x, 0.0, 1.0, _INC)
    assert np.isclose(m["area"], 0.5)


def test_decreasing_x():
    x = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    y = np.array([40.0, 30.0, 20.0, 10.0, 0.0])
    m = compute_measures(x, y, 1.0, 3.0, _DEC)
    assert m["n"] == 3
    assert m["max"] == 30.0 and m["min"] == 10.0
    # samples in original order: x=3,2,1 -> dx = 1-3 = -2
    assert m["dx"] == -2.0
    assert m["dy"] == -20.0


def test_non_monotonic_mask():
    x = np.array([0.0, 2.0, 1.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    m = compute_measures(x, y, 0.5, 2.5, _NM)
    # samples with x in range: (2,1) and (1,2)
    assert m["n"] == 2
    assert m["max"] == 2.0 and m["min"] == 1.0


def test_nan_pairs_dropped():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, np.nan, 20.0, 30.0])
    m = compute_measures(x, y, 0.0, 3.0, _INC)
    assert m["n"] == 3
    assert m["min"] == 0.0 and m["max"] == 30.0


def test_empty_interval():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    assert compute_measures(x, y, 5.0, 6.0, _INC) is None


def test_swapped_bounds():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    m = compute_measures(x, y, 2.0, 0.0, _INC)  # lo > hi
    assert m is not None and m["n"] == 3


def test_single_sample():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([5.0, 7.0, 9.0])
    m = compute_measures(x, y, 0.9, 1.1, _INC)
    assert m["n"] == 1
    assert m["max"] == m["min"] == m["mean"] == 7.0
    assert m["dx"] == 0.0 and m["dy"] == 0.0 and m["area"] == 0.0


def test_rms():
    # sine over whole periods: RMS = A / sqrt(2)
    a = 5.0
    x = np.linspace(0.0, 1.0, 10001)          # 1 s
    y = a * np.sin(2 * np.pi * 10 * x)        # 10 whole periods
    m = compute_measures(x, y, 0.0, 1.0, _INC)
    assert np.isclose(m["rms"], a / np.sqrt(2), rtol=1e-3)
    # constant: RMS = |c|
    c = np.full(50, -3.0)
    m = compute_measures(np.arange(50.0), c, 0.0, 49.0, _INC)
    assert np.isclose(m["rms"], 3.0)
    # partial interval uses only the sliced samples
    x2 = np.array([0.0, 1.0, 2.0, 3.0])
    y2 = np.array([1.0, 2.0, 100.0, 100.0])
    m = compute_measures(x2, y2, 0.0, 1.0, _INC)
    assert np.isclose(m["rms"], np.sqrt((1.0 + 4.0) / 2))
