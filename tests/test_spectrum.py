"""compute_spectrum: pure single-sided amplitude spectrum (headless)."""

import numpy as np

from plotxy_app.spectrum import compute_spectrum


def test_sine_peak_amplitude_and_frequency():
    fs, f0, n = 1000.0, 50.0, 2000
    t = np.arange(n) / fs
    y = 3.0 * np.sin(2 * np.pi * f0 * t) + 7.0   # amplitude 3, DC 7
    spec = compute_spectrum(t, y)
    assert spec is not None and spec["uniform"]
    i = int(np.argmax(spec["amp"]))
    assert abs(spec["freq"][i] - f0) <= fs / n + 1e-9   # peak at 50 Hz
    assert abs(spec["amp"][i] - 3.0) < 0.05             # amplitude ~3
    assert spec["amp"][0] == 0.0                        # DC removed
    assert abs(spec["freq"][-1] - fs / 2) < 1e-9        # up to Nyquist


def test_two_tones():
    fs, n = 500.0, 1000
    t = np.arange(n) / fs
    y = 1.0 * np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)
    spec = compute_spectrum(t, y)
    freq, amp = spec["freq"], spec["amp"]
    i10 = int(np.argmin(np.abs(freq - 10.0)))
    i60 = int(np.argmin(np.abs(freq - 60.0)))
    assert abs(amp[i10] - 1.0) < 0.05
    assert abs(amp[i60] - 0.5) < 0.05


def test_non_uniform_flag_and_nan_drop():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0.0, 1.0, 512))
    spec = compute_spectrum(t, np.sin(2 * np.pi * 5 * t))
    assert spec is not None and not spec["uniform"]

    t2 = np.linspace(0.0, 1.0, 512)
    y2 = np.sin(2 * np.pi * 5 * t2)
    y2[10] = np.nan
    spec2 = compute_spectrum(t2, y2)
    assert spec2 is not None and spec2["n"] == 511


def test_degenerate_inputs():
    assert compute_spectrum(np.arange(4.0), np.arange(4.0)) is None  # short
    assert compute_spectrum(np.zeros(10), np.ones(10)) is None       # dt = 0
    nan = np.full(20, np.nan)
    assert compute_spectrum(nan, nan) is None
