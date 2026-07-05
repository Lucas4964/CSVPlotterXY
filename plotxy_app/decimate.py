"""View-aware min/max decimation for rendering large series (pure numpy).

Only the *rendered* representation of a curve is decimated; every
computation in the app (cursors, crossings, clicks, measures,
expressions) keeps operating on the full arrays. Min/max decimation per
pixel column preserves the exact visual envelope of the full polyline,
so the rasterized result is indistinguishable from drawing every point.

Guarantees:
- output always contains the first and last points of the slice (a curve
  can never vanish, which was the root cause of the old pyqtgraph
  auto-downsampling bug);
- the global min and max of the slice always survive (autoRange yields
  the same bounds as with full data);
- all-NaN stretches keep emitting NaN, so connect="finite" gaps survive.
"""

from __future__ import annotations

import numpy as np

# below this many visible points the raw slice is uploaded unchanged
# (pixel-exact, preserves every NaN gap) — decimation only kicks in when
# it actually reduces the workload
RAW_LIMIT = 4000


def decimate_minmax(x: np.ndarray, y: np.ndarray, i0: int, i1: int,
                    buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """Decimate the slice [i0, i1) of (x, y) — x must be non-decreasing —
    to at most ~2*buckets + 4 points using per-bucket min/max picks."""
    n = i1 - i0
    if n <= max(RAW_LIMIT, 2 * buckets):
        return x[i0:i1], y[i0:i1]

    stride = -(-n // buckets)          # ceil division
    m = n // stride                    # number of full buckets
    body = i0 + m * stride

    ys = np.ascontiguousarray(y[i0:body]).reshape(m, stride)
    finite = np.isfinite(ys)
    lo = np.where(finite, ys, np.inf)
    hi = np.where(finite, ys, -np.inf)
    imin = np.argmin(lo, axis=1)
    imax = np.argmax(hi, axis=1)
    # emit both picks in index order so the polyline stays x-monotonic
    first = np.minimum(imin, imax)
    second = np.maximum(imin, imax)
    base = i0 + np.arange(m, dtype=np.int64) * stride
    idx = np.empty(2 * m + 2, dtype=np.int64)
    idx[0] = i0                        # slice endpoints always included
    idx[1:-1:2] = base + first
    idx[2:-1:2] = base + second
    idx[-1] = i1 - 1

    xd = x[idx]
    yd = y[idx]
    # note: an all-NaN bucket picks index 0 of the bucket, whose original
    # y is NaN — the gap is emitted naturally, no special-casing needed

    tail = i1 - body                   # partial last bucket (< stride)
    if tail > 2:
        seg = y[body:i1 - 1]
        fin = np.isfinite(seg)
        if fin.any():
            jmin = int(np.argmin(np.where(fin, seg, np.inf)))
            jmax = int(np.argmax(np.where(fin, seg, -np.inf)))
            j0, j1 = sorted((body + jmin, body + jmax))
            xd = np.concatenate((xd[:-1], x[[j0, j1]], xd[-1:]))
            yd = np.concatenate((yd[:-1], y[[j0, j1]], yd[-1:]))
    return xd, yd
