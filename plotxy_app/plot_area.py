"""Plot widget: curves, legend, draggable cursor, intersection markers
and a local zoom region with a synchronized magnified panel."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget

from .themes import Theme

pg.setConfigOptions(antialias=True)

_INCREASING, _DECREASING, _NON_MONOTONIC = 0, 1, 2


class PlotArea(QWidget):
    """Interactive plot working purely on arrays (no dataset knowledge).

    set_series receives the X array plus (key, label, y) tuples already
    paired/truncated by the Project. cursor_moved emits
    (x, [(label, color, y)], nearest_mode).
    """

    cursor_moved = Signal(float, list, bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._x: np.ndarray | None = None
        self._x_key = ""
        self._x_kind = _INCREASING
        self._x_sorted: np.ndarray | None = None
        self._ys: dict[str, np.ndarray] = {}
        self._labels: dict[str, str] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._zoom_curves: dict[str, pg.PlotDataItem] = {}
        self._color_of: dict[str, str] = {}
        self._theme: Theme | None = None
        self._marker_x = 0.0
        self._syncing = False  # guard: region <-> zoom-panel feedback loop

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(self._splitter)

        # main plot
        self._pw = pg.PlotWidget()
        self._splitter.addWidget(self._pw)
        self._plot_item = self._pw.getPlotItem()
        self._plot_item.showGrid(x=True, y=True, alpha=0.15)
        self._legend = self._plot_item.addLegend(offset=(10, 10))

        self._cursor = pg.InfiniteLine(angle=90, movable=True)
        self._cursor.setZValue(90)
        self._cursor.sigPositionChanged.connect(self._on_cursor_moved)
        self._pw.addItem(self._cursor)
        self._cursor.hide()

        self._markers = pg.ScatterPlotItem(size=9, pxMode=True)
        self._markers.setZValue(100)
        self._pw.addItem(self._markers)

        # zoom region on the main plot
        self._region = pg.LinearRegionItem(movable=True)
        self._region.setZValue(80)
        self._region.sigRegionChanged.connect(self._on_region_changed)
        self._pw.addItem(self._region)
        self._region.hide()

        # zoom panel (hidden by default)
        self._zoom_pw = pg.PlotWidget()
        self._zoom_item = self._zoom_pw.getPlotItem()
        self._zoom_item.showGrid(x=True, y=True, alpha=0.15)
        zoom_vb = self._zoom_item.getViewBox()
        zoom_vb.setMouseEnabled(x=True, y=False)
        zoom_vb.enableAutoRange(y=True)
        zoom_vb.setAutoVisible(y=True)
        zoom_vb.sigXRangeChanged.connect(self._on_zoom_range_changed)
        self._splitter.addWidget(self._zoom_pw)
        self._zoom_pw.hide()
        self._splitter.setSizes([3000, 1000])

    # ------------------------------------------------------------- public

    def set_series(self, x_key: str, x_label: str, x: np.ndarray | None,
                   series: list[tuple[str, str, np.ndarray]]) -> None:
        if x is None or len(x) == 0:
            self.clear()
            return
        x_changed = x_key != self._x_key
        # Arrays may shrink/grow when a shorter/longer series joins or
        # leaves the selection (truncation), so refresh even same-key X.
        data_changed = (self._x is None or len(x) != len(self._x))
        self._set_x(x_key, x_label, x)

        wanted = {key for key, _, _ in series}
        for key in list(self._curves):
            if key not in wanted or x_changed or data_changed:
                self._remove_curve(key)

        palette = self._theme.curve_palette if self._theme else ("#1f77b4",)
        for key, label, y in series:
            self._ys[key] = y
            if key in self._curves:
                if self._labels.get(key) != label:
                    # label changed (e.g. became qualified) -> recreate
                    self._remove_curve(key)
                else:
                    # data may have changed (edited custom series)
                    self._curves[key].setData(x, y, connect="finite")
                    self._zoom_curves[key].setData(x, y, connect="finite")
                    continue
            self._labels[key] = label
            color = palette[len(self._color_of) % len(palette)]
            self._color_of[key] = color
            pen = pg.mkPen(color, width=2)
            curve = self._pw.plot(x, y, pen=pen, name=label, connect="finite")
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
            self._curves[key] = curve
            zcurve = self._zoom_pw.plot(x, y, pen=pen, connect="finite")
            zcurve.setDownsampling(auto=True, method="peak")
            self._zoom_curves[key] = zcurve
        for key in list(self._ys):
            if key not in wanted:
                del self._ys[key]

        if x_changed:
            xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
            self._cursor.setBounds((xmin, xmax))
            self._cursor.setValue((xmin + xmax) / 2)
            self._reset_region()
            self._pw.autoRange()

        has_curves = bool(self._curves)
        self._cursor.setVisible(has_curves)
        self._markers.setVisible(has_curves)
        self._on_cursor_moved()

    def clear(self) -> None:
        for key in list(self._curves):
            self._remove_curve(key)
        self._ys.clear()
        self._x = None
        self._x_key = ""
        self._cursor.hide()
        self._markers.hide()
        self._markers.setData([])
        self.cursor_moved.emit(float("nan"), [], False)

    def autorange(self) -> None:
        self._pw.autoRange()

    def set_zoom_visible(self, visible: bool) -> None:
        self._zoom_pw.setVisible(visible)
        self._region.setVisible(visible and self._x is not None)
        if visible and self._x is not None:
            self._reset_region()

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        for pw, item in ((self._pw, self._plot_item),
                         (self._zoom_pw, self._zoom_item)):
            pw.setBackground(theme.plot_bg)
            for side in ("left", "bottom"):
                ax = item.getAxis(side)
                ax.setPen(pg.mkPen(theme.axis_color))
                ax.setTextPen(pg.mkPen(theme.axis_color))
            item.showGrid(x=True, y=True, alpha=theme.grid_alpha)
        self._legend.setLabelTextColor(theme.text)
        self._cursor.setPen(pg.mkPen(theme.cursor_color, width=2))
        self._cursor.setHoverPen(pg.mkPen(theme.cursor_color, width=4))

        accent = QColor(theme.accent)
        brush = QColor(accent); brush.setAlpha(40)
        hover = QColor(accent); hover.setAlpha(60)
        self._region.setBrush(brush)
        self._region.setHoverBrush(hover)
        for line in self._region.lines:
            line.setPen(pg.mkPen(theme.accent, width=1))

        for i, key in enumerate(self._curves):
            color = theme.curve_palette[i % len(theme.curve_palette)]
            self._color_of[key] = color
            pen = pg.mkPen(color, width=2)
            self._curves[key].setPen(pen)
            self._zoom_curves[key].setPen(pen)
        self._on_cursor_moved()

    # ------------------------------------------------------------ internal

    def _remove_curve(self, key: str) -> None:
        curve = self._curves.pop(key, None)
        if curve is not None:
            self._legend.removeItem(curve)
            self._pw.removeItem(curve)
        zcurve = self._zoom_curves.pop(key, None)
        if zcurve is not None:
            self._zoom_pw.removeItem(zcurve)
        self._color_of.pop(key, None)
        self._labels.pop(key, None)

    def _set_x(self, x_key: str, x_label: str, x: np.ndarray) -> None:
        self._x_key = x_key
        self._x = x
        d = np.diff(x)
        with np.errstate(invalid="ignore"):
            if np.all(d >= 0):
                self._x_kind = _INCREASING
                self._x_sorted = x
            elif np.all(d <= 0):
                self._x_kind = _DECREASING
                self._x_sorted = x[::-1]
            else:
                self._x_kind = _NON_MONOTONIC
                self._x_sorted = None
        self._plot_item.setLabel("bottom", x_label)
        self._zoom_item.setLabel("bottom", x_label)

    def _reset_region(self) -> None:
        """Place the region over the middle third of the data range and
        sync the zoom panel."""
        if self._x is None:
            return
        xmin, xmax = float(np.nanmin(self._x)), float(np.nanmax(self._x))
        span = xmax - xmin
        self._region.setRegion((xmin + span / 3, xmin + 2 * span / 3))

    def _on_region_changed(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            lo, hi = self._region.getRegion()
            self._zoom_item.getViewBox().setXRange(lo, hi, padding=0)
        finally:
            self._syncing = False

    def _on_zoom_range_changed(self, _vb, rng) -> None:
        if self._syncing or not self._region.isVisible():
            return
        self._syncing = True
        try:
            self._region.setRegion(rng)
        finally:
            self._syncing = False

    def _on_cursor_moved(self) -> None:
        if self._x is None or not self._curves:
            self._markers.setData([])
            self.cursor_moved.emit(float("nan"), [], False)
            return
        cx = float(self._cursor.value())
        keys = list(self._curves)
        ys = self._values_at(cx, keys)

        spots, rows = [], []
        for key, y in zip(keys, ys):
            color = self._color_of[key]
            rows.append((self._labels[key], color, y))
            if np.isfinite(y):
                spots.append({
                    "pos": (self._marker_x if self._x_kind == _NON_MONOTONIC else cx, y),
                    "brush": pg.mkBrush(color),
                    "pen": pg.mkPen("#ffffff" if self._theme and self._theme.name == "dark"
                                    else "#000000", width=1),
                })
        self._markers.setData(spots)
        self.cursor_moved.emit(cx, rows, self._x_kind == _NON_MONOTONIC)

    def _values_at(self, cx: float, keys: list[str]) -> list[float]:
        """Value of each series at cursor X. Monotonic X: one searchsorted
        + linear interpolation shared by all series. Non-monotonic X:
        nearest sample."""
        x = self._x
        n = len(x)
        self._marker_x = cx
        if n == 0:
            return [float("nan")] * len(keys)

        if self._x_kind == _NON_MONOTONIC:
            with np.errstate(invalid="ignore"):
                idx = int(np.nanargmin(np.abs(x - cx)))
            self._marker_x = float(x[idx])
            return [float(self._ys[key][idx]) for key in keys]

        i = int(np.searchsorted(self._x_sorted, cx))
        i = max(1, min(i, n - 1))
        if self._x_kind == _DECREASING:
            i0, i1 = n - 1 - i, n - i
            x0, x1 = float(x[i1]), float(x[i0])
            j0, j1 = i1, i0
        else:
            x0, x1 = float(x[i - 1]), float(x[i])
            j0, j1 = i - 1, i
        frac = 0.0 if x1 == x0 else (cx - x0) / (x1 - x0)
        frac = min(1.0, max(0.0, frac))

        out = []
        for key in keys:
            col = self._ys[key]
            y0, y1 = float(col[j0]), float(col[j1])
            out.append(y0 + (y1 - y0) * frac)
        return out
