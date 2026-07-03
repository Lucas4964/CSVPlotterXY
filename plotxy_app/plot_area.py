"""Plot widget: curves, legend, draggable cursor, intersection markers
and a local zoom region with a synchronized magnified panel."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.LegendItem import ItemSample
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QSplitter, QVBoxLayout, QWidget

from .themes import Theme

pg.setConfigOptions(antialias=True)

_INCREASING, _DECREASING, _NON_MONOTONIC = 0, 1, 2


class ClickableSample(ItemSample):
    """Legend sample drawn as a solid color square; clicking it invokes a
    callback (opens a color picker) instead of toggling visibility."""

    def __init__(self, item, key: str, on_click):
        super().__init__(item)
        self._key = key
        self._on_click = on_click

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, 20, 20)

    def paint(self, p, *args) -> None:
        pen = self.item.opts.get("pen")
        color = pg.mkColor(pen.color() if hasattr(pen, "color")
                           else (pen or "#888888"))
        p.setBrush(pg.mkBrush(color))
        p.setPen(pg.mkPen(color.darker(150)))
        p.drawRect(QRectF(3, 3, 14, 14))

    def mouseClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            self._on_click(self._key)


class PlotArea(QWidget):
    """Interactive plot working purely on arrays (no dataset knowledge).

    set_series receives the X array plus (key, label, y) tuples already
    paired/truncated by the Project. cursor_moved emits
    (x, [(label, color, y)], nearest_mode).
    """

    cursor_moved = Signal(float, list, bool)
    view_range_changed = Signal(float, float, float, float)

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
        self._color_override: dict[str, str] = {}  # user-picked colors, per key
        self._legend_samples: dict[str, ClickableSample] = {}
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

        # click tooltip (Desmos-style): single point, dismissable
        self._click_marker = pg.ScatterPlotItem(size=11, pxMode=True)
        self._click_marker.setZValue(110)
        self._pw.addItem(self._click_marker)
        self._click_label = pg.TextItem(anchor=(0, 1))
        self._click_label.setZValue(111)
        self._pw.addItem(self._click_label)
        self._click_label.hide()
        self._pw.scene().sigMouseClicked.connect(self._on_scene_clicked)

        # keep axis-scale fields in sync with mouse zoom/pan
        self._plot_item.getViewBox().sigRangeChanged.connect(
            self._on_view_range_changed)

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
            color = (self._color_override.get(key)
                     or palette[len(self._color_of) % len(palette)])
            self._color_of[key] = color
            pen = pg.mkPen(color, width=2)
            curve = self._pw.plot(x, y, pen=pen, connect="finite")
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
            self._curves[key] = curve
            sample = ClickableSample(curve, key, self._prompt_color)
            self._legend.addItem(sample, label)
            self._legend_samples[key] = sample
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
            self.apply_view_limits()
            self._pw.autoRange()
        else:
            self.apply_view_limits()

        self._clear_tooltip()
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
        self._clear_tooltip()
        self.cursor_moved.emit(float("nan"), [], False)

    def autorange(self) -> None:
        self._pw.getViewBox().enableAutoRange(x=True, y=True)
        self._pw.autoRange()

    # -------------------------------------------------------- axis scaling

    def apply_view_limits(self) -> None:
        """Constrain zoom so the curve never vanishes (peak-downsampling
        collapses to zero points on extreme zoom-out) and pan stays near
        the data."""
        vb = self._plot_item.getViewBox()
        if self._x is None or len(self._x) == 0:
            vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None,
                         minXRange=None, maxXRange=None,
                         minYRange=None, maxYRange=None)
            return
        xmin, xmax = float(np.nanmin(self._x)), float(np.nanmax(self._x))
        xspan = (xmax - xmin) or 1.0
        xpad = 0.05 * xspan
        if self._x_kind != _NON_MONOTONIC and len(self._x) > 1:
            gaps = np.abs(np.diff(self._x))
            gaps = gaps[gaps > 0]
            min_xrange = float(3 * gaps.max()) if len(gaps) else xspan / 1e4
        else:
            min_xrange = xspan / 1e4

        if self._ys:
            stacked = np.concatenate([y[np.isfinite(y)] for y in self._ys.values()
                                      if np.any(np.isfinite(y))] or [np.array([0.0])])
            ymin, ymax = float(stacked.min()), float(stacked.max())
        else:
            ymin, ymax = -1.0, 1.0
        yspan = (ymax - ymin) or 1.0
        ypad = 0.05 * yspan

        vb.setLimits(
            xMin=xmin - xpad, xMax=xmax + xpad,
            yMin=ymin - ypad, yMax=ymax + ypad,
            minXRange=min_xrange, maxXRange=20 * xspan,
            minYRange=yspan / 1e4, maxYRange=20 * yspan)

    def set_x_range(self, lo: float, hi: float) -> None:
        self._plot_item.getViewBox().setXRange(lo, hi, padding=0)

    def set_y_range(self, lo: float, hi: float) -> None:
        self._plot_item.getViewBox().setYRange(lo, hi, padding=0)

    def view_ranges(self) -> tuple[float, float, float, float]:
        (x0, x1), (y0, y1) = self._plot_item.getViewBox().viewRange()
        return float(x0), float(x1), float(y0), float(y1)

    def x_range(self) -> tuple[float, float] | None:
        if self._x is None or len(self._x) == 0:
            return None
        return float(np.nanmin(self._x)), float(np.nanmax(self._x))

    def set_cursor_x(self, value: float) -> None:
        self._cursor.setValue(value)

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
            color = (self._color_override.get(key)
                     or theme.curve_palette[i % len(theme.curve_palette)])
            self._color_of[key] = color
            pen = pg.mkPen(color, width=2)
            self._curves[key].setPen(pen)
            self._zoom_curves[key].setPen(pen)
            if key in self._legend_samples:
                self._legend_samples[key].update()
        if self._click_label.isVisible():
            self._apply_tooltip_theme()
        self._on_cursor_moved()

    # ------------------------------------------------------------ internal

    def _remove_curve(self, key: str) -> None:
        curve = self._curves.pop(key, None)
        if curve is not None:
            self._legend.removeItem(curve)
            self._pw.removeItem(curve)
        self._legend_samples.pop(key, None)
        zcurve = self._zoom_curves.pop(key, None)
        if zcurve is not None:
            self._zoom_pw.removeItem(zcurve)
        self._color_of.pop(key, None)
        self._labels.pop(key, None)
        # note: _color_override is intentionally kept so a re-checked
        # series keeps its user-picked color

    def _prompt_color(self, key: str) -> None:
        initial = QColor(self._color_of.get(key, "#888888"))
        color = QColorDialog.getColor(initial, self, "Cor da série")
        if color.isValid():
            self.set_curve_color(key, color.name())

    def set_curve_color(self, key: str, hexcolor: str) -> None:
        self._color_override[key] = hexcolor
        self._color_of[key] = hexcolor
        pen = pg.mkPen(hexcolor, width=2)
        if key in self._curves:
            self._curves[key].setPen(pen)
        if key in self._zoom_curves:
            self._zoom_curves[key].setPen(pen)
        sample = self._legend_samples.get(key)
        if sample is not None:
            sample.update()
        self._on_cursor_moved()

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

    def _on_view_range_changed(self, *_args) -> None:
        if self._x is None:
            return
        x0, x1, y0, y1 = self.view_ranges()
        self.view_range_changed.emit(x0, x1, y0, y1)

    # ------------------------------------------------------ click tooltip

    def _on_scene_clicked(self, event) -> None:
        if self._x is None or not self._curves:
            return
        try:
            left = event.button() == Qt.MouseButton.LeftButton
        except Exception:
            left = True
        if not left or event.isAccepted():
            return
        vb = self._plot_item.getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        pt = vb.mapSceneToView(event.scenePos())
        cx, cy = pt.x(), pt.y()
        pw_, ph_ = vb.viewPixelSize()

        best = None  # (dist2_px, key, xi, yi)
        dx = (self._x - cx) / (pw_ or 1.0)
        for key, y in self._ys.items():
            dy = (y - cy) / (ph_ or 1.0)
            d2 = dx * dx + dy * dy
            if not np.any(np.isfinite(d2)):
                continue
            idx = int(np.nanargmin(d2))
            if best is None or d2[idx] < best[0]:
                best = (float(d2[idx]), key, float(self._x[idx]), float(y[idx]))

        if best is None or best[0] ** 0.5 > 25.0:
            self._clear_tooltip()
            return
        self._show_tooltip(best[1], best[2], best[3])

    def _show_tooltip(self, key: str, x: float, y: float) -> None:
        color = self._color_of.get(key, "#888888")
        pen_col = ("#ffffff" if self._theme and self._theme.name == "dark"
                   else "#000000")
        self._click_marker.setData(
            [{"pos": (x, y), "brush": pg.mkBrush(color),
              "pen": pg.mkPen(pen_col, width=1.5), "size": 11}])
        self._click_label.setText(f"({x:.6g}, {y:.6g})")
        self._click_label.setPos(x, y)
        self._apply_tooltip_theme()
        self._click_label.show()

    def _clear_tooltip(self) -> None:
        self._click_marker.setData([])
        self._click_label.hide()

    def _apply_tooltip_theme(self) -> None:
        t = self._theme
        if t is None:
            return
        self._click_label.setColor(QColor(t.text))
        fill = QColor(t.panel); fill.setAlpha(235)
        self._click_label.fill = pg.mkBrush(fill)
        self._click_label.border = pg.mkPen(t.border, width=1)
        self._click_label.update()

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
