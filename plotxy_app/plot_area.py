"""Plot widget: curves, legend, draggable cursor, intersection markers
and a local zoom region with a synchronized magnified panel."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QSplitter, QVBoxLayout, QWidget

from .themes import Theme

pg.setConfigOptions(antialias=True)

_INCREASING, _DECREASING, _NON_MONOTONIC = 0, 1, 2


def polyline_crossings(a: np.ndarray, b: np.ndarray, level: float) -> np.ndarray:
    """All interpolated values of `b` where the polyline (a, b) crosses
    a == level, sorted ascending. Vectorized; segments containing NaN are
    skipped; coincident crossings from consecutive segments sharing an
    endpoint are deduplicated."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2:
        if len(a) == 1 and np.isfinite(a[0]) and a[0] == level and np.isfinite(b[0]):
            return np.array([b[0]])
        return np.empty(0)
    d = a - level
    d0, d1 = d[:-1], d[1:]
    b0, b1 = b[:-1], b[1:]
    valid = (np.isfinite(d0) & np.isfinite(d1)
             & np.isfinite(b0) & np.isfinite(b1))
    with np.errstate(invalid="ignore"):
        cross = valid & (d0 * d1 <= 0) & ~((d0 == 0) & (d1 == 0))
    idx = np.nonzero(cross)[0]
    if len(idx) == 0:
        # a stretch lying exactly on the level (d == 0 throughout)
        flat = valid & (d0 == 0) & (d1 == 0)
        if np.any(flat):
            i = int(np.nonzero(flat)[0][0])
            return np.array([b[i]])
        return np.empty(0)
    denom = d0[idx] - d1[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(denom != 0, d0[idx] / denom, 0.0)
    vals = np.sort(b0[idx] + t * (b1[idx] - b0[idx]))
    if len(vals) > 1:
        span = max(float(abs(vals[-1] - vals[0])), 1e-30)
        keep = np.concatenate(([True], np.abs(np.diff(vals)) > 1e-9 * span))
        vals = vals[keep]
    return vals


class PlotArea(QWidget):
    """Interactive plot working purely on arrays (no dataset knowledge).

    set_series receives the X array plus (key, label, y) tuples already
    paired/truncated by the Project. Two draggable cursors (vertical and
    horizontal) report every intersection with the plotted series:
    v_cursor_moved / h_cursor_moved emit
    (coord, [(key, label, color, [values…])]).
    """

    v_cursor_moved = Signal(float, list)
    h_cursor_moved = Signal(float, list)
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
        self._theme: Theme | None = None
        self._v_enabled = True   # vertical cursor shown by default
        self._h_enabled = False  # horizontal cursor off by default
        self._click_interpolate = False  # click-tooltip snaps to samples
        self._snap_to_samples = False    # restrict cursor motion to samples
        self._y_bounds: tuple[float, float] | None = None
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
        self._cursor.sigPositionChanged.connect(self._on_v_cursor_moved)
        self._pw.addItem(self._cursor)
        self._cursor.hide()

        self._markers = pg.ScatterPlotItem(size=9, pxMode=True)
        self._markers.setZValue(100)
        self._pw.addItem(self._markers)

        self._hcursor = pg.InfiniteLine(angle=0, movable=True)
        self._hcursor.setZValue(90)
        self._hcursor.sigPositionChanged.connect(self._on_h_cursor_moved)
        self._pw.addItem(self._hcursor)
        self._hcursor.hide()

        self._h_markers = pg.ScatterPlotItem(size=9, pxMode=True)
        self._h_markers.setZValue(100)
        self._pw.addItem(self._h_markers)

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
            # no auto-downsampling: pyqtgraph's "peak" mode collapses to an
            # empty path at extreme zoom-out, making the curve vanish;
            # clipToView keeps extreme zoom-IN cheap and correct
            curve = self._pw.plot(x, y, pen=pen, name=label, connect="finite")
            curve.setClipToView(True)
            self._curves[key] = curve
            zcurve = self._zoom_pw.plot(x, y, pen=pen, connect="finite")
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
        self._update_hcursor_bounds(reset=x_changed)

        self._clear_tooltip()
        has_curves = bool(self._curves)
        self._cursor.setVisible(has_curves and self._v_enabled)
        self._markers.setVisible(has_curves and self._v_enabled)
        self._hcursor.setVisible(has_curves and self._h_enabled)
        self._h_markers.setVisible(has_curves and self._h_enabled)
        self._update_cursor_readouts()

    def clear(self) -> None:
        for key in list(self._curves):
            self._remove_curve(key)
        self._ys.clear()
        self._x = None
        self._x_key = ""
        self._cursor.hide()
        self._markers.hide()
        self._markers.setData([])
        self._hcursor.hide()
        self._h_markers.hide()
        self._h_markers.setData([])
        self._y_bounds = None
        self._clear_tooltip()
        self.v_cursor_moved.emit(float("nan"), [])
        self.h_cursor_moved.emit(float("nan"), [])

    def autorange(self) -> None:
        self._pw.getViewBox().enableAutoRange(x=True, y=True)
        self._pw.autoRange()

    # -------------------------------------------------------- axis scaling

    def set_manual_ranges(self, xmin: float, xmax: float,
                          ymin: float, ymax: float) -> None:
        """Set the visible region to exactly the requested limits
        (Desmos-style). Zoom/pan is unlimited, so nothing to clamp."""
        self._plot_item.getViewBox().setRange(
            xRange=(xmin, xmax), yRange=(ymin, ymax), padding=0)

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

    def set_cursor_y(self, value: float) -> None:
        self._hcursor.setValue(value)

    def cursor_positions(self) -> tuple[float, float]:
        return float(self._cursor.value()), float(self._hcursor.value())

    def y_data_range(self) -> tuple[float, float] | None:
        return self._y_bounds

    def set_cursor_visible(self, orientation: str, visible: bool) -> None:
        """Show/hide the vertical ('v') or horizontal ('h') cursor."""
        has_curves = bool(self._curves)
        if orientation == "v":
            self._v_enabled = visible
            self._cursor.setVisible(visible and has_curves)
            self._markers.setVisible(visible and has_curves)
            self._on_v_cursor_moved()
        else:
            self._h_enabled = visible
            self._hcursor.setVisible(visible and has_curves)
            self._h_markers.setVisible(visible and has_curves)
            self._on_h_cursor_moved()

    def _update_hcursor_bounds(self, reset: bool) -> None:
        finite = [y[np.isfinite(y)] for y in self._ys.values()
                  if np.any(np.isfinite(y))]
        if not finite:
            self._y_bounds = None
            return
        stacked = np.concatenate(finite)
        ymin, ymax = float(stacked.min()), float(stacked.max())
        self._y_bounds = (ymin, ymax)
        self._hcursor.setBounds((ymin, ymax))
        cur = float(self._hcursor.value())
        if reset or not (ymin <= cur <= ymax):
            self._hcursor.setValue((ymin + ymax) / 2)

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
        # setLabelTextColor only updates the label's color option; the HTML
        # is re-rendered (picking up the new color) only on setText, so
        # force a re-render for the already-existing legend entries.
        for _sample, label in self._legend.items:
            label.setText(label.text)
        self._cursor.setPen(pg.mkPen(theme.cursor_color, width=2))
        self._cursor.setHoverPen(pg.mkPen(theme.cursor_color, width=4))
        # horizontal cursor uses the accent color for instant distinction
        self._hcursor.setPen(pg.mkPen(theme.accent, width=2))
        self._hcursor.setHoverPen(pg.mkPen(theme.accent, width=4))

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
        if self._click_label.isVisible():
            self._apply_tooltip_theme()
        self._update_cursor_readouts()

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
        # note: _color_override is intentionally kept so a re-checked
        # series keeps its user-picked color

    def prompt_color(self, key: str) -> None:
        """Open a color picker for a series (invoked from the readout
        panel swatch)."""
        if key not in self._curves:
            return
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
        self._update_cursor_readouts()

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

        if self._click_interpolate:
            best = self._nearest_on_curve(cx, cy, pw_ or 1.0, ph_ or 1.0)
        else:
            best = self._nearest_sample(cx, cy, pw_ or 1.0, ph_ or 1.0)

        if best is None or best[0] ** 0.5 > 25.0:
            self._clear_tooltip()
            return
        self._show_tooltip(best[1], best[2], best[3])

    def set_click_interpolation(self, enabled: bool) -> None:
        """When enabled, clicking anywhere along a curve selects the
        interpolated point on the nearest segment instead of snapping to
        the nearest original sample."""
        self._click_interpolate = enabled

    def _nearest_sample(self, cx: float, cy: float,
                        pw_: float, ph_: float):
        """Closest original data point to the click, in pixel space.
        Returns (dist2_px, key, x, y) or None."""
        best = None
        dx = (self._x - cx) / pw_
        for key, y in self._ys.items():
            dy = (y - cy) / ph_
            d2 = dx * dx + dy * dy
            if not np.any(np.isfinite(d2)):
                continue
            idx = int(np.nanargmin(d2))
            if best is None or d2[idx] < best[0]:
                best = (float(d2[idx]), key, float(self._x[idx]), float(y[idx]))
        return best

    def _nearest_on_curve(self, cx: float, cy: float,
                          pw_: float, ph_: float):
        """Closest point on any curve segment to the click (interpolated),
        in pixel space. Returns (dist2_px, key, x, y) or None."""
        if len(self._x) < 2:
            return self._nearest_sample(cx, cy, pw_, ph_)
        best = None
        # pixel-space coordinates relative to the click (click at origin)
        px = (self._x - cx) / pw_
        for key, y in self._ys.items():
            py = (y - cy) / ph_
            p0x, p1x = px[:-1], px[1:]
            p0y, p1y = py[:-1], py[1:]
            vx, vy = p1x - p0x, p1y - p0y
            len2 = vx * vx + vy * vy
            valid = (np.isfinite(p0x) & np.isfinite(p0y)
                     & np.isfinite(p1x) & np.isfinite(p1y))
            with np.errstate(invalid="ignore", divide="ignore"):
                t = np.where(len2 > 0,
                             -(p0x * vx + p0y * vy) / np.where(len2 > 0, len2, 1.0),
                             0.0)
            t = np.clip(t, 0.0, 1.0)
            qx = p0x + t * vx
            qy = p0y + t * vy
            d2 = qx * qx + qy * qy
            d2 = np.where(valid, d2, np.inf)
            if not np.any(np.isfinite(d2)):
                continue
            i = int(np.argmin(d2))
            if best is None or d2[i] < best[0]:
                ti = float(t[i])
                x_int = float(self._x[i] + ti * (self._x[i + 1] - self._x[i]))
                y_int = float(y[i] + ti * (y[i + 1] - y[i]))
                best = (float(d2[i]), key, x_int, y_int)
        return best

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

    def _update_cursor_readouts(self) -> None:
        self._on_v_cursor_moved()
        self._on_h_cursor_moved()

    def _spot(self, x: float, y: float, color: str) -> dict:
        return {
            "pos": (x, y),
            "brush": pg.mkBrush(color),
            "pen": pg.mkPen("#ffffff" if self._theme and self._theme.name == "dark"
                            else "#000000", width=1),
        }

    def set_cursor_snap(self, enabled: bool) -> None:
        """When enabled, both cursors move only onto original sample
        values (nearest-sample snapping)."""
        self._snap_to_samples = enabled
        self._on_v_cursor_moved()
        self._on_h_cursor_moved()

    def _y_samples(self) -> np.ndarray | None:
        finite = [y[np.isfinite(y)] for y in self._ys.values()
                  if np.any(np.isfinite(y))]
        return np.concatenate(finite) if finite else None

    def _maybe_snap(self, line: pg.InfiniteLine,
                    samples: np.ndarray | None) -> float:
        """Return the cursor value, snapped to the nearest sample when
        snapping is on. Updates the line without re-triggering its signal."""
        value = float(line.value())
        if not self._snap_to_samples or samples is None or len(samples) == 0:
            return value
        with np.errstate(invalid="ignore"):
            idx = int(np.nanargmin(np.abs(samples - value)))
        snapped = float(samples[idx])
        if snapped != value:
            line.blockSignals(True)
            line.setValue(snapped)
            line.blockSignals(False)
        return snapped

    def _on_v_cursor_moved(self) -> None:
        if self._x is None or not self._curves or not self._v_enabled:
            self._markers.setData([])
            self.v_cursor_moved.emit(float("nan"), [])
            return
        cx = self._maybe_snap(self._cursor, self._x)
        spots, rows = [], []
        if self._x_kind != _NON_MONOTONIC:
            # fast path: single interpolated value per series
            keys = list(self._curves)
            ys = self._values_at(cx, keys)
            for key, y in zip(keys, ys):
                color = self._color_of[key]
                vals = [y] if np.isfinite(y) else []
                rows.append((key, self._labels[key], color, vals))
                if vals:
                    spots.append(self._spot(cx, y, color))
        else:
            # general case: every real crossing of x == cx
            for key in self._curves:
                color = self._color_of[key]
                vals = [float(v) for v in
                        polyline_crossings(self._x, self._ys[key], cx)]
                rows.append((key, self._labels[key], color, vals))
                spots.extend(self._spot(cx, v, color) for v in vals)
        self._markers.setData(spots)
        self.v_cursor_moved.emit(cx, rows)

    def _on_h_cursor_moved(self) -> None:
        if self._x is None or not self._curves or not self._h_enabled:
            self._h_markers.setData([])
            self.h_cursor_moved.emit(float("nan"), [])
            return
        cy = self._maybe_snap(self._hcursor, self._y_samples())
        spots, rows = [], []
        for key in self._curves:
            color = self._color_of[key]
            vals = [float(v) for v in
                    polyline_crossings(self._ys[key], self._x, cy)]
            rows.append((key, self._labels[key], color, vals))
            spots.extend(self._spot(v, cy, color) for v in vals)
        self._h_markers.setData(spots)
        self.h_cursor_moved.emit(cy, rows)

    def _values_at(self, cx: float, keys: list[str]) -> list[float]:
        """Value of each series at cursor X for monotonic X: one
        searchsorted + linear interpolation shared by all series."""
        x = self._x
        n = len(x)
        if n == 0:
            return [float("nan")] * len(keys)

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
