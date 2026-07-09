"""Amplitude spectrum (FFT): pure computation + floating window.

compute_spectrum is pure numpy (headless-testable); SpectrumWindow only
plots what it is given and holds no dataset — same split as measures.py.
"""

from __future__ import annotations

import html

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .themes import Theme


def compute_spectrum(x: np.ndarray, y: np.ndarray) -> dict | None:
    """Single-sided amplitude spectrum of the series (x, y).

    NaN pairs are dropped and samples are sorted by X. The sample period
    is the median spacing; `uniform` is False when the spacing deviates
    from it (the FFT then treats the data as uniformly sampled, so the
    result is an approximation). The DC component is removed.

    Returns {"freq", "amp", "uniform", "dt", "n"} or None when there is
    not enough data (fewer than 8 finite samples or zero time span).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 8:
        return None
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    d = np.diff(x)
    dt = float(np.median(d))
    if dt <= 0 or not np.isfinite(dt):
        return None
    uniform = bool(np.max(np.abs(d - dt)) <= 1e-3 * dt)

    amp = np.abs(np.fft.rfft(y - y.mean())) * (2.0 / n)
    amp[0] = 0.0                    # DC removed
    if n % 2 == 0 and len(amp) > 1:
        amp[-1] /= 2.0              # Nyquist bin is not doubled
    freq = np.fft.rfftfreq(n, dt)
    return {"freq": freq, "amp": amp, "uniform": uniform, "dt": dt, "n": n}


class SpectrumWindow(QWidget):
    """Floating window with the amplitude spectrum of the visible
    series. set_rows(rows) receives [(key, label, color, spec|None)]
    with spec from compute_spectrum. visibility_changed(bool) mirrors
    show/hide/close so the owner can stop refreshing a hidden window."""

    visibility_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Tool)
        self.setWindowTitle("Espectro")
        self.resize(640, 420)
        layout = QVBoxLayout(self)

        self._warn = QLabel(
            "⚠ Amostragem não uniforme — espectro aproximado.")
        self._warn.setStyleSheet("color: #e6a23c;")
        self._warn.hide()
        layout.addWidget(self._warn)

        self._pw = pg.PlotWidget()
        self._item = self._pw.getPlotItem()
        self._item.showGrid(x=True, y=True, alpha=0.15)
        self._item.setLabel("bottom", "Frequência (Hz)")
        self._item.setLabel("left", "Amplitude")
        self._legend = self._item.addLegend(offset=(10, 10))
        layout.addWidget(self._pw, stretch=1)

        # frequency cursor: draggable line + per-series readout of the
        # nearest FFT bin (coalesced like the main plot's cursors)
        self._readout = QLabel("")
        self._readout.setTextFormat(Qt.TextFormat.RichText)
        self._readout.setWordWrap(True)
        self._readout.hide()
        layout.addWidget(self._readout)
        self._cursor_timer = QTimer(self)
        self._cursor_timer.setSingleShot(True)
        self._cursor_timer.setInterval(0)
        self._cursor_timer.timeout.connect(self._update_readout)
        self._cursor = pg.InfiniteLine(angle=90, movable=True)
        self._cursor.setZValue(90)
        self._cursor.sigPositionChanged.connect(
            lambda: self._cursor_timer.start())
        self._pw.addItem(self._cursor)
        self._cursor.hide()

        self._curves: dict[str, pg.PlotDataItem] = {}
        self._data: list[tuple[str, str, np.ndarray, np.ndarray]] = []

    def set_rows(self, rows: list[tuple[str, str, str, dict | None]]) -> None:
        for curve in self._curves.values():
            self._legend.removeItem(curve)
            self._item.removeItem(curve)
        self._curves.clear()
        self._data = []
        non_uniform = False
        for key, label, color, spec in rows:
            if not spec:
                continue
            non_uniform = non_uniform or not spec["uniform"]
            # spectra carry up to n/2 points: thin non-AA pen keeps the
            # repaint cheap (same rationale as the dense-envelope mode)
            curve = self._item.plot(spec["freq"], spec["amp"],
                                    pen=pg.mkPen(color, width=1),
                                    name=label, antialias=False)
            self._curves[key] = curve
            self._data.append((label, color, spec["freq"], spec["amp"]))
        self._warn.setVisible(non_uniform)
        self._item.enableAutoRange()
        self._place_cursor()

    def _place_cursor(self) -> None:
        """Keep the cursor where the user left it while it is still inside
        the frequency range; otherwise start at the strongest peak so the
        first reading is a useful one."""
        if not self._data:
            self._cursor.hide()
            self._readout.hide()
            return
        lo = min(float(f[0]) for _l, _c, f, _a in self._data)
        hi = max(float(f[-1]) for _l, _c, f, _a in self._data)
        current = float(self._cursor.value())
        if not (lo <= current <= hi) or not self._cursor.isVisible():
            best_f, best_a = lo, -np.inf
            for _label, _color, freq, amp in self._data:
                i = int(np.nanargmax(amp)) if len(amp) else 0
                if len(amp) and amp[i] > best_a:
                    best_a, best_f = float(amp[i]), float(freq[i])
            self._cursor.setValue(best_f)
        self._cursor.setBounds((lo, hi))
        self._cursor.show()
        self._readout.show()
        self._update_readout()

    def _update_readout(self) -> None:
        if not self._data:
            return
        f = float(self._cursor.value())
        parts = [f"<b>f = {f:.6g} Hz</b>"]
        for label, color, freq, amp in self._data:
            i = int(np.searchsorted(freq, f))
            if i >= len(freq):
                i = len(freq) - 1
            elif i > 0 and abs(freq[i - 1] - f) <= abs(freq[i] - f):
                i -= 1
            parts.append(f'<span style="color:{color}">'
                         f'{html.escape(label)}</span>: {amp[i]:.6g}')
        self._readout.setText(" &nbsp;•&nbsp; ".join(parts))

    def apply_theme(self, theme: Theme) -> None:
        self._pw.setBackground(theme.plot_bg)
        for side in ("left", "bottom"):
            ax = self._item.getAxis(side)
            ax.setPen(pg.mkPen(theme.axis_color))
            ax.setTextPen(pg.mkPen(theme.axis_color))
        self._item.showGrid(x=True, y=True, alpha=theme.grid_alpha)
        self._legend.setLabelTextColor(theme.text)
        for _sample, label in self._legend.items:
            label.setText(label.text)
        self._cursor.setPen(pg.mkPen(theme.cursor_color, width=2))
        self._cursor.setHoverPen(pg.mkPen(theme.cursor_color, width=4))

    # visibility notifications (no focus dance: nothing on the main plot
    # depends on this window being active, unlike the measures region)
    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.visibility_changed.emit(True)

    def closeEvent(self, event) -> None:
        self.visibility_changed.emit(False)
        super().closeEvent(event)

    def hideEvent(self, event) -> None:
        if not self.isVisible():
            self.visibility_changed.emit(False)
        super().hideEvent(event)
