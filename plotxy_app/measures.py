"""Medidas: interval statistics per series + dedicated analysis window.

compute_measures is pure numpy (headless-testable). The window is a
discreet Qt.Tool floater; the selection region on the plot is active only
while this window is open and the application keeps focus.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QHeaderView, QLabel, QMenu, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .readout_panel import _COLOR_ROLE, _SWATCH_COLUMN_WIDTH, _SwatchDelegate

_INCREASING, _DECREASING, _NON_MONOTONIC = 0, 1, 2

COLUMNS = ["Máx", "Mín", "Média", "ΔX", "ΔY", "Área"]
_TOOLTIPS = {
    "Máx": "Maior valor de Y no intervalo",
    "Mín": "Menor valor de Y no intervalo",
    "Média": "Média dos valores de Y no intervalo",
    "ΔX": "X final − X inicial (amostras dentro do intervalo)",
    "ΔY": "Y final − Y inicial (amostras dentro do intervalo)",
    "Área": "Integral trapezoidal de Y em relação a X (com sinal)",
}


def compute_measures(x: np.ndarray, y: np.ndarray, lo: float, hi: float,
                     x_kind: int) -> dict | None:
    """Statistics of (x, y) samples whose x lies in [lo, hi].

    Monotonic X uses a searchsorted slice (views, no copy); non-monotonic
    falls back to a boolean mask. Non-finite pairs are dropped. Returns
    None when the interval contains no finite samples.
    """
    if lo > hi:
        lo, hi = hi, lo
    if x_kind == _INCREASING:
        i0 = int(np.searchsorted(x, lo, side="left"))
        i1 = int(np.searchsorted(x, hi, side="right"))
        xs, ys = x[i0:i1], y[i0:i1]
    elif x_kind == _DECREASING:
        rev = x[::-1]
        i0 = int(np.searchsorted(rev, lo, side="left"))
        i1 = int(np.searchsorted(rev, hi, side="right"))
        n = len(x)
        xs, ys = x[n - i1:n - i0], y[n - i1:n - i0]
    else:
        mask = (x >= lo) & (x <= hi)
        xs, ys = x[mask], y[mask]

    finite = np.isfinite(xs) & np.isfinite(ys)
    if not np.all(finite):
        xs, ys = xs[finite], ys[finite]
    if len(xs) == 0:
        return None

    area = float(np.trapezoid(ys, xs)) if len(xs) > 1 else 0.0
    return {
        "max": float(ys.max()),
        "min": float(ys.min()),
        "mean": float(ys.mean()),
        "dx": float(xs[-1] - xs[0]),
        "dy": float(ys[-1] - ys[0]),
        "area": area,
        "n": int(len(xs)),
    }


class MeasuresWindow(QWidget):
    """Discreet floating window listing interval statistics per active
    series. visibility_changed(active) drives the plot's selection region:
    True while the window is shown and the application holds focus."""

    visibility_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Tool)
        self.setWindowTitle("Medidas")
        self.resize(600, 240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hint = QLabel("Arraste as bordas da região destacada no gráfico "
                      "para definir o intervalo de análise.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._interval_label = QLabel("")
        self._interval_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._interval_label)

        # columns: [swatch | Série | Máx | Mín | Média | ΔX | ΔY | Área]
        self._table = QTableWidget(0, 2 + len(COLUMNS))
        self._table.setHorizontalHeaderLabels(["", "Série"] + COLUMNS)
        for i, name in enumerate(COLUMNS, start=2):
            self._table.horizontalHeaderItem(i).setToolTip(_TOOLTIPS[name])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._table.setAlternatingRowColors(True)
        self._table.setItemDelegateForColumn(0, _SwatchDelegate(self._table))
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(0, _SWATCH_COLUMN_WIDTH)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for i in range(2, 2 + len(COLUMNS)):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_table_menu)
        layout.addWidget(self._table, stretch=1)

    # ------------------------------------------------------------- public

    def set_rows(self, lo: float, hi: float,
                 rows: list[tuple[str, str, str, dict | None]]) -> None:
        self._interval_label.setText(f"X ∈ [{lo:.6g}, {hi:.6g}]")
        self._table.setRowCount(len(rows))
        for r, (_key, label, color, m) in enumerate(rows):
            color_item = QTableWidgetItem()
            color_item.setData(_COLOR_ROLE, color)
            self._table.setItem(r, 0, color_item)
            self._table.setItem(r, 1, QTableWidgetItem(label))
            values = (["—"] * len(COLUMNS) if m is None else
                      [f"{m['max']:.6g}", f"{m['min']:.6g}", f"{m['mean']:.6g}",
                       f"{m['dx']:.6g}", f"{m['dy']:.6g}", f"{m['area']:.6g}"])
            for c, text in enumerate(values, start=2):
                item = QTableWidgetItem(text)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(r, c, item)

    # ----------------------------------------------------- window lifecycle

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.visibility_changed.emit(True)

    def closeEvent(self, event) -> None:
        self.visibility_changed.emit(False)
        super().closeEvent(event)

    def hideEvent(self, event) -> None:
        self.visibility_changed.emit(False)
        super().hideEvent(event)

    def changeEvent(self, event) -> None:
        if event.type() == QEvent.Type.WindowStateChange:
            self.visibility_changed.emit(
                self.isVisible() and not self.isMinimized())
        super().changeEvent(event)

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.WindowDeactivate:
            # after focus settles: if it left the application entirely,
            # deactivate; if it went to the main window (user dragging the
            # region on the plot), keep the selection active
            QTimer.singleShot(0, self._check_app_focus)
        elif event.type() == QEvent.Type.WindowActivate:
            if self.isVisible() and not self.isMinimized():
                self.visibility_changed.emit(True)
        return super().event(event)

    def _check_app_focus(self) -> None:
        if not self.isVisible():
            return
        if QGuiApplication.applicationState() != Qt.ApplicationState.ApplicationActive \
                or QApplication.activeWindow() is None:
            self.visibility_changed.emit(False)

    # ------------------------------------------------------------ internal

    def _on_table_menu(self, pos) -> None:
        item = self._table.itemAt(pos)
        if item is None or item.column() < 2 or item.text() == "—":
            return
        text = item.text()
        menu = QMenu(self)
        menu.addAction("Copiar valor").triggered.connect(
            lambda: QApplication.clipboard().setText(text))
        menu.exec(self._table.viewport().mapToGlobal(pos))
