"""Cursor readout panel: table of series values at the cursor position."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)


class ReadoutPanel(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._header = QLabel("Cursor")
        self._header.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._header)

        self._hint = QLabel("")
        self._hint.setVisible(False)
        layout.addWidget(self._hint)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Série", "Valor"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._table.setAlternatingRowColors(True)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self._table, stretch=1)

    def update_values(self, x: float, rows: list[tuple[str, str, float]],
                      nearest_mode: bool) -> None:
        if not self.isVisible():
            return
        if math.isnan(x) or not rows:
            self._header.setText("Cursor")
            self._hint.setVisible(False)
            self._table.setRowCount(0)
            return

        self._header.setText(f"X = {x:.6g}")
        self._hint.setText("(ponto mais próximo — X não monotônico)")
        self._hint.setVisible(nearest_mode)

        self._table.setRowCount(len(rows))
        for r, (name, color, y) in enumerate(rows):
            name_item = QTableWidgetItem(name)
            swatch = QPixmap(10, 10)
            swatch.fill(QColor(color))
            name_item.setData(Qt.ItemDataRole.DecorationRole, swatch)
            value_item = QTableWidgetItem(
                f"{y:.6g}" if math.isfinite(y) else "—")
            value_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(r, 0, name_item)
            self._table.setItem(r, 1, value_item)
