"""Cursor readout panel: table of series values at the cursor position."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

_KEY_ROLE = Qt.ItemDataRole.UserRole


class ReadoutPanel(QWidget):
    """Table of series values at the cursor. Clicking a series' color
    swatch emits color_change_requested(key) so the plot can recolor it."""

    color_change_requested = Signal(str)

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

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["", "Série", "Valor"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._table.setAlternatingRowColors(True)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.cellClicked.connect(self._on_cell_clicked)
        layout.addWidget(self._table, stretch=1)

    def update_values(self, x: float,
                      rows: list[tuple[str, str, str, float]],
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
        for r, (key, name, color, y) in enumerate(rows):
            swatch = QPixmap(12, 12)
            swatch.fill(QColor(color))
            color_item = QTableWidgetItem()
            color_item.setData(Qt.ItemDataRole.DecorationRole, swatch)
            color_item.setData(_KEY_ROLE, key)
            color_item.setToolTip("Clique para mudar a cor da série")
            name_item = QTableWidgetItem(name)
            value_item = QTableWidgetItem(
                f"{y:.6g}" if math.isfinite(y) else "—")
            value_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(r, 0, color_item)
            self._table.setItem(r, 1, name_item)
            self._table.setItem(r, 2, value_item)

    def _on_cell_clicked(self, row: int, col: int) -> None:
        if col != 0:
            return
        item = self._table.item(row, 0)
        key = item.data(_KEY_ROLE) if item is not None else None
        if key:
            self.color_change_requested.emit(key)
