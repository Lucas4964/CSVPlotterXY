"""Cursor readout panels: per-cursor tree of series intersection values.

Follows the pattern of professional plotting tools (Origin, LabVIEW,
digital oscilloscopes): one row per series; a single intersection shows
its value inline, multiple intersections show the count with the values
nested below.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QHeaderView, QLabel, QStyledItemDelegate, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

_KEY_ROLE = Qt.ItemDataRole.UserRole
_COLOR_ROLE = Qt.ItemDataRole.UserRole + 1
_SWATCH_SIZE = 12
_SWATCH_COLUMN_WIDTH = 28
_MAX_CHILDREN = 50  # crossings listed per series before "+N pontos"

_RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter


class _SwatchDelegate(QStyledItemDelegate):
    """Paints a small filled square centered in the cell, instead of the
    default left-anchored icon rendering used by DecorationRole."""

    def paint(self, painter: QPainter, option, index) -> None:
        color = index.data(_COLOR_ROLE)
        if color is None:
            super().paint(painter, option, index)
            return
        rect = option.rect
        x = rect.x() + (rect.width() - _SWATCH_SIZE) // 2
        y = rect.y() + (rect.height() - _SWATCH_SIZE) // 2
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        qcolor = QColor(color)
        painter.setBrush(qcolor)
        painter.setPen(qcolor.darker(150))
        painter.drawRoundedRect(x, y, _SWATCH_SIZE, _SWATCH_SIZE, 2, 2)
        painter.restore()


class CursorReadout(QWidget):
    """Intersection values for one cursor. update_values(coord, rows)
    receives rows = [(key, label, color, [values…])]. Clicking a series'
    color swatch emits color_change_requested(key)."""

    color_change_requested = Signal(str)

    def __init__(self, title: str, coord_symbol: str, value_symbol: str,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._title = title
        self._coord_symbol = coord_symbol
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._header = QLabel(title)
        self._header.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._header)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["", "Série", value_symbol])
        self._tree.setItemDelegateForColumn(0, _SwatchDelegate(self._tree))
        self._tree.setSelectionMode(QTreeWidget.SelectionMode.NoSelection)
        self._tree.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._tree.setAlternatingRowColors(True)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._tree.setColumnWidth(0, _SWATCH_COLUMN_WIDTH)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree, stretch=1)

    def update_values(self, coord: float,
                      rows: list[tuple[str, str, str, list[float]]]) -> None:
        if not self.isVisible():
            return
        self._tree.clear()
        if math.isnan(coord) or not rows:
            self._header.setText(self._title)
            return
        self._header.setText(f"{self._coord_symbol} = {coord:.6g}")

        for key, name, color, values in rows:
            top = QTreeWidgetItem(["", name, ""])
            top.setData(0, _KEY_ROLE, key)
            top.setData(0, _COLOR_ROLE, color)
            top.setToolTip(0, "Clique para mudar a cor da série")
            top.setTextAlignment(2, _RIGHT)
            if not values:
                top.setText(2, "—")
            elif len(values) == 1:
                top.setText(2, f"{values[0]:.6g}")
            else:
                top.setText(2, f"{len(values)} pontos")
                for v in values[:_MAX_CHILDREN]:
                    child = QTreeWidgetItem(["", "", f"{v:.6g}"])
                    child.setTextAlignment(2, _RIGHT)
                    top.addChild(child)
                if len(values) > _MAX_CHILDREN:
                    more = QTreeWidgetItem(
                        ["", "", f"+{len(values) - _MAX_CHILDREN} pontos"])
                    more.setTextAlignment(2, _RIGHT)
                    top.addChild(more)
            self._tree.addTopLevelItem(top)
            top.setExpanded(True)

    def _on_item_clicked(self, item: QTreeWidgetItem, col: int) -> None:
        if col != 0:
            return
        key = item.data(0, _KEY_ROLE)
        if key:
            self.color_change_requested.emit(key)
