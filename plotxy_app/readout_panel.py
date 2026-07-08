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
    QApplication, QHeaderView, QLabel, QMenu, QStyledItemDelegate, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False

_KEY_ROLE = Qt.ItemDataRole.UserRole
_COLOR_ROLE = Qt.ItemDataRole.UserRole + 1
_X_ROLE = Qt.ItemDataRole.UserRole + 2
_Y_ROLE = Qt.ItemDataRole.UserRole + 3
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
    point_clicked = Signal(str, float, float)  # (key, x, y) of a value cell
    goto_point = Signal(str, float, float)     # context menu "Ir até o ponto"
    width_hint_changed = Signal(int)           # px needed to show all columns

    def __init__(self, title: str, coord_symbol: str, value_symbol: str,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._title = title
        self._coord_symbol = coord_symbol
        self._is_vertical = coord_symbol == "X"  # else horizontal cursor
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._header = QLabel(title)
        self._header.setStyleSheet("font-weight: bold;")
        self._header.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._header.customContextMenuRequested.connect(self._on_header_menu)
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
        self._tree.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_tree_menu)
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
                self._tag_point(top, key, coord, values[0])
            else:
                top.setText(2, f"{len(values)} pontos")
                for v in values[:_MAX_CHILDREN]:
                    child = QTreeWidgetItem(["", "", f"{v:.6g}"])
                    child.setTextAlignment(2, _RIGHT)
                    self._tag_point(child, key, coord, v)
                    top.addChild(child)
                if len(values) > _MAX_CHILDREN:
                    more = QTreeWidgetItem(
                        ["", "", f"+{len(values) - _MAX_CHILDREN} pontos"])
                    more.setTextAlignment(2, _RIGHT)
                    top.addChild(more)
            self._tree.addTopLevelItem(top)
            top.setExpanded(True)
        self._emit_width_hint()

    def _emit_width_hint(self) -> None:
        """Announce the width needed to show every column without a
        horizontal scrollbar, so the main window can widen the panel."""
        tree = self._tree
        header = tree.header()
        w1 = max(tree.sizeHintForColumn(1), header.sectionSizeHint(1))
        w2 = max(tree.sizeHintForColumn(2), header.sectionSizeHint(2))
        needed = (_SWATCH_COLUMN_WIDTH + w1 + w2 + 2 * tree.frameWidth()
                  + tree.verticalScrollBar().sizeHint().width())
        margins = self.layout().contentsMargins()
        needed += margins.left() + margins.right() + 8
        if needed > self.width():
            self.width_hint_changed.emit(needed)

    def _tag_point(self, item: QTreeWidgetItem, key: str,
                   coord: float, value: float) -> None:
        """Attach the (x, y) of an intersection point to a value cell so a
        click can show the graph tooltip. Vertical cursor: x=coord, y=value;
        horizontal cursor: x=value, y=coord."""
        x, y = (coord, value) if self._is_vertical else (value, coord)
        item.setData(0, _KEY_ROLE, key)
        item.setData(0, _X_ROLE, float(x))
        item.setData(0, _Y_ROLE, float(y))

    def _on_item_clicked(self, item: QTreeWidgetItem, col: int) -> None:
        # clicking the color swatch (col 0 of a series row) changes color
        if col == 0 and item.data(0, _COLOR_ROLE) is not None:
            key = item.data(0, _KEY_ROLE)
            if key:
                self.color_change_requested.emit(key)
            return
        # clicking anywhere else on a point-bearing row shows its tooltip
        x = item.data(0, _X_ROLE)
        y = item.data(0, _Y_ROLE)
        if x is not None and y is not None:
            self.point_clicked.emit(item.data(0, _KEY_ROLE), x, y)

    # ---------------------------------------------------- copy to clipboard

    @staticmethod
    def _copy(text: str) -> None:
        QApplication.clipboard().setText(text)

    def _on_tree_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        if item.childCount() > 0:
            values = [item.child(i).text(2)
                      for i in range(item.childCount())
                      if _is_number(item.child(i).text(2))]
            if not values:
                return
            menu.addAction("Copiar valores").triggered.connect(
                lambda: self._copy("\n".join(values)))
        elif _is_number(item.text(2)):
            value = item.text(2)
            menu.addAction("Copiar valor").triggered.connect(
                lambda: self._copy(value))
            x = item.data(0, _X_ROLE)
            y = item.data(0, _Y_ROLE)
            if x is not None and y is not None:
                key = item.data(0, _KEY_ROLE)
                menu.addAction("Ir até o ponto").triggered.connect(
                    lambda: self.goto_point.emit(key, x, y))
        else:
            return
        menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _on_header_menu(self, pos) -> None:
        text = self._header.text()
        if "=" not in text:
            return
        value = text.split("=", 1)[1].strip()
        menu = QMenu(self)
        menu.addAction("Copiar valor").triggered.connect(
            lambda: self._copy(value))
        menu.exec(self._header.mapToGlobal(pos))
