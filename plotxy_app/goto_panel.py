""""Ir para" popup: position the vertical/horizontal cursors by value.

Both fields are always visible; each is enabled only while its cursor is
enabled in the Cursores menu. Validation errors are shown inline (a
message box would dismiss the Qt.Popup window).
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QWidget,
)

from .axis_panel import _period_validator


class GotoPanel(QWidget):
    """goto_requested(x, y) carries a float per field, or None when the
    field is disabled or left empty."""

    goto_requested = Signal(object, object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)

        box = QGroupBox("Ir para")
        grid = QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)

        grid.addWidget(QLabel("Cursor vertical — X"), 0, 0)
        self._x_edit = QLineEdit()
        self._x_edit.setValidator(_period_validator(self._x_edit))
        self._x_edit.returnPressed.connect(self._on_go_x)
        grid.addWidget(self._x_edit, 0, 1)

        grid.addWidget(QLabel("Cursor horizontal — Y"), 1, 0)
        self._y_edit = QLineEdit()
        self._y_edit.setValidator(_period_validator(self._y_edit))
        self._y_edit.returnPressed.connect(self._on_go_y)
        grid.addWidget(self._y_edit, 1, 1)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #e74c3c;")
        self._error_label.setWordWrap(True)
        self._error_label.hide()
        grid.addWidget(self._error_label, 2, 0, 1, 2)

        self._go_x_btn = QPushButton("Ir para X")
        self._go_x_btn.clicked.connect(self._on_go_x)
        grid.addWidget(self._go_x_btn, 3, 0)
        self._go_y_btn = QPushButton("Ir para Y")
        self._go_y_btn.clicked.connect(self._on_go_y)
        grid.addWidget(self._go_y_btn, 3, 1)

        layout.addWidget(box)

    # ------------------------------------------------------------- public

    def set_enabled_states(self, vertical: bool, horizontal: bool) -> None:
        """Fields and buttons stay visible at all times; only their enabled
        state follows the cursor toggles."""
        self._x_edit.setEnabled(vertical)
        self._go_x_btn.setEnabled(vertical)
        self._y_edit.setEnabled(horizontal)
        self._go_y_btn.setEnabled(horizontal)

    def set_positions(self, x: float | None, y: float | None) -> None:
        if x is not None and not self._x_edit.hasFocus():
            self._x_edit.setText(f"{x:.6g}")
        if y is not None and not self._y_edit.hasFocus():
            self._y_edit.setText(f"{y:.6g}")

    def show_error(self, text: str) -> None:
        self._error_label.setText(text)
        self._error_label.show()

    def clear_error(self) -> None:
        self._error_label.hide()

    # ------------------------------------------------------------ internal

    def _parse(self, edit: QLineEdit) -> tuple[float | None, bool]:
        """Returns (value, ok). Disabled/empty fields yield (None, True)."""
        if not edit.isEnabled() or not edit.text().strip():
            return None, True
        try:
            return float(edit.text().strip()), True
        except ValueError:
            return None, False

    def _on_go_x(self) -> None:
        x, ok = self._parse(self._x_edit)
        if not ok:
            self.show_error("Valor inválido.")
            return
        self.clear_error()
        self.goto_requested.emit(x, None)

    def _on_go_y(self) -> None:
        y, ok = self._parse(self._y_edit)
        if not ok:
            self.show_error("Valor inválido.")
            return
        self.clear_error()
        self.goto_requested.emit(None, y)
