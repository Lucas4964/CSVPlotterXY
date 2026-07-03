"""Axis-scale controls: numeric X/Y min-max fields + Apply/Auto."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QWidget,
)


class AxisPanel(QWidget):
    """Lets the user set exact X/Y ranges. range_changed carries the four
    values; auto_requested asks the plot to autorange. Fields also reflect
    the current view (set_ranges), enabling mouse zoom/pan to update them
    live. Only user actions emit — no feedback loop with set_ranges."""

    range_changed = Signal(float, float, float, float)
    auto_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)

        box = QGroupBox("Escala dos eixos")
        grid = QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)

        self._fields: dict[str, QLineEdit] = {}
        for row, (label, key) in enumerate(
                [("X mín", "xmin"), ("X máx", "xmax"),
                 ("Y mín", "ymin"), ("Y máx", "ymax")]):
            grid.addWidget(QLabel(label), row, 0)
            edit = QLineEdit()
            edit.setValidator(QDoubleValidator())
            edit.returnPressed.connect(self._on_apply)
            self._fields[key] = edit
            grid.addWidget(edit, row, 1)

        buttons = QHBoxLayout()
        apply_btn = QPushButton("Aplicar")
        apply_btn.clicked.connect(self._on_apply)
        buttons.addWidget(apply_btn)
        auto_btn = QPushButton("Auto")
        auto_btn.clicked.connect(self.auto_requested)
        buttons.addWidget(auto_btn)
        grid.addLayout(buttons, 4, 0, 1, 2)

        layout.addWidget(box)

    def set_ranges(self, xmin: float, xmax: float,
                   ymin: float, ymax: float) -> None:
        """Reflect the current view. Skips any field being edited so the
        user's typing isn't clobbered by live mouse updates."""
        values = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
        for key, edit in self._fields.items():
            if edit.hasFocus():
                continue
            edit.setText(f"{values[key]:.6g}")
            edit.setStyleSheet("")

    def _on_apply(self) -> None:
        try:
            vals = {k: float(e.text().replace(",", "."))
                    for k, e in self._fields.items()}
        except ValueError:
            self._flag(lambda k: self._not_float(self._fields[k].text()))
            return
        bad_x = vals["xmin"] >= vals["xmax"]
        bad_y = vals["ymin"] >= vals["ymax"]
        if bad_x or bad_y:
            self._flag(lambda k: (bad_x if k.startswith("x") else bad_y))
            return
        self._flag(lambda _k: False)
        self.range_changed.emit(
            vals["xmin"], vals["xmax"], vals["ymin"], vals["ymax"])

    def _flag(self, is_bad) -> None:
        for key, e in self._fields.items():
            e.setStyleSheet("border: 1px solid #e74c3c;" if is_bad(key) else "")

    @staticmethod
    def _not_float(text: str) -> bool:
        try:
            float(text.replace(",", "."))
            return False
        except ValueError:
            return True
