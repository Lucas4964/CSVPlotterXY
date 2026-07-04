"""Dropdown popup for toggling the vertical/horizontal cursors."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QGroupBox, QVBoxLayout, QWidget


class CursorMenu(QWidget):
    """Cursor/inspection toggles shown as a toolbar dropdown (Qt.Popup
    window flag is applied by the caller). Vertical cursor defaults to on,
    horizontal to off. cursors_changed(vertical, horizontal) fires on any
    cursor toggle; interpolation_changed(bool) on the point-selection
    toggle."""

    cursors_changed = Signal(bool, bool)
    interpolation_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)

        box = QGroupBox("Cursores")
        inner = QVBoxLayout(box)
        inner.setContentsMargins(8, 8, 8, 8)
        inner.setSpacing(4)

        self._v_check = QCheckBox("Exibir cursor vertical")
        self._v_check.setChecked(True)
        self._h_check = QCheckBox("Exibir cursor horizontal")
        self._h_check.setChecked(False)
        for check in (self._v_check, self._h_check):
            check.toggled.connect(self._emit)
            inner.addWidget(check)

        layout.addWidget(box)

        sel_box = QGroupBox("Seleção de pontos")
        sel_inner = QVBoxLayout(sel_box)
        sel_inner.setContentsMargins(8, 8, 8, 8)
        self._interp_check = QCheckBox("Interpolar ao clicar na curva")
        self._interp_check.setChecked(False)
        self._interp_check.setToolTip(
            "Desligado: o clique seleciona o ponto original mais próximo.\n"
            "Ligado: o clique seleciona o ponto interpolado sobre a curva.")
        self._interp_check.toggled.connect(self.interpolation_changed)
        sel_inner.addWidget(self._interp_check)
        layout.addWidget(sel_box)

    def states(self) -> tuple[bool, bool]:
        return self._v_check.isChecked(), self._h_check.isChecked()

    def _emit(self, _checked: bool) -> None:
        self.cursors_changed.emit(*self.states())
