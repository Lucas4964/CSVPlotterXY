"""Dialog for creating/editing a custom expression series."""

from __future__ import annotations

from PySide6.QtCore import QSortFilterProxyModel, Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QLineEdit, QListView, QVBoxLayout,
    QWidget,
)

from . import expressions
from .project import CustomSeries, Project, ProjectError


class SeriesDialog(QDialog):
    def __init__(self, project: Project,
                 existing: CustomSeries | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._project = project
        self._existing = existing
        self.setWindowTitle("Editar série personalizada" if existing
                            else "Nova série personalizada")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        layout.addWidget(QLabel("Nome"))
        self._name_edit = QLineEdit()
        layout.addWidget(self._name_edit)

        layout.addWidget(QLabel("Expressão"))
        self._expr_edit = QLineEdit()
        self._expr_edit.setPlaceholderText("ex.: P1 + P2 + P3")
        layout.addWidget(self._expr_edit)

        hint = QLabel("Operações: +  −  *  /  **  abs()  sqrt()  parênteses.\n"
                      "D(série) = derivada em relação ao eixo X atual.\n"
                      "Duplo clique numa série abaixo insere o nome na "
                      "expressão (aspas automáticas quando necessário).")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #e74c3c;")
        self._error_label.setWordWrap(True)
        self._error_label.hide()
        layout.addWidget(self._error_label)

        layout.addWidget(QLabel("Séries disponíveis"))
        self._list_filter = QLineEdit()
        self._list_filter.setPlaceholderText("Filtrar…")
        self._list_filter.setClearButtonEnabled(True)
        layout.addWidget(self._list_filter)

        self._list_model = QStandardItemModel(self)
        exclude = existing.name if existing else None
        for ref in project._all_refs():
            if ref.kind == "custom" and ref.name == exclude:
                continue
            item = QStandardItem(project.label(ref))
            item.setEditable(False)
            self._list_model.appendRow(item)
        proxy = QSortFilterProxyModel(self)
        proxy.setSourceModel(self._list_model)
        proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._list_filter.textChanged.connect(proxy.setFilterFixedString)

        self._list = QListView()
        self._list.setModel(proxy)
        self._list.setEditTriggers(QListView.EditTrigger.NoEditTriggers)
        self._list.doubleClicked.connect(self._insert_series_name)
        layout.addWidget(self._list, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if existing:
            self._name_edit.setText(existing.name)
            self._expr_edit.setText(existing.expression)

    def result_values(self) -> tuple[str, str]:
        return self._name_edit.text().strip(), self._expr_edit.text().strip()

    # ------------------------------------------------------------ internal

    def _insert_series_name(self, index) -> None:
        name = index.data()
        if not name.isidentifier():
            name = f'"{name}"'
        self._expr_edit.insert(name)
        self._expr_edit.setFocus()

    def _on_accept(self) -> None:
        name, expr = self.result_values()
        try:
            ignore = self._existing.name if self._existing else None
            self._project.check_custom(name, expr, ignore=ignore)
            # dry-run the evaluation too (checks unknown series etc.)
            expressions.evaluate(
                expr, lambda n: self._project.values(self._project.ref_by_name(n)),
                x=self._project.x_values())
        except (ProjectError, expressions.ExpressionError) as e:
            self._error_label.setText(str(e))
            self._error_label.show()
            return
        except KeyError as e:
            self._error_label.setText(f"Série não encontrada: {e.args[0]!r}")
            self._error_label.show()
            return
        self.accept()
