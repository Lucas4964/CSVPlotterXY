"""Main window: toolbar, three-panel splitter, project ownership,
signal wiring and status bar."""

from __future__ import annotations

import os

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QInputDialog, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSplitter, QToolBar, QVBoxLayout, QWidget,
)

from . import __version__
from .axis_panel import AxisPanel
from .data_model import DataLoadError, load_csv
from .plot_area import PlotArea
from .project import INDEX_NAME, Project, ProjectError, SeriesRef
from .readout_panel import ReadoutPanel
from .series_dialog import SeriesDialog
from .series_panel import PanelGroup, PanelSeries, SeriesPanel
from .themes import THEMES, apply_app_theme


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PlotXY-Py {__version__}")
        self.resize(1280, 760)
        self._settings = QSettings("PlotXYPy", "PlotXYPy")
        self._project = Project()
        self._theme = THEMES.get(
            str(self._settings.value("theme", "dark")), THEMES["dark"])

        # --- toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_btn = QPushButton("Adicionar CSV…")
        open_btn.clicked.connect(self._open_file_dialog)
        toolbar.addWidget(open_btn)

        new_series_btn = QPushButton("Nova série…")
        new_series_btn.clicked.connect(self._on_new_series)
        toolbar.addWidget(new_series_btn)

        autorange_btn = QPushButton("Ajustar zoom")
        autorange_btn.clicked.connect(lambda: self._plot.autorange())
        toolbar.addWidget(autorange_btn)

        self._zoom_btn = QPushButton("Zoom local")
        self._zoom_btn.setCheckable(True)
        self._zoom_btn.toggled.connect(self._plot_zoom_toggled)
        toolbar.addWidget(self._zoom_btn)

        goto_btn = QPushButton("Ir para X")
        goto_btn.clicked.connect(self._on_goto_x)
        toolbar.addWidget(goto_btn)

        self._theme_btn = QPushButton()
        self._theme_btn.clicked.connect(self._toggle_theme)
        toolbar.addWidget(self._theme_btn)

        self._file_label = QLabel("Nenhum arquivo carregado")
        toolbar.addWidget(self._file_label)

        # --- central splitter
        self._panel = SeriesPanel()
        self._plot = PlotArea()
        self._readout = ReadoutPanel()
        self._axis = AxisPanel()

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self._readout, stretch=1)
        right_layout.addWidget(self._axis)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._panel)
        splitter.addWidget(self._plot)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([280, 760, 240])
        splitter.setCollapsible(1, False)
        self.setCentralWidget(splitter)

        # --- status bar
        self._status_info = QLabel("")
        self.statusBar().addWidget(self._status_info)

        # --- wiring
        self._panel.selection_changed.connect(self._on_selection_changed)
        self._panel.remove_file_requested.connect(self._on_remove_file)
        self._panel.new_series_requested.connect(self._on_new_series)
        self._panel.edit_series_requested.connect(self._on_edit_series)
        self._panel.delete_series_requested.connect(self._on_delete_series)
        self._plot.cursor_moved.connect(self._readout.update_values)
        self._axis.range_changed.connect(self._on_axis_range_changed)
        self._axis.auto_requested.connect(lambda: self._plot.autorange())
        self._plot.view_range_changed.connect(self._axis.set_ranges)

        self._apply_theme()

    # ------------------------------------------------------------- public

    def open_path(self, path: str) -> None:
        try:
            dataset = load_csv(path)
        except DataLoadError as e:
            QMessageBox.critical(self, "Erro ao abrir arquivo", str(e))
            return
        first_file = not self._project.files()
        file_id = self._project.add_file(dataset)
        self._settings.setValue("last_dir", os.path.dirname(os.path.abspath(path)))

        if dataset.dropped_columns:
            shown = ", ".join(dataset.dropped_columns[:5])
            more = len(dataset.dropped_columns) - 5
            self.statusBar().showMessage(
                f"Colunas não numéricas descartadas: {shown}"
                + (f" (+{more})" if more > 0 else ""), 6000)

        if first_file:
            # v1 default: X = first column, second column pre-checked
            x_fallback = SeriesRef("column", file_id, dataset.names[0])
            initial_y = (SeriesRef("column", file_id, dataset.names[1])
                         if dataset.n_cols > 1 else None)
            self._refresh_panel(x_fallback=x_fallback, pre_check=initial_y)
        else:
            self._refresh_panel()

    # ------------------------------------------------------------ internal

    def _refresh_panel(self, x_fallback: SeriesRef | None = None,
                       pre_check: SeriesRef | None = None) -> None:
        groups: list[PanelGroup] = []
        for file_id, ds in self._project.files():
            base = os.path.basename(ds.source_path)
            series = [PanelSeries(ref=SeriesRef("index", file_id, INDEX_NAME),
                                  plain_name=INDEX_NAME, is_index=True,
                                  tooltip="Índice das linhas (0, 1, 2, …)")]
            series += [PanelSeries(ref=SeriesRef("column", file_id, n),
                                   plain_name=n) for n in ds.names]
            groups.append(PanelGroup(file_id=file_id, title=base,
                                     tooltip=ds.source_path, series=series))
        custom_series = [
            PanelSeries(ref=SeriesRef("custom", "", c.name),
                        plain_name=c.name,
                        tooltip=f"{c.name} = {c.expression}")
            for c in self._project.custom()]
        groups.append(PanelGroup(file_id=None, title="Séries personalizadas",
                                 tooltip="", series=custom_series))

        self._panel.set_project(groups, x_fallback=x_fallback,
                                label_of=self._project.label)
        if pre_check is not None:
            self._panel.check_ref(pre_check)

        n_files = len(self._project.files())
        self._file_label.setText(
            "Nenhum arquivo carregado" if n_files == 0
            else f"{n_files} arquivo(s) carregado(s)")
        total_series = sum(len(g.series) for g in groups)
        self._status_info.setText(
            f"{total_series} séries disponíveis" if total_series else "")

    def _open_file_dialog(self) -> None:
        last_dir = str(self._settings.value("last_dir", ""))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Adicionar arquivos CSV", last_dir,
            "Arquivos CSV (*.csv);;Todos (*)")
        for path in paths:
            self.open_path(path)

    def _on_selection_changed(self, x_ref, y_refs: list) -> None:
        if x_ref is None or not self._project.files():
            self._plot.clear()
            return
        rp = self._project.resolve_plot(x_ref, list(y_refs))
        self._plot.set_series(rp.x_key, rp.x_label, rp.x, rp.series)
        if rp.truncated:
            self.statusBar().showMessage(
                f"Séries com tamanhos diferentes — usando os primeiros "
                f"{rp.n_points} pontos.", 6000)

    def _on_remove_file(self, file_id: str) -> None:
        deps = self._project.dependents_on_file(file_id)
        ds = dict(self._project.files()).get(file_id)
        base = os.path.basename(ds.source_path) if ds else file_id
        msg = f'Remover o arquivo "{base}"?'
        if deps:
            msg += ("\n\nAs séries personalizadas que dependem dele também "
                    f"serão removidas: {', '.join(deps)}.")
        if QMessageBox.question(self, "Remover arquivo", msg) \
                != QMessageBox.StandardButton.Yes:
            return
        removed = self._project.remove_file(file_id)
        remaining = self._project.files()
        x_fallback = (SeriesRef("index", remaining[0][0], INDEX_NAME)
                      if remaining else None)
        self._refresh_panel(x_fallback=x_fallback)
        if not remaining:
            self._plot.clear()
        if removed:
            self.statusBar().showMessage(
                f"Séries personalizadas removidas: {', '.join(removed)}", 6000)

    def _on_new_series(self) -> None:
        if not self._project.files():
            QMessageBox.information(
                self, "Nova série", "Adicione um arquivo CSV primeiro.")
            return
        dialog = SeriesDialog(self._project, parent=self)
        if dialog.exec() == SeriesDialog.DialogCode.Accepted:
            name, expr = dialog.result_values()
            try:
                _, truncated = self._project.add_custom(name, expr)
            except ProjectError as e:
                QMessageBox.critical(self, "Nova série", str(e))
                return
            self._refresh_panel()
            if truncated:
                self.statusBar().showMessage(
                    "Séries com tamanhos diferentes na expressão — "
                    "truncadas ao menor comprimento.", 6000)

    def _on_edit_series(self, name: str) -> None:
        existing = next((c for c in self._project.custom() if c.name == name),
                        None)
        if existing is None:
            return
        dialog = SeriesDialog(self._project, existing=existing, parent=self)
        if dialog.exec() == SeriesDialog.DialogCode.Accepted:
            new_name, expr = dialog.result_values()
            try:
                recomputed = self._project.edit_custom(name, new_name, expr)
            except ProjectError as e:
                QMessageBox.critical(self, "Editar série", str(e))
                return
            self._refresh_panel()
            if recomputed:
                self.statusBar().showMessage(
                    f"Séries recalculadas: {', '.join(recomputed)}", 6000)

    def _on_delete_series(self, name: str) -> None:
        deps = self._project.dependents_on_custom(name)
        msg = f'Excluir a série "{name}"?'
        if deps:
            msg += ("\n\nAs séries que dependem dela também serão "
                    f"excluídas: {', '.join(deps)}.")
        if QMessageBox.question(self, "Excluir série", msg) \
                != QMessageBox.StandardButton.Yes:
            return
        self._project.remove_custom(name)
        self._refresh_panel()

    def _plot_zoom_toggled(self, checked: bool) -> None:
        self._plot.set_zoom_visible(checked)

    def _on_axis_range_changed(self, xmin, xmax, ymin, ymax) -> None:
        self._plot.set_x_range(xmin, xmax)
        self._plot.set_y_range(ymin, ymax)

    def _on_goto_x(self) -> None:
        rng = self._plot.x_range()
        if rng is None:
            QMessageBox.information(
                self, "Ir para X", "Plote uma série primeiro.")
            return
        xmin, xmax = rng
        text, ok = QInputDialog.getText(
            self, "Ir para X",
            f"Valor de X (entre {xmin:.6g} e {xmax:.6g}):")
        if not ok:
            return
        try:
            value = float(text.strip().replace(",", "."))
        except ValueError:
            QMessageBox.warning(self, "Ir para X",
                                f'Valor inválido: "{text}".')
            return
        if not (xmin <= value <= xmax):
            QMessageBox.warning(
                self, "Ir para X",
                f"O valor {value:.6g} está fora do intervalo "
                f"[{xmin:.6g}, {xmax:.6g}].")
            return
        self._plot.set_cursor_x(value)
        self.statusBar().showMessage(f"Cursor posicionado em X = {value:.6g}", 4000)

    def _toggle_theme(self) -> None:
        self._theme = THEMES["light" if self._theme.name == "dark" else "dark"]
        self._settings.setValue("theme", self._theme.name)
        self._apply_theme()

    def _apply_theme(self) -> None:
        app = QApplication.instance()
        apply_app_theme(app, self._theme)
        self._plot.apply_theme(self._theme)
        self._theme_btn.setText(
            "☀ Tema claro" if self._theme.name == "dark" else "🌙 Tema escuro")
