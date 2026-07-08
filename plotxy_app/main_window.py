"""Main window: toolbar, three-panel splitter, project ownership,
signal wiring and status bar."""

from __future__ import annotations

import os
import time

from PySide6.QtCore import QEvent, QPoint, QSettings, Qt
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSplitter, QToolBar,
)

from . import __version__
from .axis_panel import AxisPanel
from .cursor_menu import CursorMenu
from .data_model import DataLoadError, load_csv
from .goto_panel import GotoPanel
from .measures import MeasuresWindow
from .plot_area import PlotArea
from .project import INDEX_NAME, Project, ProjectError, SeriesRef
from .readout_panel import CursorReadout
from .series_dialog import SeriesDialog
from .series_panel import PanelGroup, PanelSeries, SeriesPanel
from .themes import THEMES, apply_app_theme


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"CSVPlotterXY {__version__}")
        self.resize(1280, 760)
        self._settings = QSettings("CSVPlotterXY", "CSVPlotterXY")
        self._project = Project()
        self._theme = THEMES.get(
            str(self._settings.value("theme", "dark")), THEMES["dark"])
        self.setAcceptDrops(True)  # drop CSVs anywhere on the window

        # --- menu bar (file-level actions live here, not on the toolbar).
        # Keep Python references to the menus: letting the wrappers be
        # garbage-collected deletes the underlying C++ QMenu (PySide6).
        self._file_menu = self.menuBar().addMenu("&Arquivo")
        self._file_menu.addAction("Abrir CSV…", self._open_file_dialog)
        self._recent_menu = self._file_menu.addMenu("Recentes")
        self._recent_menu.aboutToShow.connect(self._populate_recent_menu)
        self._file_menu.addSeparator()
        self._file_menu.addAction("Exportar imagem…", self._export_image_dialog)
        self._file_menu.addAction("Copiar imagem", self._copy_image)

        # --- toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_btn = QPushButton("Adicionar CSV")
        open_btn.clicked.connect(self._open_file_dialog)
        toolbar.addWidget(open_btn)

        new_series_btn = QPushButton("Nova série")
        new_series_btn.clicked.connect(self._on_new_series)
        toolbar.addWidget(new_series_btn)

        autorange_btn = QPushButton("Ajustar zoom")
        autorange_btn.clicked.connect(lambda: self._plot.autorange())
        toolbar.addWidget(autorange_btn)

        self._zoom_btn = QPushButton("Zoom local")
        self._zoom_btn.setCheckable(True)
        self._zoom_btn.toggled.connect(self._plot_zoom_toggled)
        toolbar.addWidget(self._zoom_btn)

        self._goto_btn = QPushButton("Ir para")
        self._goto_btn.clicked.connect(self._open_goto_popup)
        toolbar.addWidget(self._goto_btn)

        self._cursors_btn = QPushButton("Cursores")
        self._cursors_btn.clicked.connect(self._open_cursor_popup)
        toolbar.addWidget(self._cursors_btn)

        self._scale_btn = QPushButton("Janela")
        self._scale_btn.clicked.connect(self._open_axis_popup)
        toolbar.addWidget(self._scale_btn)

        measures_btn = QPushButton("Medidas")
        measures_btn.clicked.connect(self._open_measures)
        toolbar.addWidget(measures_btn)

        self._theme_btn = QPushButton()
        self._theme_btn.clicked.connect(self._toggle_theme)
        toolbar.addWidget(self._theme_btn)

        self._file_label = QLabel("Nenhum arquivo carregado")
        toolbar.addWidget(self._file_label)

        # --- central splitter
        self._panel = SeriesPanel()
        self._plot = PlotArea()
        # one readout section per cursor, stacked vertically on the right
        self._v_readout = CursorReadout("Cursor vertical", "X", "Y")
        self._h_readout = CursorReadout("Cursor horizontal", "Y", "X")

        readouts = QSplitter(Qt.Orientation.Vertical)
        readouts.addWidget(self._v_readout)
        readouts.addWidget(self._h_readout)
        self._h_readout.hide()  # horizontal cursor starts disabled

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._panel)
        splitter.addWidget(self._plot)
        splitter.addWidget(readouts)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([280, 760, 230])
        splitter.setCollapsible(1, False)
        self._splitter = splitter  # for the readouts' auto-width requests
        self.setCentralWidget(splitter)

        # axis-scale controls live in a dropdown popup (opened from toolbar)
        self._axis = AxisPanel()
        self._axis.setWindowFlags(Qt.WindowType.Popup)

        # cursor toggles live in a dropdown popup as well
        self._cursor_menu = CursorMenu()
        self._cursor_menu.setWindowFlags(Qt.WindowType.Popup)
        self._cursor_menu.cursors_changed.connect(self._on_cursors_changed)

        # "Ir para" popup: position cursors by value
        self._goto = GotoPanel()
        self._goto.setWindowFlags(Qt.WindowType.Popup)
        self._goto.set_enabled_states(*self._cursor_menu.states())
        self._goto.goto_requested.connect(self._on_goto_requested)

        # Medidas window (lazy) + measures cache
        self._measures: MeasuresWindow | None = None
        self._measures_cache_key = None

        # toolbar dropdowns work as toggles: a Qt.Popup closes on the
        # press that lands on its own button, so an event filter records
        # each hide time and the toggle handler skips the immediate reopen
        self._popups = (self._axis, self._cursor_menu, self._goto)
        self._popup_hidden_at: dict[int, float] = {}
        for w in self._popups:
            w.installEventFilter(self)

        # --- status bar
        self._status_info = QLabel("")
        self.statusBar().addWidget(self._status_info)

        # --- wiring
        self._panel.selection_changed.connect(self._on_selection_changed)
        self._panel.remove_file_requested.connect(self._on_remove_file)
        self._panel.new_series_requested.connect(self._on_new_series)
        self._panel.edit_series_requested.connect(self._on_edit_series)
        self._panel.delete_series_requested.connect(self._on_delete_series)
        self._plot.v_cursor_moved.connect(self._v_readout.update_values)
        self._plot.h_cursor_moved.connect(self._h_readout.update_values)
        self._v_readout.color_change_requested.connect(self._plot.prompt_color)
        self._h_readout.color_change_requested.connect(self._plot.prompt_color)
        self._v_readout.point_clicked.connect(self._plot.show_point_tooltip)
        self._h_readout.point_clicked.connect(self._plot.show_point_tooltip)
        self._v_readout.goto_point.connect(self._on_goto_point)
        self._h_readout.goto_point.connect(self._on_goto_point)
        self._v_readout.width_hint_changed.connect(self._on_readout_width_hint)
        self._h_readout.width_hint_changed.connect(self._on_readout_width_hint)
        self._cursor_menu.interpolation_changed.connect(
            self._plot.set_click_interpolation)
        self._cursor_menu.snap_changed.connect(self._plot.set_cursor_snap)
        self._plot.cursors_enabled_changed.connect(self._cursor_menu.set_states)
        self._plot.measure_region_changed.connect(self._update_measures)
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
        self._add_recent_file(path)

        if dataset.dropped_columns:
            shown = ", ".join(dataset.dropped_columns[:5])
            more = len(dataset.dropped_columns) - 5
            self.statusBar().showMessage(
                f"Colunas não numéricas descartadas: {shown}"
                + (f" (+{more})" if more > 0 else ""), 6000)

        if first_file:
            # default: X = row index, first data column pre-checked as Y
            x_fallback = SeriesRef("index", file_id, INDEX_NAME)
            initial_y = SeriesRef("column", file_id, dataset.names[0])
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
                                  tooltip="index — número da linha (0, 1, 2, …)")]
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

    # ------------------------------------------------------- recent files

    def _recent_files(self) -> list[str]:
        val = self._settings.value("recent_files", [])
        if isinstance(val, str):   # QSettings collapses 1-item lists
            val = [val]
        return list(val or [])

    def _add_recent_file(self, path: str) -> None:
        path = os.path.abspath(path)
        recent = [p for p in self._recent_files()
                  if os.path.normcase(p) != os.path.normcase(path)]
        recent.insert(0, path)
        self._settings.setValue("recent_files", recent[:10])

    def _populate_recent_menu(self) -> None:
        self._recent_menu.clear()
        recent = self._recent_files()
        if not recent:
            action = self._recent_menu.addAction("(vazio)")
            action.setEnabled(False)
            return
        for path in recent:
            action = self._recent_menu.addAction(os.path.basename(path))
            action.setToolTip(path)
            action.setEnabled(os.path.exists(path))
            action.triggered.connect(
                lambda checked=False, p=path: self.open_path(p))

    # --------------------------------------------------------- drag & drop

    def dragEnterEvent(self, event) -> None:
        if any(u.isLocalFile() and u.toLocalFile().lower().endswith(".csv")
               for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if path.lower().endswith(".csv"):
                    self.open_path(path)
        event.acceptProposedAction()

    # ------------------------------------------------------- image export

    def _export_image_dialog(self) -> None:
        if not self._plot._curves:
            QMessageBox.information(self, "Exportar imagem",
                                    "Plote pelo menos uma série primeiro.")
            return
        last_dir = str(self._settings.value("last_dir", ""))
        path, selected = QFileDialog.getSaveFileName(
            self, "Exportar imagem", os.path.join(last_dir, "grafico.png"),
            "Imagem PNG (*.png);;Imagem SVG (*.svg)")
        if not path:
            return
        if not path.lower().endswith((".png", ".svg")):
            path += ".svg" if "SVG" in selected else ".png"
        try:
            self._plot.export_image(path)
        except Exception as e:  # exporter errors (permissions, disk…)
            QMessageBox.critical(self, "Exportar imagem",
                                 f"Falha ao exportar: {e}")
            return
        self.statusBar().showMessage(f"Imagem exportada: {path}", 5000)

    def _copy_image(self) -> None:
        if not self._plot._curves:
            QMessageBox.information(self, "Copiar imagem",
                                    "Plote pelo menos uma série primeiro.")
            return
        self._plot.copy_image()
        self.statusBar().showMessage(
            "Imagem copiada para a área de transferência", 4000)

    def _on_selection_changed(self, x_ref, y_refs: list) -> None:
        # keep the project's X axis current so D() snapshots the right axis
        self._project.set_x_axis(x_ref)
        if x_ref is None or not self._project.files():
            self._plot.clear()
            return
        rp = self._project.resolve_plot(x_ref, list(y_refs))
        self._plot.set_series(rp.x_key, rp.x_label, rp.x, rp.series)
        if rp.truncated:
            self.statusBar().showMessage(
                f"Séries com tamanhos diferentes — usando os primeiros "
                f"{rp.n_points} pontos.", 6000)
        # active series changed: refresh the measures table for the
        # current interval (the cache key includes the curve set)
        if (self._measures is not None and self._measures.isVisible()
                and self._plot._measure_region.isVisible()):
            lo, hi = self._plot._measure_region.getRegion()
            self._update_measures(float(lo), float(hi))

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

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Hide and obj in self._popups:
            self._popup_hidden_at[id(obj)] = time.monotonic()
        return super().eventFilter(obj, event)

    def _toggle_popup(self, widget, button, prepare=None) -> None:
        """Open the dropdown, or close it if this click just dismissed it
        (or it is still visible) — a consistent toggle."""
        if widget.isVisible():
            widget.hide()
            return
        if time.monotonic() - self._popup_hidden_at.get(id(widget), 0.0) < 0.25:
            return
        if prepare is not None and prepare() is False:
            return
        pos = button.mapToGlobal(QPoint(0, button.height()))
        widget.move(pos)
        widget.show()
        widget.raise_()

    def _open_axis_popup(self) -> None:
        def prepare():
            self._axis.set_ranges(*self._plot.view_ranges(), force=True)
        self._toggle_popup(self._axis, self._scale_btn, prepare)

    def _open_cursor_popup(self) -> None:
        self._toggle_popup(self._cursor_menu, self._cursors_btn)

    def _on_cursors_changed(self, vertical: bool, horizontal: bool) -> None:
        # panels first, so the plot's re-emit isn't dropped by the
        # readouts' hidden-panel early-out
        self._v_readout.setVisible(vertical)
        self._h_readout.setVisible(horizontal)
        self._plot.set_cursor_visible("v", vertical)
        self._plot.set_cursor_visible("h", horizontal)
        # "Ir para" fields stay visible; only their enabled state follows
        self._goto.set_enabled_states(vertical, horizontal)

    def _on_axis_range_changed(self, xmin, xmax, ymin, ymax) -> None:
        self._plot.set_manual_ranges(xmin, xmax, ymin, ymax)

    # ------------------------------------------------------------ measures

    def _open_measures(self) -> None:
        if not self._plot._curves:
            QMessageBox.information(
                self, "Medidas", "Plote pelo menos uma série primeiro.")
            return
        if self._measures is None:
            self._measures = MeasuresWindow(self)
            self._measures.visibility_changed.connect(
                self._on_measures_visibility)
            self._measures.point_activated.connect(self._plot.show_point_tooltip)
            self._measures.goto_x_requested.connect(self._on_measures_goto)
            self._measures.interval_edited.connect(self._plot.set_measure_region)
            self._plot.measure_region_changing.connect(
                self._on_measure_region_changing)
        self._measures.show()
        self._measures.raise_()
        self._measures.activateWindow()

    def _on_measures_visibility(self, active: bool) -> None:
        self._plot.set_measure_region_visible(active)

    def _on_measures_goto(self, x: float) -> None:
        self._plot.set_cursor_x(x)
        self.statusBar().showMessage(f"Cursor em X = {x:.6g}", 4000)

    def _on_goto_point(self, key: str, x: float, y: float) -> None:
        self._plot.focus_on_point(key, x, y)
        self.statusBar().showMessage(
            f"Vista centralizada em ({x:.6g}, {y:.6g})", 4000)

    def _on_readout_width_hint(self, hint: int) -> None:
        """Grow (never shrink) the readout pane so its content fits without
        a horizontal scrollbar. Capped at 40% of the window, keeping at
        least 300 px for the plot."""
        sizes = self._splitter.sizes()
        if len(sizes) != 3 or hint <= sizes[2]:
            return
        total = sum(sizes)
        new_width = min(hint, int(total * 0.4))
        delta = new_width - sizes[2]
        if delta <= 0 or sizes[1] - delta < 300:
            return
        self._splitter.setSizes([sizes[0], sizes[1] - delta, new_width])

    def _on_measure_region_changing(self, lo: float, hi: float) -> None:
        # live A/B field sync while the region is being dragged
        if self._measures is not None and self._measures.isVisible():
            self._measures.set_interval(lo, hi)

    def _update_measures(self, lo: float, hi: float) -> None:
        """Recompute the measures table only when the interval (or the
        active series set) actually changed — cached otherwise."""
        if self._measures is None or not self._measures.isVisible():
            return
        key = (round(lo, 12), round(hi, 12), self._plot._x_key,
               frozenset(self._plot._curves))
        if key == self._measures_cache_key:
            return
        self._measures_cache_key = key
        self._measures.set_rows(lo, hi, self._plot.measures_rows(lo, hi))

    def _open_goto_popup(self) -> None:
        def prepare():
            if self._plot.x_range() is None:
                QMessageBox.information(
                    self, "Ir para", "Plote uma série primeiro.")
                return False
            cx, cy = self._plot.cursor_positions()
            self._goto.set_positions(cx, cy)
            self._goto.clear_error()
        self._toggle_popup(self._goto, self._goto_btn, prepare)

    def _on_goto_requested(self, x, y) -> None:
        # validate both enabled fields against the data ranges before
        # moving anything; errors are shown inline in the popup
        if x is not None:
            rng = self._plot.x_range()
            if rng is None:
                return
            if not (rng[0] <= x <= rng[1]):
                self._goto.show_error(
                    f"X fora do intervalo [{rng[0]:.6g}, {rng[1]:.6g}].")
                return
        if y is not None:
            yrng = self._plot.y_data_range()
            if yrng is None:
                return
            if not (yrng[0] <= y <= yrng[1]):
                self._goto.show_error(
                    f"Y fora do intervalo [{yrng[0]:.6g}, {yrng[1]:.6g}].")
                return
        moved = []
        if x is not None:
            self._plot.set_cursor_x(x)
            moved.append(f"X = {x:.6g}")
        if y is not None:
            self._plot.set_cursor_y(y)
            moved.append(f"Y = {y:.6g}")
        if moved:
            # keep the popup open so the other cursor can also be moved
            self.statusBar().showMessage(
                "Cursor posicionado em " + ", ".join(moved), 4000)

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
