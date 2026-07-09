"""Series browser: tree of files/series with checkboxes, X-axis combo,
filter box and context menus. Replaces the v1 flat column panel."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSortFilterProxyModel, Qt, QTimer, Signal
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton, QTreeView,
    QVBoxLayout, QWidget,
)

from .project import SeriesRef

_REF_ROLE = Qt.ItemDataRole.UserRole + 1
_FILE_ROLE = Qt.ItemDataRole.UserRole + 2
_CUSTOM_GROUP_ROLE = Qt.ItemDataRole.UserRole + 3


@dataclass(frozen=True)
class PanelSeries:
    ref: SeriesRef
    plain_name: str
    is_index: bool = False
    tooltip: str = ""


@dataclass(frozen=True)
class PanelGroup:
    file_id: str | None      # None for the custom-series group
    title: str
    tooltip: str
    series: list[PanelSeries]


class SeriesPanel(QWidget):
    """Emits selection_changed(x_ref | None, [y_refs]) debounced.
    Uses the _updating guard pattern (never model.blockSignals — it
    breaks proxy/view sync)."""

    selection_changed = Signal(object, list)
    remove_file_requested = Signal(str)
    new_series_requested = Signal()
    edit_series_requested = Signal(str)
    delete_series_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._updating = False
        self._emit_timer = QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(0)
        self._emit_timer.timeout.connect(self._emit_selection)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(QLabel("Eixo X"))
        self._x_combo = QComboBox()
        self._x_combo.currentIndexChanged.connect(self._on_x_changed)
        layout.addWidget(self._x_combo)

        layout.addWidget(QLabel("Séries do eixo Y"))
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filtrar séries…")
        self._filter_edit.setClearButtonEnabled(True)
        layout.addWidget(self._filter_edit)

        self._model = QStandardItemModel(self)
        self._model.itemChanged.connect(self._on_item_changed)
        self._proxy = QSortFilterProxyModel(self)
        self._proxy.setSourceModel(self._model)
        self._proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._proxy.setRecursiveFilteringEnabled(True)
        self._filter_edit.textChanged.connect(self._on_filter_changed)

        self._tree = QTreeView()
        self._tree.setModel(self._proxy)
        self._tree.setHeaderHidden(True)
        self._tree.setEditTriggers(QTreeView.EditTrigger.NoEditTriggers)
        self._tree.clicked.connect(self._on_item_clicked)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._tree, stretch=1)

        buttons = QHBoxLayout()
        new_btn = QPushButton("Nova série…")
        new_btn.clicked.connect(self.new_series_requested)
        buttons.addWidget(new_btn)
        deselect_btn = QPushButton("Desmarcar tudo")
        deselect_btn.setToolTip("Desmarca todas as séries do gráfico "
                                "(mantém os arquivos e as séries "
                                "personalizadas).")
        deselect_btn.clicked.connect(self._deselect_all)
        buttons.addWidget(deselect_btn)
        layout.addLayout(buttons)

    # ------------------------------------------------------------- public

    def set_project(self, groups: list[PanelGroup],
                    x_fallback: SeriesRef | None = None,
                    label_of=None) -> None:
        """Rebuild tree + X combo, restoring checks and X by surviving
        SeriesRef. `label_of(ref) -> str` provides qualified combo labels."""
        old_checked = set(self._checked_refs())
        old_x = self.x_ref()

        self._updating = True
        try:
            self._model.clear()
            all_refs: list[SeriesRef] = []
            for group in groups:
                g_item = QStandardItem(group.title)
                g_item.setEditable(False)
                g_item.setSelectable(False)
                font = g_item.font(); font.setBold(True); g_item.setFont(font)
                if group.tooltip:
                    g_item.setToolTip(group.tooltip)
                if group.file_id is not None:
                    g_item.setData(group.file_id, _FILE_ROLE)
                else:
                    g_item.setData(True, _CUSTOM_GROUP_ROLE)
                for s in group.series:
                    item = QStandardItem(s.plain_name)
                    item.setEditable(False)
                    item.setCheckable(True)
                    item.setData(s.ref, _REF_ROLE)
                    if s.tooltip:
                        item.setToolTip(s.tooltip)
                    if s.is_index:
                        f = item.font(); f.setItalic(True); item.setFont(f)
                    if s.ref in old_checked:
                        item.setCheckState(Qt.CheckState.Checked)
                    g_item.appendRow(item)
                    all_refs.append(s.ref)
                self._model.appendRow(g_item)

            self._x_combo.clear()
            for ref in all_refs:
                label = label_of(ref) if label_of else ref.name
                self._x_combo.addItem(label, userData=ref)
            # restore X: surviving old X > fallback > first ref
            target = None
            if old_x is not None and old_x in all_refs:
                target = old_x
            elif x_fallback is not None and x_fallback in all_refs:
                target = x_fallback
            if target is not None:
                self._x_combo.setCurrentIndex(all_refs.index(target))
            elif all_refs:
                self._x_combo.setCurrentIndex(0)
            self._sync_x_disable()
        finally:
            self._updating = False

        self._tree.expandAll()
        self._emit_timer.start()

    def x_ref(self) -> SeriesRef | None:
        return self._x_combo.currentData()

    def set_x_ref(self, ref: SeriesRef) -> None:
        for i in range(self._x_combo.count()):
            if self._x_combo.itemData(i) == ref:
                self._x_combo.setCurrentIndex(i)
                return

    def check_ref(self, ref: SeriesRef) -> None:
        for item in self._iter_series_items():
            if item.data(_REF_ROLE) == ref:
                item.setCheckState(Qt.CheckState.Checked)
                return

    def y_refs(self) -> list[SeriesRef]:
        out = []
        for item in self._iter_series_items():
            if (item.checkState() == Qt.CheckState.Checked and item.isEnabled()):
                out.append(item.data(_REF_ROLE))
        return out

    # ------------------------------------------------------------ internal

    def _iter_series_items(self):
        for g in range(self._model.rowCount()):
            group = self._model.item(g)
            for r in range(group.rowCount()):
                yield group.child(r)

    def _checked_refs(self) -> list[SeriesRef]:
        return [item.data(_REF_ROLE) for item in self._iter_series_items()
                if item.checkState() == Qt.CheckState.Checked]

    def _on_filter_changed(self, text: str) -> None:
        self._proxy.setFilterFixedString(text)
        self._tree.expandAll()

    def _on_item_changed(self, _item) -> None:
        if not self._updating:
            self._emit_timer.start()

    def _on_item_clicked(self, proxy_index) -> None:
        item = self._model.itemFromIndex(self._proxy.mapToSource(proxy_index))
        if item is None or item.data(_REF_ROLE) is None or not item.isEnabled():
            return
        new = (Qt.CheckState.Unchecked
               if item.checkState() == Qt.CheckState.Checked
               else Qt.CheckState.Checked)
        item.setCheckState(new)

    def _on_x_changed(self, _index: int) -> None:
        if self._updating:
            return
        self._sync_x_disable()
        self._emit_timer.start()

    def _sync_x_disable(self) -> None:
        x = self._x_combo.currentData()
        was = self._updating
        self._updating = True
        try:
            for item in self._iter_series_items():
                item.setEnabled(item.data(_REF_ROLE) != x)
        finally:
            self._updating = was

    def _deselect_all(self) -> None:
        """Uncheck every series (remove them all from the plot) without
        touching the loaded files or the custom series — they stay in the
        tree, ready to be re-checked."""
        self._updating = True
        try:
            for item in self._iter_series_items():
                item.setCheckState(Qt.CheckState.Unchecked)
        finally:
            self._updating = False
        self._emit_timer.start()

    def _emit_selection(self) -> None:
        self.selection_changed.emit(self.x_ref(), self.y_refs())

    def _on_context_menu(self, pos) -> None:
        proxy_index = self._tree.indexAt(pos)
        menu = QMenu(self)
        if proxy_index.isValid():
            item = self._model.itemFromIndex(self._proxy.mapToSource(proxy_index))
            file_id = item.data(_FILE_ROLE)
            ref: SeriesRef | None = item.data(_REF_ROLE)
            if file_id is not None:
                act = menu.addAction("Remover arquivo")
                act.triggered.connect(
                    lambda _=False, fid=file_id: self.remove_file_requested.emit(fid))
            elif item.data(_CUSTOM_GROUP_ROLE):
                menu.addAction("Nova série…").triggered.connect(
                    self.new_series_requested)
            elif ref is not None:
                act = menu.addAction("Usar como eixo X")
                act.triggered.connect(lambda _=False, r=ref: self.set_x_ref(r))
                if ref.kind == "custom":
                    menu.addSeparator()
                    menu.addAction("Editar…").triggered.connect(
                        lambda _=False, n=ref.name: self.edit_series_requested.emit(n))
                    menu.addAction("Excluir").triggered.connect(
                        lambda _=False, n=ref.name: self.delete_series_requested.emit(n))
        else:
            menu.addAction("Nova série…").triggered.connect(
                self.new_series_requested)
        if menu.actions():
            menu.exec(self._tree.viewport().mapToGlobal(pos))
