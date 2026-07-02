"""Offscreen GUI smoke test for the multi-file + expressions + zoom flow."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from plotxy_app.main_window import MainWindow
from plotxy_app.project import INDEX_NAME, SeriesRef


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def csv_a(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text('"time","P1","P2"\n' + "\n".join(
        f"{i * 0.1},{i},{i * 2}" for i in range(10)))
    return str(p)


@pytest.fixture
def csv_b(tmp_path):
    p = tmp_path / "b.csv"  # shares "time" name, shorter (6 rows)
    p.write_text('"time","Q"\n' + "\n".join(
        f"{i * 0.2},{i * 10}" for i in range(6)))
    return str(p)


@pytest.fixture
def win(app, csv_a, csv_b):
    w = MainWindow()
    w.show()
    w.open_path(csv_a)
    w.open_path(csv_b)
    app.processEvents()
    return w


def find_item(win, ref):
    for item in win._panel._iter_series_items():
        if item.data(Qt.ItemDataRole.UserRole + 1) == ref:
            return item
    return None


def file_ids(win):
    return [fid for fid, _ in win._project.files()]


def test_two_files_tree_and_labels(win, app):
    f1, f2 = file_ids(win)
    # tree: 2 file groups + custom group
    assert win._panel._model.rowCount() == 3
    assert win._panel._model.item(0).text() == "a.csv"
    assert win._panel._model.item(1).text() == "b.csv"
    assert win._panel._model.item(2).text() == "Séries personalizadas"
    # "time" is ambiguous -> qualified labels in X combo
    labels = [win._panel._x_combo.itemText(i)
              for i in range(win._panel._x_combo.count())]
    assert "time (a.csv)" in labels and "time (b.csv)" in labels
    # default selection survived the second file load: X=time(a), P1 checked
    assert win._panel.x_ref() == SeriesRef("column", f1, "time")
    assert win._panel.y_refs() == [SeriesRef("column", f1, "P1")]
    assert len(win._plot._curves) == 1


def test_cross_file_truncation(win, app):
    f1, f2 = file_ids(win)
    win._panel.check_ref(SeriesRef("column", f2, "Q"))  # 6 rows vs X 10 rows
    app.processEvents()
    assert len(win._plot._curves) == 2
    # all plotted arrays truncated to 6 points
    assert all(len(y) == 6 for y in win._plot._ys.values())
    assert len(win._plot._x) == 6


def test_index_series_as_x(win, app):
    f1, _ = file_ids(win)
    win._panel.set_x_ref(SeriesRef("index", f1, INDEX_NAME))
    app.processEvents()
    assert win._plot._x_kind == 0  # increasing
    assert np.allclose(win._plot._x, np.arange(10))


def test_custom_series_flow(win, app):
    f1, f2 = file_ids(win)
    _, truncated = win._project.add_custom("total", "P1 + P2")
    assert not truncated
    win._refresh_panel()
    app.processEvents()
    ref = SeriesRef("custom", "", "total")
    assert find_item(win, ref) is not None
    win._panel.check_ref(ref)
    app.processEvents()
    assert any(k == ref.key() for k in win._plot._curves)
    # custom series values correct at cursor
    win._plot._cursor.setValue(0.45)
    app.processEvents()
    rows = {win._readout._table.item(r, 0).text():
            win._readout._table.item(r, 1).text()
            for r in range(win._readout._table.rowCount())}
    t = np.arange(10) * 0.1
    expected = float(np.interp(0.45, t, np.arange(10) * 3.0))
    assert abs(float(rows["total"]) - expected) < 1e-4


def test_remove_file_cascade_and_x_fallback(win, app):
    f1, f2 = file_ids(win)
    win._project.add_custom("dobro", "P1 * 2")  # depends on f1
    win._refresh_panel()
    app.processEvents()
    assert win._project.dependents_on_file(f1) == ["dobro"]
    # remove f1 directly through the project + refresh (skip the QMessageBox)
    win._project.remove_file(f1)
    x_fallback = SeriesRef("index", f2, INDEX_NAME)
    win._refresh_panel(x_fallback=x_fallback)
    app.processEvents()
    assert [c.name for c in win._project.custom()] == []
    assert win._panel.x_ref() == x_fallback  # old X died -> fallback
    assert win._panel._model.rowCount() == 2  # b.csv + custom group
    # plot still consistent (no curves from removed file)
    for key in win._plot._curves:
        assert "|f1|" not in f"|{key}|"


def test_edit_custom_updates_curve(win, app):
    f1, _ = file_ids(win)
    win._project.add_custom("c1", "P1 * 2")
    win._refresh_panel()
    app.processEvents()
    ref = SeriesRef("custom", "", "c1")
    win._panel.check_ref(ref)
    app.processEvents()
    win._project.edit_custom("c1", "c1", "P1 * 100")
    win._refresh_panel()
    app.processEvents()
    y = win._plot._ys[ref.key()]
    assert np.allclose(y, np.arange(10) * 100.0)


def test_zoom_panel_sync(win, app):
    assert not win._plot._zoom_pw.isVisible()
    win._zoom_btn.setChecked(True)
    app.processEvents()
    assert win._plot._region.isVisible()
    # dragging region updates zoom panel range
    win._plot._region.setRegion((0.2, 0.5))
    app.processEvents()
    lo, hi = win._plot._zoom_item.getViewBox().viewRange()[0]
    assert abs(lo - 0.2) < 1e-6 and abs(hi - 0.5) < 1e-6
    # zoom panel pan drags the region back
    win._plot._zoom_item.getViewBox().setXRange(0.3, 0.6, padding=0)
    app.processEvents()
    rlo, rhi = win._plot._region.getRegion()
    assert abs(rlo - 0.3) < 1e-6 and abs(rhi - 0.6) < 1e-6
    win._zoom_btn.setChecked(False)
    app.processEvents()
    assert not win._plot._zoom_pw.isVisible()


def test_theme_toggle_with_everything_active(win, app):
    win._zoom_btn.setChecked(True)
    win._project.add_custom("temp", "P1 + 1")
    win._refresh_panel()
    win._panel.check_ref(SeriesRef("custom", "", "temp"))
    app.processEvents()
    n_curves = len(win._plot._curves)
    for _ in range(3):
        win._toggle_theme()
        app.processEvents()
    assert len(win._plot._curves) == n_curves
    assert len(win._plot._zoom_curves) == n_curves


def test_remove_all_files_clears_plot(win, app):
    for fid in file_ids(win):
        win._project.remove_file(fid)
    win._refresh_panel(x_fallback=None)
    win._plot.clear()
    app.processEvents()
    assert win._panel.x_ref() is None
    assert len(win._plot._curves) == 0
    assert not win._plot._cursor.isVisible()
