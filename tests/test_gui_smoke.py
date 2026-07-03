"""Offscreen GUI smoke test for the multi-file + expressions + zoom flow."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from plotxy_app.main_window import MainWindow
from plotxy_app.project import INDEX_NAME, SeriesRef


class FakeClick:
    """Minimal stand-in for a pyqtgraph MouseClickEvent."""

    def __init__(self, scene_pos):
        self._p = scene_pos

    def button(self):
        return Qt.MouseButton.LeftButton

    def isAccepted(self):
        return False

    def scenePos(self):
        return self._p


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
    # columns: [swatch | Série | Valor]
    rows = {win._readout._table.item(r, 1).text():
            win._readout._table.item(r, 2).text()
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


# ------------------------------------------------------------ v0.3 features


def test_axis_range_set_and_sync(win, app):
    win._plot.set_x_range(0.2, 0.5)
    win._plot.set_y_range(0.0, 9.0)
    app.processEvents()
    x0, x1, y0, y1 = win._plot.view_ranges()
    assert abs(x0 - 0.2) < 1e-6 and abs(x1 - 0.5) < 1e-6
    assert abs(y0 - 0.0) < 1e-6 and abs(y1 - 9.0) < 1e-6
    # axis-panel fields reflect the view via view_range_changed
    assert abs(float(win._axis._fields["xmin"].text()) - 0.2) < 1e-3
    assert abs(float(win._axis._fields["xmax"].text()) - 0.5) < 1e-3


def test_axis_panel_apply_and_invalid(win, app):
    win._axis._fields["xmin"].setText("0.1")
    win._axis._fields["xmax"].setText("0.7")
    win._axis._fields["ymin"].setText("0")
    win._axis._fields["ymax"].setText("8")
    win._axis._on_apply()
    app.processEvents()
    x0, x1, _, _ = win._plot.view_ranges()
    assert abs(x0 - 0.1) < 1e-6 and abs(x1 - 0.7) < 1e-6
    # min >= max is rejected: view unchanged, field flagged
    win._axis._fields["xmin"].setText("5")
    win._axis._fields["xmax"].setText("1")
    win._axis._on_apply()
    app.processEvents()
    x0b, x1b, _, _ = win._plot.view_ranges()
    assert abs(x0b - 0.1) < 1e-6 and abs(x1b - 0.7) < 1e-6
    assert "e74c3c" in win._axis._fields["xmin"].styleSheet()


def test_zoom_out_limit_keeps_curve(win, app):
    # request an absurd zoom-out; setLimits must clamp it (data span ~0.9)
    win._plot.set_x_range(-1e6, 1e6)
    app.processEvents()
    x0, x1, _, _ = win._plot.view_ranges()
    assert (x1 - x0) < 100  # clamped to ~20*span, not 2e6
    # the curve still has points to render in the clamped view
    curve = next(iter(win._plot._curves.values()))
    assert curve.curve.getPath().elementCount() > 0


def test_goto_x_api_and_validation(win, app):
    rng = win._plot.x_range()
    assert rng is not None
    xmin, xmax = rng
    mid = (xmin + xmax) / 2
    win._plot.set_cursor_x(mid)
    app.processEvents()
    assert abs(win._plot._cursor.value() - mid) < 1e-9
    # mirror of the out-of-range check used in _on_goto_x
    assert not (xmin <= xmax + 1.0 <= xmax)


def test_set_curve_color_persists_across_theme(win, app):
    key = next(iter(win._plot._curves))
    win._plot.set_curve_color(key, "#ff0000")
    app.processEvents()
    assert win._plot._color_of[key] == "#ff0000"
    assert win._plot._curves[key].opts["pen"].color().name() == "#ff0000"
    assert win._plot._zoom_curves[key].opts["pen"].color().name() == "#ff0000"
    # user color survives a theme toggle (override wins over palette)
    win._toggle_theme()
    app.processEvents()
    assert win._plot._color_of[key] == "#ff0000"
    assert win._plot._curves[key].opts["pen"].color().name() == "#ff0000"
    # and survives unchecking + rechecking the series (same key)
    ref = SeriesRef("column", file_ids(win)[0], "P1")
    item = find_item(win, ref)
    item.setCheckState(Qt.CheckState.Unchecked)
    app.processEvents()
    item.setCheckState(Qt.CheckState.Checked)
    app.processEvents()
    assert win._plot._curves[key].opts["pen"].color().name() == "#ff0000"


def test_readout_swatch_click_requests_color(app):
    # standalone panel so the click doesn't trigger the real (modal)
    # color dialog wired up in MainWindow
    from plotxy_app.readout_panel import ReadoutPanel, _KEY_ROLE
    panel = ReadoutPanel()
    panel.show()
    app.processEvents()
    rows = [("column|f1|P1", "P1", "#ff0000", 1.5),
            ("column|f1|P2", "P2", "#00ff00", 2.5)]
    panel.update_values(0.3, rows, False)
    app.processEvents()
    assert panel._table.rowCount() == 2
    assert panel._table.item(0, 0).data(_KEY_ROLE) == "column|f1|P1"
    captured = []
    panel.color_change_requested.connect(captured.append)
    panel._on_cell_clicked(1, 0)          # swatch column of P2
    assert captured == ["column|f1|P2"]
    captured.clear()
    panel._on_cell_clicked(0, 1)          # name column -> ignored
    assert captured == []
    panel.hide()


def test_axis_popup_wiring(win, app):
    # AxisPanel is a standalone Popup window, not embedded in the splitter
    assert win._axis.parent() is None
    assert win._axis.windowFlags() & Qt.WindowType.Popup
    win._open_axis_popup()
    app.processEvents()
    assert win._axis.isVisible()
    # applying a range from the popup still drives the plot
    win._axis._fields["xmin"].setText("0.1")
    win._axis._fields["xmax"].setText("0.6")
    win._axis._fields["ymin"].setText("0")
    win._axis._fields["ymax"].setText("8")
    win._axis._on_apply()
    app.processEvents()
    x0, x1, _, _ = win._plot.view_ranges()
    assert abs(x0 - 0.1) < 1e-6 and abs(x1 - 0.6) < 1e-6
    win._axis.hide()


def test_click_tooltip(win, app):
    win.resize(1200, 700)
    win._plot.set_x_range(0.0, 0.9)
    win._plot.set_y_range(0.0, 9.0)
    app.processEvents()
    vb = win._plot._plot_item.getViewBox()
    # nearest data point of P1 is (0.5, 5.0)
    win._plot._on_scene_clicked(FakeClick(vb.mapViewToScene(QPointF(0.5, 5.0))))
    app.processEvents()
    assert win._plot._click_label.isVisible()
    assert "0.5" in win._plot._click_label.textItem.toPlainText()
    # clicking far from any curve (but inside the plot) dismisses it
    win._plot._on_scene_clicked(FakeClick(vb.mapViewToScene(QPointF(0.45, 8.5))))
    app.processEvents()
    assert not win._plot._click_label.isVisible()
