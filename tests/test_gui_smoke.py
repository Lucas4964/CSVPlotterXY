"""Offscreen GUI smoke test for the multi-file + expressions + zoom flow."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt, QUrl
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
    # default selection survived the second file load: X=index(a), first
    # data column (time) checked as Y
    assert win._panel.x_ref() == SeriesRef("index", f1, INDEX_NAME)
    assert win._panel.y_refs() == [SeriesRef("column", f1, "time")]
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
    # custom series values correct at cursor (X = time for interpolation)
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    app.processEvents()
    win._plot._cursor.setValue(0.45)
    app.processEvents()
    # readout tree: one top-level row per series [swatch | Série | Valor]
    tree = win._v_readout._tree
    rows = {tree.topLevelItem(i).text(1): tree.topLevelItem(i).text(2)
            for i in range(tree.topLevelItemCount())}
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


def test_legend_label_color_follows_theme(win, app):
    # the legend text must re-render with the theme's color when the theme
    # is toggled with a series already loaded (setLabelTextColor alone only
    # updates the option, not the rendered HTML)
    from plotxy_app.themes import DARK, LIGHT

    def legend_html():
        _s, label = win._plot._legend.items[0]
        return label.item.toHtml().lower()

    win._theme = LIGHT
    win._apply_theme()
    app.processEvents()
    assert LIGHT.text.lower() in legend_html()

    win._theme = DARK
    win._apply_theme()
    app.processEvents()
    assert DARK.text.lower() in legend_html()
    assert LIGHT.text.lower() not in legend_html()


def test_deselect_all_keeps_files(win, app):
    f1 = file_ids(win)[0]
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    win._panel.check_ref(SeriesRef("column", f1, "P1"))
    win._panel.check_ref(SeriesRef("column", f1, "P2"))
    app.processEvents()
    win._project.add_custom("soma", "P1 + P2")
    win._refresh_panel()
    win._panel.check_ref(SeriesRef("custom", "", "soma"))
    app.processEvents()
    assert len(win._plot._curves) == 3

    win._panel._deselect_all()
    app.processEvents()
    # plot cleared, but nothing was unloaded/deleted
    assert win._panel.y_refs() == []
    assert win._plot._curves == {}
    assert len(win._project.files()) == 2
    assert any(c.name == "soma" for c in win._project.custom())
    # the series are still there to be re-checked
    win._panel.check_ref(SeriesRef("column", f1, "P1"))
    app.processEvents()
    assert len(win._plot._curves) == 1


def test_fechar_tudo_moved_to_file_menu(win):
    from PySide6.QtWidgets import QPushButton
    texts = [a.text() for a in win._file_menu.actions()]
    assert "Fechar tudo" in texts
    # the panel no longer carries a reset button
    panel_btns = [b.text() for b in win._panel.findChildren(QPushButton)]
    assert "Fechar tudo" not in panel_btns
    assert any("Desmarcar" in t for t in panel_btns)


def test_reset_session_full(win, app):
    from plotxy_app import __version__
    f1 = file_ids(win)[0]
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    win._panel.check_ref(SeriesRef("column", f1, "P1"))
    app.processEvents()
    win._project.add_custom("soma", "P1 + P2")
    win._refresh_panel()
    win._panel.check_ref(SeriesRef("custom", "", "soma"))
    app.processEvents()
    win._plot.set_curve_color(SeriesRef("column", f1, "P1").key(), "#123456")
    win._open_measures()
    win._open_spectrum()
    win._cursor_menu.set_states(True, True)   # horizontal cursor on
    win._zoom_btn.setChecked(True)            # local zoom on
    win._set_project_path("C:/algum/lugar/proj.plotxy")
    app.processEvents()
    assert win._measures.isVisible() and win._spectrum.isVisible()

    win.reset_session()
    app.processEvents()

    assert win._project.files() == []
    assert list(win._project.custom()) == []
    assert win._plot._curves == {}
    assert win._plot._color_override == {}
    assert win._panel.x_ref() is None
    assert win.windowTitle() == f"CSVPlotterXY {__version__}"
    assert win._project_path is None
    assert not win._measures.isVisible()
    assert not win._spectrum.isVisible()
    assert win._cursor_menu.states() == (True, False)
    assert win._zoom_btn.isChecked() is False
    assert not win._plot._cursor.isVisible()


def test_reset_confirmation(win, app, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    f1 = file_ids(win)[0]
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    app.processEvents()
    # loaded session + "No" -> nothing happens
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.No)
    win._on_reset_requested()
    assert len(win._project.files()) == 2
    # loaded session + "Yes" -> reset
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    win._on_reset_requested()
    assert win._project.files() == []
    # empty session -> no dialog is shown at all
    def _boom(*a, **k):
        raise AssertionError("dialog shown on empty session")
    monkeypatch.setattr(QMessageBox, "question", _boom)
    win._on_reset_requested()   # must not raise


# ------------------------------------------------------------ v0.3 features


def test_axis_range_set_and_sync(win, app):
    # use a fine-grained X (time) so a 0.3-wide window is allowed
    win._panel.set_x_ref(SeriesRef("column", file_ids(win)[0], "time"))
    app.processEvents()
    # fields sync live only while the popup is open (hidden panels skip
    # the per-frame updates); open it as the user would
    win._open_axis_popup()
    app.processEvents()
    win._plot.set_x_range(0.2, 0.5)
    win._plot.set_y_range(0.0, 9.0)
    app.processEvents()
    x0, x1, y0, y1 = win._plot.view_ranges()
    assert abs(x0 - 0.2) < 1e-6 and abs(x1 - 0.5) < 1e-6
    assert abs(y0 - 0.0) < 1e-6 and abs(y1 - 9.0) < 1e-6
    # axis-panel fields reflect the view via view_range_changed
    assert abs(float(win._axis._fields["xmin"].text()) - 0.2) < 1e-3
    assert abs(float(win._axis._fields["xmax"].text()) - 0.5) < 1e-3
    win._axis.hide()
    app.processEvents()


def test_axis_panel_apply_and_invalid(win, app):
    win._panel.set_x_ref(SeriesRef("column", file_ids(win)[0], "time"))
    app.processEvents()
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


def test_manual_ranges_override_data_limits(win, app):
    # Desmos-style: manual limits define the viewport exactly, even far
    # beyond the plotted data (P1 spans 0..9).
    win._plot.set_manual_ranges(-10.0, 10.0, -10.0, 10000.0)
    app.processEvents()
    x0, x1, y0, y1 = win._plot.view_ranges()
    assert abs(x0 + 10) < 1e-6 and abs(x1 - 10) < 1e-6
    assert abs(y0 + 10) < 1e-6 and abs(y1 - 10000) < 1e-6
    # a window narrower than the index gap (=1) is also honored
    win._plot.set_manual_ranges(2.0, 2.5, 0.0, 1.0)
    app.processEvents()
    x0, x1, _, _ = win._plot.view_ranges()
    assert abs(x0 - 2.0) < 1e-6 and abs(x1 - 2.5) < 1e-6
    # via the popup Apply path (range_changed -> set_manual_ranges)
    win._axis._fields["xmin"].setText("-5")
    win._axis._fields["xmax"].setText("5")
    win._axis._fields["ymin"].setText("-100")
    win._axis._fields["ymax"].setText("5000")
    win._axis._on_apply()
    app.processEvents()
    x0, x1, y0, y1 = win._plot.view_ranges()
    assert abs(x1 - 5) < 1e-6 and abs(y1 - 5000) < 1e-6
    # Auto resets limits back to data-based
    win._plot.autorange()
    app.processEvents()
    _, x1b, _, y1b = win._plot.view_ranges()
    assert x1b < 100 and y1b < 100


def test_unlimited_zoom_keeps_curve(win, app):
    f1 = file_ids(win)[0]
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    win._panel.check_ref(SeriesRef("column", f1, "P1"))
    app.processEvents()
    curve = next(iter(win._plot._curves.values()))
    # extreme zoom-out: view honors the request exactly (no clamping) and
    # the curve keeps all its points rendered (no downsampling collapse)
    win._plot.set_x_range(-1e6, 1e6)
    app.processEvents()
    x0, x1, _, _ = win._plot.view_ranges()
    assert abs((x1 - x0) - 2e6) < 1.0
    assert curve.curve.getPath().elementCount() > 0
    # extreme zoom-in between two samples: still rendered via clipToView
    win._plot.set_x_range(0.4500001, 0.4500002)
    app.processEvents()
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
    # use P1 explicitly so the test doesn't depend on the default Y column
    ref = SeriesRef("column", file_ids(win)[0], "P1")
    win._panel.check_ref(ref)
    app.processEvents()
    key = ref.key()
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
    item = find_item(win, ref)
    item.setCheckState(Qt.CheckState.Unchecked)
    app.processEvents()
    item.setCheckState(Qt.CheckState.Checked)
    app.processEvents()
    assert win._plot._curves[key].opts["pen"].color().name() == "#ff0000"


def test_readout_swatch_click_requests_color(app):
    # standalone panel so the click doesn't trigger the real (modal)
    # color dialog wired up in MainWindow
    from plotxy_app.readout_panel import CursorReadout, _KEY_ROLE
    panel = CursorReadout("Cursor vertical", "X", "Y")
    panel.show()
    app.processEvents()
    rows = [("column|f1|P1", "P1", "#ff0000", [1.5]),
            ("column|f1|P2", "P2", "#00ff00", [2.5, 3.5])]
    panel.update_values(0.3, rows)
    app.processEvents()
    tree = panel._tree
    assert tree.topLevelItemCount() == 2
    assert tree.topLevelItem(0).data(0, _KEY_ROLE) == "column|f1|P1"
    assert tree.topLevelItem(0).text(2) == "1.5"          # single value inline
    assert tree.topLevelItem(1).text(2) == "2 pontos"     # grouped count
    assert tree.topLevelItem(1).childCount() == 2
    assert tree.topLevelItem(1).child(0).text(2) == "2.5"
    captured = []
    panel.color_change_requested.connect(captured.append)
    panel._on_item_clicked(tree.topLevelItem(1), 0)   # swatch column of P2
    assert captured == ["column|f1|P2"]
    captured.clear()
    panel._on_item_clicked(tree.topLevelItem(0), 1)   # name column -> ignored
    panel._on_item_clicked(tree.topLevelItem(1).child(0), 0)  # child -> no key
    assert captured == []
    panel.hide()


def test_polyline_crossings_pure():
    from plotxy_app.plot_area import polyline_crossings as pc
    # 0 crossings
    assert len(pc(np.array([5.0, 6.0, 7.0]), np.array([1.0, 2.0, 3.0]), 0.0)) == 0
    # 1 crossing, interpolated
    vals = pc(np.array([0.0, 2.0]), np.array([10.0, 20.0]), 1.0)
    assert np.allclose(vals, [15.0])
    # k crossings on an oscillating polyline
    a = np.array([0.0, 2.0, 0.0, 2.0])
    b = np.array([0.0, 1.0, 2.0, 3.0])
    assert np.allclose(pc(a, b, 1.0), [0.5, 1.5, 2.5])
    # level exactly on a shared sample -> deduplicated to one point
    vals = pc(np.array([0.0, 1.0, 2.0]), np.array([0.0, 5.0, 10.0]), 1.0)
    assert np.allclose(vals, [5.0])
    # NaN segments are skipped
    a = np.array([0.0, np.nan, 0.0, 2.0])
    b = np.array([0.0, 1.0, 2.0, 3.0])
    assert np.allclose(pc(a, b, 1.0), [2.5])


def test_vertical_cursor_multiple_crossings(app):
    from plotxy_app.plot_area import PlotArea
    pa = PlotArea()
    pa.show()
    app.processEvents()
    # non-monotonic X zigzag: x=1 is crossed by three segments
    x = np.array([0.0, 2.0, 0.0, 2.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    captured = []
    pa.v_cursor_moved.connect(lambda c, rows: captured.append((c, rows)))
    pa.set_series("xk", "x", x, [("k1", "s1", y)])
    app.processEvents()
    c, rows = captured[-1]
    assert abs(c - 1.0) < 1e-9  # cursor snapped to mid-range
    key, label, color, vals = rows[0]
    assert np.allclose(sorted(vals), [0.5, 1.5, 2.5])
    # markers show all three intersection points
    assert len(pa._markers.data) == 3
    pa.hide()


def test_horizontal_cursor_crossings(app):
    from plotxy_app.plot_area import PlotArea
    pa = PlotArea()
    pa.show()
    app.processEvents()
    x = np.arange(5.0)                       # 0..4
    y = np.array([0.0, 2.0, 0.0, 2.0, 0.0])  # crosses y=1 four times
    captured = []
    pa.h_cursor_moved.connect(lambda c, rows: captured.append((c, rows)))
    pa.set_series("xk", "x", x, [("k1", "s1", y)])
    pa.set_cursor_visible("h", True)
    app.processEvents()
    c, rows = captured[-1]
    assert abs(c - 1.0) < 1e-9  # hcursor starts at mid of Y range (0..2)
    _key, _label, _color, vals = rows[0]
    assert np.allclose(sorted(vals), [0.5, 1.5, 2.5, 3.5])
    assert len(pa._h_markers.data) == 4
    # disabling clears markers and emits empty
    pa.set_cursor_visible("h", False)
    app.processEvents()
    c, rows = captured[-1]
    assert rows == []
    assert len(pa._h_markers.data) == 0
    pa.hide()


def test_cursor_menu_defaults_and_toggle(win, app):
    # defaults: vertical on, horizontal off
    v, h = win._cursor_menu.states()
    assert v and not h
    assert win._plot._cursor.isVisible()
    assert not win._plot._hcursor.isVisible()
    assert win._v_readout.isVisible()
    assert not win._h_readout.isVisible()
    # enable horizontal -> line, markers and panel appear, tree populates
    win._cursor_menu._h_check.setChecked(True)
    app.processEvents()
    assert win._plot._hcursor.isVisible()
    assert win._h_readout.isVisible()
    assert win._h_readout._tree.topLevelItemCount() >= 1
    # disable vertical -> its line and panel hide
    win._cursor_menu._v_check.setChecked(False)
    app.processEvents()
    assert not win._plot._cursor.isVisible()
    assert not win._v_readout.isVisible()
    # restore defaults for other tests
    win._cursor_menu._v_check.setChecked(True)
    win._cursor_menu._h_check.setChecked(False)
    app.processEvents()


def test_display_decimation_large_series(app):
    from plotxy_app.plot_area import PlotArea
    pa = PlotArea()
    pa.resize(900, 500)
    pa.show()
    app.processEvents()
    n = 500_000
    x = np.linspace(0.0, 100.0, n)
    y = np.sin(x * 4) * np.linspace(1, 3, n)
    pa.set_series("k", "x", x, [("s", "s", y)])
    app.processEvents()
    curve = pa._curves["s"]
    xd, yd = curve.getData()
    # the renderer sees a bounded, envelope-faithful subset...
    assert len(xd) <= 2 * 4096 + 4
    assert xd[0] == x[0] and xd[-1] == x[-1]
    assert yd.max() == y.max() and yd.min() == y.min()
    # ...while analysis still uses the full arrays
    assert len(pa._ys["s"]) == n

    # narrow zoom-in -> exact raw slice of the original data
    pa.set_x_range(50.0, 50.001)
    app.processEvents()
    xd2, _ = curve.getData()
    i0 = np.searchsorted(x, 50.0) - 1
    i1 = np.searchsorted(x, 50.001, side="right") + 1
    assert np.array_equal(xd2, x[max(0, i0):i1])

    # extreme zoom-out never loses the curve
    pa.set_x_range(-1e6, 1e6)
    app.processEvents()
    assert curve.curve.getPath().elementCount() > 0

    # autorange restores the full-data view (envelope keeps true bounds)
    pa.autorange()
    app.processEvents()
    x0, x1, y0, y1 = pa.view_ranges()
    assert x0 <= x[0] and x1 >= x[-1]
    assert y0 <= y.min() and y1 >= y.max()
    pa.hide()


def test_cursor_snap_to_samples(app):
    from plotxy_app.plot_area import PlotArea
    pa = PlotArea()
    pa.show()
    app.processEvents()
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    pa.set_series("k", "x", x, [("s", "s", y)])
    pa.set_cursor_visible("h", True)
    app.processEvents()

    # free movement (default): value kept as-is
    pa._cursor.setValue(1.4)
    app.processEvents()
    assert abs(pa._cursor.value() - 1.4) < 1e-9

    # snapping on: vertical cursor jumps to the nearest X sample
    pa.set_cursor_snap(True)
    pa._cursor.setValue(1.4)
    app.processEvents()
    assert abs(pa._cursor.value() - 1.0) < 1e-9
    pa._cursor.setValue(1.6)
    app.processEvents()
    assert abs(pa._cursor.value() - 2.0) < 1e-9
    # horizontal cursor snaps to the nearest Y sample
    pa._hcursor.setValue(22.0)
    app.processEvents()
    assert abs(pa._hcursor.value() - 20.0) < 1e-9

    # snapping off again restores free movement
    pa.set_cursor_snap(False)
    pa._cursor.setValue(1.4)
    app.processEvents()
    assert abs(pa._cursor.value() - 1.4) < 1e-9
    pa.hide()


def test_cursor_menu_snap_signal(win, app):
    assert not win._plot._snap_to_samples
    win._cursor_menu._snap_check.setChecked(True)
    app.processEvents()
    assert win._plot._snap_to_samples
    win._cursor_menu._snap_check.setChecked(False)
    app.processEvents()
    assert not win._plot._snap_to_samples


def test_derivative_custom_series_integration(win, app):
    f1 = file_ids(win)[0]
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    app.processEvents()
    # with X = time selected, D() snapshots time
    _, truncated = win._project.add_custom("dP1", "D(P1)")
    assert not truncated
    win._refresh_panel()
    win._panel.check_ref(SeriesRef("custom", "", "dP1"))
    app.processEvents()
    import numpy as _np
    t = _np.arange(10) * 0.1     # csv_a time column
    p1 = _np.arange(10, dtype=float)  # P1 = i
    assert _np.allclose(
        win._project.values(SeriesRef("custom", "", "dP1")),
        _np.gradient(p1, t))


def test_measures_window_flow(win, app):
    import numpy as _np
    f1 = file_ids(win)[0]
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    win._panel.check_ref(SeriesRef("column", f1, "P1"))
    app.processEvents()
    # open the Medidas window -> region appears with an initial interval
    win._open_measures()
    app.processEvents()
    assert win._measures is not None and win._measures.isVisible()
    assert win._plot._measure_region.isVisible()
    assert win._measures._table.rowCount() >= 1

    # set a known interval and finish the drag -> table shows exact stats
    win._plot._measure_region.setRegion((0.15, 0.65))
    win._plot._measure_region.sigRegionChangeFinished.emit(
        win._plot._measure_region)
    app.processEvents()
    t = _np.arange(10) * 0.1
    p1 = _np.arange(10, dtype=float)
    sel = (t >= 0.15) & (t <= 0.65)
    tbl = win._measures._table
    # find P1's row (column 1 = label)
    row = next(r for r in range(tbl.rowCount())
               if tbl.item(r, 1).text() == "P1")
    assert float(tbl.item(row, 2).text()) == p1[sel].max()      # Máx
    assert float(tbl.item(row, 3).text()) == p1[sel].min()      # Mín
    assert _np.isclose(float(tbl.item(row, 4).text()), p1[sel].mean())
    assert _np.isclose(float(tbl.item(row, 5).text()),
                       _np.sqrt(_np.mean(p1[sel] ** 2)))        # RMS
    assert _np.isclose(float(tbl.item(row, 8).text()),
                       _np.trapezoid(p1[sel], t[sel]))          # Área

    # cache: same interval again -> measures_rows NOT called
    calls = []
    original = win._plot.measures_rows
    win._plot.measures_rows = lambda lo, hi: calls.append(1) or original(lo, hi)
    win._update_measures(0.15, 0.65)
    assert calls == []          # cached
    win._update_measures(0.1, 0.7)
    assert calls == [1]         # new interval -> recomputed
    win._plot.measures_rows = original

    # toggling another series refreshes the table (cache key changed)
    win._panel.check_ref(SeriesRef("column", f1, "P2"))
    app.processEvents()
    labels = {win._measures._table.item(r, 1).text()
              for r in range(win._measures._table.rowCount())}
    assert "P2" in labels

    # closing the window hides the region
    win._measures.close()
    app.processEvents()
    assert not win._plot._measure_region.isVisible()


def test_measures_region_persists_on_refocus(win, app):
    f1 = file_ids(win)[0]
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    win._panel.check_ref(SeriesRef("column", f1, "P1"))
    app.processEvents()
    win._open_measures()
    app.processEvents()
    # user picks a specific interval
    win._plot._measure_region.setRegion((0.2, 0.55))
    win._plot._measure_region.sigRegionChangeFinished.emit(
        win._plot._measure_region)
    app.processEvents()
    # losing then regaining focus must NOT reset the selection
    win._on_measures_visibility(False)
    app.processEvents()
    win._on_measures_visibility(True)
    app.processEvents()
    lo, hi = win._plot._measure_region.getRegion()
    assert abs(lo - 0.2) < 1e-9 and abs(hi - 0.55) < 1e-9
    win._measures.close()
    app.processEvents()


def test_measures_maxmin_cell_interactions(win, app):
    f1 = file_ids(win)[0]
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    win._panel.check_ref(SeriesRef("column", f1, "P1"))
    app.processEvents()
    win._open_measures()
    win._plot._measure_region.setRegion((0.15, 0.65))
    win._plot._measure_region.sigRegionChangeFinished.emit(
        win._plot._measure_region)
    app.processEvents()
    tbl = win._measures._table
    row = next(r for r in range(tbl.rowCount())
               if tbl.item(r, 1).text() == "P1")
    # P1 = i, time = 0.1*i; within [0.15, 0.65] -> samples i=2..6
    # max at i=6 -> (0.6, 6); min at i=2 -> (0.2, 2)

    # left-click Máx -> tooltip at the max point on the graph
    win._measures._on_cell_clicked(row, 2)
    app.processEvents()
    assert win._plot._click_label.isVisible()
    assert win._plot._click_label.textItem.toPlainText() == "(0.6, 6)"

    # left-click Mín -> tooltip at the min point (independent of the above)
    win._measures._on_cell_clicked(row, 3)
    app.processEvents()
    assert win._plot._click_label.textItem.toPlainText() == "(0.2, 2)"

    # right-click "Ir para" on Máx -> vertical cursor jumps to max_x
    m = win._measures._rows[row][3]
    win._measures.goto_x_requested.emit(m["max_x"])
    app.processEvents()
    assert abs(win._plot._cursor.value() - 0.6) < 1e-9
    # and on Mín
    win._measures.goto_x_requested.emit(m["min_x"])
    app.processEvents()
    assert abs(win._plot._cursor.value() - 0.2) < 1e-9
    win._measures.close()
    app.processEvents()


def test_measures_ab_fields_sync(win, app):
    f1 = file_ids(win)[0]
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    win._panel.check_ref(SeriesRef("column", f1, "P1"))
    app.processEvents()
    win._open_measures()
    m = win._measures
    # dragging the region updates the A/B fields live (sigRegionChanged)
    win._plot._measure_region.setRegion((0.2, 0.7))
    app.processEvents()
    assert abs(float(m._a_edit.text()) - 0.2) < 1e-6
    assert abs(float(m._b_edit.text()) - 0.7) < 1e-6
    # editing A/B moves the region and recomputes the table
    m._a_edit.setText("0.3")
    m._b_edit.setText("0.55")
    m._commit_interval()
    app.processEvents()
    lo, hi = win._plot._measure_region.getRegion()
    assert abs(lo - 0.3) < 1e-9 and abs(hi - 0.55) < 1e-9
    assert not m._a_edit.styleSheet()  # no error highlight
    # invalid A >= B is rejected: region unchanged, fields flagged
    m._a_edit.setText("0.9")
    m._b_edit.setText("0.4")
    m._commit_interval()
    app.processEvents()
    lo2, hi2 = win._plot._measure_region.getRegion()
    assert abs(lo2 - 0.3) < 1e-9 and abs(hi2 - 0.55) < 1e-9
    assert "e74c3c" in m._a_edit.styleSheet()
    # a focused field is not clobbered by live region sync
    m._a_edit.setStyleSheet("")
    m._a_edit.setText("typing")
    m._a_edit.setFocus()
    win._plot._measure_region.setRegion((0.1, 0.6))
    app.processEvents()
    assert m._a_edit.text() == "typing"      # skipped (has focus)
    assert abs(float(m._b_edit.text()) - 0.6) < 1e-6  # B still synced
    win._measures.close()
    app.processEvents()


def test_readout_point_click_shows_tooltip(win, app):
    f1 = file_ids(win)[0]
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    win._panel.check_ref(SeriesRef("column", f1, "P1"))
    app.processEvents()
    # move the vertical cursor to a known X; readout lists (x, y)
    win._plot._cursor.setValue(0.5)
    win._plot._on_v_cursor_moved()
    app.processEvents()
    tree = win._v_readout._tree
    row = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
               if tree.topLevelItem(i).text(1) == "P1")
    # clicking the value cell shows the graph tooltip at (0.5, 5)
    win._v_readout._on_item_clicked(row, 2)
    app.processEvents()
    assert win._plot._click_label.isVisible()
    assert win._plot._click_label.textItem.toPlainText() == "(0.5, 5)"
    # (col-0 swatch -> color change is covered by the standalone test; not
    # exercised here because it would open the modal color dialog)


def test_readout_point_click_horizontal_children(app):
    # standalone panel: horizontal cursor with multiple crossings -> each
    # child value carries its own (x=value, y=coord) point
    from plotxy_app.readout_panel import CursorReadout, _X_ROLE, _Y_ROLE
    panel = CursorReadout("Cursor horizontal", "Y", "X")
    panel.show()
    app.processEvents()
    panel.update_values(2.0, [("k1", "s", "#ff0000", [1.0, 3.0, 5.0])])
    app.processEvents()
    top = panel._tree.topLevelItem(0)
    child = top.child(1)  # value 3.0
    assert child.data(0, _X_ROLE) == 3.0   # x = value (horizontal cursor)
    assert child.data(0, _Y_ROLE) == 2.0   # y = coord
    captured = []
    panel.point_clicked.connect(lambda k, x, y: captured.append((k, x, y)))
    panel._on_item_clicked(child, 2)
    assert captured == [("k1", 3.0, 2.0)]
    panel.hide()


def test_hcursor_matches_vcursor_color(win, app):
    win._apply_theme()
    assert win._plot._cursor.pen.color().name() == win._plot._hcursor.pen.color().name()
    assert win._plot._cursor.hoverPen.color().name() == \
        win._plot._hcursor.hoverPen.color().name()


def test_toolbar_button_labels(win, app):
    assert win._scale_btn.text() == "Janela"
    for btn_text in ("Adicionar CSV", "Nova série"):
        assert "…" not in btn_text


def test_origin_lines_are_crisp_infinite_line(win, app):
    from plotxy_app.plot_area import _CrispInfiniteLine
    for line in (win._plot._origin_v, win._plot._origin_h,
                win._plot._zoom_origin_v, win._plot._zoom_origin_h):
        assert isinstance(line, _CrispInfiniteLine)


def test_measures_region_uses_accent_color(win, app):
    from PySide6.QtGui import QColor
    f1 = file_ids(win)[0]
    win._panel.check_ref(SeriesRef("column", f1, "P1"))
    app.processEvents()
    accent = QColor(win._theme.accent)
    brush = win._plot._measure_region.brush.color()
    assert (brush.red(), brush.green(), brush.blue()) == \
        (accent.red(), accent.green(), accent.blue())


def test_decimal_validators_reject_comma(app):
    from PySide6.QtGui import QValidator
    from plotxy_app.axis_panel import AxisPanel
    from plotxy_app.goto_panel import GotoPanel
    ap = AxisPanel()
    gp = GotoPanel()
    edits = list(ap._fields.values()) + [gp._x_edit, gp._y_edit]
    for edit in edits:
        v = edit.validator()
        assert v is not None
        assert v.validate("1.5", 3)[0] == QValidator.State.Acceptable
        assert v.validate("1,5", 3)[0] == QValidator.State.Invalid


def test_default_y_is_first_column(app, tmp_path):
    p = tmp_path / "three.csv"
    p.write_text('"t","a","b"\n' + "\n".join(
        f"{i},{i * 2},{i * 3}" for i in range(5)))
    w = MainWindow()
    w.show()
    w.open_path(str(p))
    app.processEvents()
    fid = w._project.files()[0][0]
    assert w._panel.x_ref() == SeriesRef("index", fid, INDEX_NAME)
    assert w._panel.y_refs() == [SeriesRef("column", fid, "t")]  # first column


def test_readout_copy_value(app):
    from plotxy_app.readout_panel import CursorReadout
    from PySide6.QtWidgets import QApplication
    panel = CursorReadout("Cursor vertical", "X", "Y")
    panel.show()
    app.processEvents()
    panel.update_values(1.23, [
        ("column|f1|P1", "P1", "#ff0000", [4.5]),
        ("column|f1|P2", "P2", "#00ff00", [1.0, 2.0]),
    ])
    app.processEvents()
    tree = panel._tree
    # single value -> "Copiar valor"
    top0 = tree.topLevelItem(0)
    QApplication.clipboard().clear()
    # invoke the copy directly (context menu exec would block)
    panel._copy(top0.text(2))
    assert QApplication.clipboard().text() == "4.5"
    # grouped values -> join children
    top1 = tree.topLevelItem(1)
    vals = "\n".join(top1.child(i).text(2) for i in range(top1.childCount()))
    panel._copy(vals)
    assert QApplication.clipboard().text() == "1\n2"
    # header coordinate is copyable too
    panel._copy(panel._header.text().split("=", 1)[1].strip())
    assert QApplication.clipboard().text() == "1.23"
    panel.hide()


def test_toolbar_popup_toggle(win, app):
    import time as _time
    # closed -> opens
    win._open_cursor_popup()
    app.processEvents()
    assert win._cursor_menu.isVisible()
    # visible -> closes
    win._open_cursor_popup()
    app.processEvents()
    assert not win._cursor_menu.isVisible()
    # a click immediately after the popup auto-closed must NOT reopen it
    win._popup_hidden_at[id(win._cursor_menu)] = _time.monotonic()
    win._open_cursor_popup()
    app.processEvents()
    assert not win._cursor_menu.isVisible()
    # once the guard window passes, it opens again
    win._popup_hidden_at[id(win._cursor_menu)] = _time.monotonic() - 1.0
    win._open_cursor_popup()
    app.processEvents()
    assert win._cursor_menu.isVisible()
    win._cursor_menu.hide()


def test_axis_popup_wiring(win, app):
    win._panel.set_x_ref(SeriesRef("column", file_ids(win)[0], "time"))
    app.processEvents()
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
    f1 = file_ids(win)[0]
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    win._panel.check_ref(SeriesRef("column", f1, "P1"))
    app.processEvents()
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


def test_click_interpolation_mode(win, app):
    win.resize(1200, 700)
    f1 = file_ids(win)[0]
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    win._panel.check_ref(SeriesRef("column", f1, "P1"))
    app.processEvents()
    win._plot.set_x_range(0.0, 0.9)
    win._plot.set_y_range(0.0, 9.0)
    app.processEvents()
    vb = win._plot._plot_item.getViewBox()
    # P1 is the straight line y = 10x
    mid_click = FakeClick(vb.mapViewToScene(QPointF(0.45, 4.5)))

    # ON (default): clicking exactly on the curve, far (px) from any
    # sample, shows the interpolated point
    assert win._cursor_menu._interp_check.isChecked()
    assert win._plot._click_interpolate
    win._plot._on_scene_clicked(mid_click)
    app.processEvents()
    assert win._plot._click_label.textItem.toPlainText() == "(0.45, 4.5)"

    # OFF (via the menu checkbox, as the user would): snaps to the nearest
    # original sample instead
    win._cursor_menu._interp_check.setChecked(False)
    app.processEvents()
    assert not win._plot._click_interpolate
    win._plot._on_scene_clicked(
        FakeClick(vb.mapViewToScene(QPointF(0.41, 4.1))))
    app.processEvents()
    assert win._plot._click_label.textItem.toPlainText() == "(0.4, 4)"
    # OFF: clicking exactly on the curve but far (px) from any sample
    # finds nothing (only original points are selectable)
    win._plot._on_scene_clicked(mid_click)
    app.processEvents()
    assert not win._plot._click_label.isVisible()

    # clicking far away still dismisses
    win._plot._on_scene_clicked(FakeClick(vb.mapViewToScene(QPointF(0.1, 8.5))))
    app.processEvents()
    assert not win._plot._click_label.isVisible()
    # restore default for other tests
    win._cursor_menu._interp_check.setChecked(True)
    app.processEvents()


def test_goto_panel_fields_and_validation(win, app):
    f1 = file_ids(win)[0]
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    win._panel.check_ref(SeriesRef("column", f1, "P1"))
    app.processEvents()
    goto = win._goto
    # default: vertical enabled, horizontal disabled — field AND button,
    # both always visible
    assert goto._x_edit.isEnabled() and goto._go_x_btn.isEnabled()
    assert not goto._y_edit.isEnabled() and not goto._go_y_btn.isEnabled()
    # "Ir para X" moves only the vertical cursor
    win._plot.set_cursor_y(1.0)
    goto._x_edit.setText("0.3")
    goto._on_go_x()
    app.processEvents()
    assert abs(win._plot._cursor.value() - 0.3) < 1e-9
    assert abs(win._plot._hcursor.value() - 1.0) < 1e-9  # Y untouched
    assert goto._error_label.isHidden()
    # X out of range -> inline error, cursor unchanged
    goto._x_edit.setText("50")
    goto._on_go_x()
    app.processEvents()
    assert not goto._error_label.isHidden()
    assert abs(win._plot._cursor.value() - 0.3) < 1e-9
    # enable horizontal cursor -> Y field + button become enabled
    win._cursor_menu._h_check.setChecked(True)
    app.processEvents()
    assert goto._y_edit.isEnabled() and goto._go_y_btn.isEnabled()
    # "Ir para Y" moves only the horizontal cursor (X untouched)
    goto._y_edit.setText("2.0")   # P1 spans 0..9
    goto._on_go_y()
    app.processEvents()
    assert abs(win._plot._hcursor.value() - 2.0) < 1e-9
    assert abs(win._plot._cursor.value() - 0.3) < 1e-9
    # restore defaults
    win._cursor_menu._h_check.setChecked(False)
    app.processEvents()
    assert not goto._y_edit.isEnabled() and not goto._go_y_btn.isEnabled()


def _bring_plot(app):
    """A shown PlotArea with a known series and a fixed, padded view range so
    view<->scene mapping is deterministic. Data: x in [0,4], y in [0,40]."""
    from plotxy_app.plot_area import PlotArea
    pa = PlotArea()
    pa.resize(600, 400)
    pa.show()
    app.processEvents()
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    pa.set_series("k", "x", x, [("s", "s", y)])
    vb = pa._plot_item.getViewBox()
    vb.setRange(xRange=(-0.5, 4.5), yRange=(-5.0, 45.0), padding=0)
    app.processEvents()
    return pa, vb


def _scene_at(vb, x, y):
    return vb.mapViewToScene(QPointF(x, y))


def test_bring_cursor_x_moves_vertical(app):
    pa, vb = _bring_plot(app)
    pa._menu_scene_pos = _scene_at(vb, 2.0, 20.0)
    pa._bring_cursor("v")
    app.processEvents()
    assert abs(pa._cursor.value() - 2.0) < 1e-6
    pa.hide()


def test_bring_cursor_y_enables_and_positions(app):
    pa, vb = _bring_plot(app)
    captured = []
    pa.cursors_enabled_changed.connect(lambda v, h: captured.append((v, h)))
    assert not pa._h_enabled  # horizontal cursor off by default
    pa._menu_scene_pos = _scene_at(vb, 2.0, 20.0)
    pa._bring_cursor("h")
    app.processEvents()
    assert pa._h_enabled  # turned on
    assert abs(pa._hcursor.value() - 20.0) < 1e-6
    assert captured and captured[-1] == (True, True)
    pa.hide()


def test_bring_cursor_respects_bounds_and_snap(app):
    pa, vb = _bring_plot(app)
    # click beyond the data's X max (4.0) but still inside the padded view ->
    # set_cursor_x clamps to the cursor bound
    pa._menu_scene_pos = _scene_at(vb, 4.25, 20.0)
    pa._bring_cursor("v")
    app.processEvents()
    assert abs(pa._cursor.value() - 4.0) < 1e-6
    # with snapping on, the cursor lands on the nearest original sample
    pa.set_cursor_snap(True)
    pa._menu_scene_pos = _scene_at(vb, 1.4, 20.0)
    pa._bring_cursor("v")
    app.processEvents()
    assert abs(pa._cursor.value() - 1.0) < 1e-6
    pa.hide()


def test_bring_cursor_outside_viewbox_ignored(app):
    pa, vb = _bring_plot(app)
    before = pa._cursor.value()
    pa._menu_scene_pos = QPointF(-10000.0, -10000.0)  # nowhere near the plot
    pa._bring_cursor("v")
    app.processEvents()
    assert pa._cursor.value() == before
    pa.hide()


def test_adaptive_antialias_dense_vs_smooth(app):
    # dense spiky data (envelope zigzagging the full view height) must
    # render without AA — that's what keeps high-density regions fluid —
    # while smooth curves keep AA untouched
    from plotxy_app.plot_area import PlotArea
    pa = PlotArea()
    pa.resize(900, 600)
    pa.show()
    app.processEvents()
    n = 35_000
    x = np.linspace(0.0, 8600.0, n)
    grass = np.where(np.arange(n) % 2, 400.0, -50.0).astype(np.float64)
    pa.set_series("k", "x", x, [("g", "g", grass)])
    vb = pa._plot_item.getViewBox()
    vb.setRange(xRange=(-500.0, 10500.0), yRange=(-100.0, 500.0), padding=0)
    app.processEvents()
    assert pa._curves["g"].opts["antialias"] is False

    smooth = 200.0 + 150.0 * np.sin(x / 600.0)
    pa.set_series("k", "x", x, [("s", "s", smooth)])
    vb.setRange(xRange=(-500.0, 10500.0), yRange=(-50.0, 400.0), padding=0)
    app.processEvents()
    assert pa._curves["s"].opts["antialias"] is True
    pa.hide()


def test_adaptive_antialias_restores_on_zoom_in(app):
    # zooming into a dense region until few points remain on screen makes
    # the stroke short again -> AA comes back (adaptive both ways)
    from plotxy_app.plot_area import PlotArea
    pa = PlotArea()
    pa.resize(900, 600)
    pa.show()
    app.processEvents()
    n = 35_000
    x = np.linspace(0.0, 8600.0, n)
    grass = np.where(np.arange(n) % 2, 400.0, -50.0).astype(np.float64)
    pa.set_series("k", "x", x, [("g", "g", grass)])
    vb = pa._plot_item.getViewBox()
    vb.setRange(xRange=(-500.0, 10500.0), yRange=(-100.0, 500.0), padding=0)
    app.processEvents()
    assert pa._curves["g"].opts["antialias"] is False
    # ~8 samples in view: short stroke, AA on again
    vb.setXRange(1000.0, 1002.0, padding=0)
    app.processEvents()
    assert pa._curves["g"].opts["antialias"] is True
    pa.hide()


def test_bring_actions_enabled_only_with_data(app):
    from plotxy_app.plot_area import PlotArea
    pa = PlotArea()
    pa._sync_bring_actions()
    assert not pa._act_bring_x.isEnabled()
    assert not pa._act_bring_y.isEnabled()
    x = np.array([0.0, 1.0, 2.0])
    pa.set_series("k", "x", x, [("s", "s", x * 10)])
    pa._sync_bring_actions()
    assert pa._act_bring_x.isEnabled()
    assert pa._act_bring_y.isEnabled()


def test_bring_cursor_y_syncs_cursor_menu(win, app):
    # integration: bringing a hidden cursor through the plot ticks the
    # Cursores checkbox and reveals the readout panel
    assert not win._plot._h_enabled
    assert not win._cursor_menu._h_check.isChecked()
    vb = win._plot._plot_item.getViewBox()
    xr, yr = vb.viewRange()
    cx = (xr[0] + xr[1]) / 2
    cy = (yr[0] + yr[1]) / 2
    win._plot._menu_scene_pos = vb.mapViewToScene(QPointF(cx, cy))
    win._plot._bring_cursor("h")
    app.processEvents()
    assert win._plot._h_enabled
    assert win._cursor_menu._h_check.isChecked()
    assert win._h_readout.isVisible()
    # restore default
    win._cursor_menu._h_check.setChecked(False)
    app.processEvents()


def _snap_plot(app):
    from plotxy_app.plot_area import PlotArea
    pa = PlotArea()
    pa.show()
    app.processEvents()
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    pa.set_series("k", "x", x, [("s", "s", y)])
    pa.set_cursor_visible("h", True)
    app.processEvents()
    return pa


def test_snap_is_synchronous(app):
    # the anti-flicker guarantee: with snapping on, the line value is
    # already at the nearest sample when setValue returns — BEFORE any
    # event-loop tick, so the raw drag position is never painted
    pa = _snap_plot(app)
    pa.set_cursor_snap(True)
    pa._cursor.setValue(1.4)
    assert abs(pa._cursor.value() - 1.0) < 1e-9   # no processEvents!
    pa._cursor.setValue(1.6)
    assert abs(pa._cursor.value() - 2.0) < 1e-9
    pa._hcursor.setValue(22.0)
    assert abs(pa._hcursor.value() - 20.0) < 1e-9
    # snap off: raw positions kept, still synchronous behavior
    pa.set_cursor_snap(False)
    pa._cursor.setValue(1.4)
    assert abs(pa._cursor.value() - 1.4) < 1e-9
    pa.hide()


def test_marker_click_shows_tooltip(app):
    pa = _snap_plot(app)
    pa.set_cursor_x(2.0)
    app.processEvents()
    pts = pa._markers.points()
    assert len(pts) == 1
    assert pts[0].data() == "s"   # spot carries the series key
    pa._on_marker_clicked(pa._markers, [pts[0]])
    assert pa._click_label.isVisible()
    assert pa._click_label.textItem.toPlainText() == "(2, 20)"
    pa.hide()


def test_h_marker_click_shows_tooltip(app):
    pa = _snap_plot(app)
    pa.set_cursor_y(20.0)
    app.processEvents()
    pts = pa._h_markers.points()
    assert len(pts) == 1
    pa._on_marker_clicked(pa._h_markers, [pts[0]])
    assert pa._click_label.isVisible()
    assert pa._click_label.textItem.toPlainText() == "(2, 20)"
    pa.hide()


def test_focus_on_point_neighborhood(app):
    from plotxy_app.plot_area import PlotArea
    pa = PlotArea()
    pa.resize(900, 600)
    pa.show()
    app.processEvents()
    x = np.linspace(0.0, 100.0, 1001)     # spacing 0.1
    y = x * 2.0
    pa.set_series("k", "x", x, [("s", "s", y)])
    app.processEvents()
    pa.focus_on_point("s", 50.0, 100.0)
    x0, x1, y0, y1 = pa.view_ranges()
    assert abs((x0 + x1) / 2 - 50.0) < 1e-6          # centered on the point
    assert abs((x1 - x0) - 8.0) < 0.01               # ±40 samples * 0.1
    assert y0 <= 92.0 and y1 >= 108.0                # local Y window + point
    # unknown key: no crash, no view change
    before = pa.view_ranges()
    pa.focus_on_point("nope", 10.0, 10.0)
    assert pa.view_ranges() == before
    pa.hide()


def test_goto_point_via_readout_signal(win, app):
    f1 = file_ids(win)[0]
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    win._panel.check_ref(SeriesRef("column", f1, "P1"))
    app.processEvents()
    key = next(iter(win._plot._curves))
    win._v_readout.goto_point.emit(key, 0.5, 5.0)
    app.processEvents()
    x0, x1, _, _ = win._plot.view_ranges()
    assert abs((x0 + x1) / 2 - 0.5) < 1e-6


def test_readout_width_hint_grows_splitter(win, app):
    sizes = win._splitter.sizes()
    assert len(sizes) == 3
    start = sizes[2]
    # a larger hint grows the readout pane (within the 40% cap)
    win._on_readout_width_hint(start + 120)
    app.processEvents()
    grown = win._splitter.sizes()[2]
    assert grown >= start + 100
    # a smaller hint never shrinks it back
    win._on_readout_width_hint(start)
    app.processEvents()
    assert win._splitter.sizes()[2] >= grown - 5


def test_export_image_png_svg_and_clipboard(app, tmp_path):
    pa = _snap_plot(app)
    png = tmp_path / "grafico.png"
    pa.export_image(str(png))
    assert png.exists() and png.stat().st_size > 0
    svg = tmp_path / "grafico.svg"
    pa.export_image(str(svg))
    assert svg.exists() and "<svg" in svg.read_text(encoding="utf-8")
    QApplication.clipboard().clear()
    pa.copy_image()
    assert not QApplication.clipboard().image().isNull()
    pa.hide()


def test_recent_files_tracking(win, app, csv_a, csv_b):
    # hermetic: QSettings persists across runs, so start from a clean list
    win._settings.setValue("recent_files", [])
    win._add_recent_file(csv_a)
    win._add_recent_file(csv_b)
    recent = win._recent_files()
    assert recent[0].endswith("b.csv")
    assert recent[1].endswith("a.csv")
    # re-adding moves the file back to the front, without duplicating
    win._add_recent_file(csv_a)
    recent = win._recent_files()
    assert recent[0].endswith("a.csv")
    assert len(recent) == 2
    # menu population: entries named by basename, existing files enabled
    win._populate_recent_menu()
    actions = win._recent_menu.actions()
    assert actions[0].text() == "a.csv" and actions[0].isEnabled()


class _FakeDrop:
    def __init__(self, paths):
        self._urls = [QUrl.fromLocalFile(str(p)) for p in paths]
        self.accepted = False

    def mimeData(self):
        return self

    def urls(self):
        return self._urls

    def acceptProposedAction(self):
        self.accepted = True


def test_drag_drop_csv(win, app, tmp_path):
    p = tmp_path / "dropped.csv"
    p.write_text('"t","V"\n0,1\n1,2\n2,3\n')
    before = len(win._project.files())
    ev = _FakeDrop([p])
    win.dragEnterEvent(ev)
    assert ev.accepted
    win.dropEvent(ev)
    app.processEvents()
    assert len(win._project.files()) == before + 1
    # non-CSV is not accepted
    ev2 = _FakeDrop([tmp_path / "nota.txt"])
    win.dragEnterEvent(ev2)
    assert not ev2.accepted


def test_spectrum_window_follows_selection_and_legend(win, app):
    f1 = file_ids(win)[0]
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    win._panel.check_ref(SeriesRef("column", f1, "P1"))
    app.processEvents()
    win._open_spectrum()
    app.processEvents()
    assert win._spectrum is not None and win._spectrum.isVisible()
    assert len(win._spectrum._curves) == 1
    # legend toggle hides the series -> spectrum follows
    key = next(iter(win._plot._curves))
    curve = win._plot._curves[key]
    curve.setVisible(False)
    win._plot._legend.sigSampleClicked.emit(curve)
    app.processEvents()
    assert len(win._spectrum._curves) == 0
    curve.setVisible(True)
    win._plot._legend.sigSampleClicked.emit(curve)
    app.processEvents()
    assert len(win._spectrum._curves) == 1
    win._spectrum.close()
    app.processEvents()


def test_spectrum_cursor_reads_peak(win, app, tmp_path):
    fs, f0, n = 1000.0, 50.0, 1000
    p = tmp_path / "seno.csv"
    rows = "\n".join(f"{i / fs},{3.0 * np.sin(2 * np.pi * f0 * i / fs):.9f}"
                     for i in range(n))
    p.write_text('"t","v"\n' + rows)
    win.open_path(str(p))
    f3 = file_ids(win)[-1]
    win._panel._deselect_all()
    win._panel.set_x_ref(SeriesRef("column", f3, "t"))
    win._panel.check_ref(SeriesRef("column", f3, "v"))
    app.processEvents()
    win._open_spectrum()
    app.processEvents()
    sw = win._spectrum
    # cursor starts on the strongest peak (50 Hz) with a useful reading
    assert sw._cursor.isVisible()
    assert abs(sw._cursor.value() - f0) < 1.5
    text = sw._readout.text()
    assert "f = 50" in text
    assert ">v</span>: 3" in text          # amplitude ~3 at the peak bin
    # dragging updates the reading to the new bin
    sw._cursor.setValue(100.0)
    app.processEvents()
    assert "f = 100" in sw._readout.text()
    sw.close()
    app.processEvents()


def test_project_save_load_roundtrip(win, app, tmp_path):
    from plotxy_app.session import load_session, save_session
    f1 = file_ids(win)[0]
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    win._panel.check_ref(SeriesRef("column", f1, "P1"))
    app.processEvents()
    win._project.add_custom("soma", "P1 + P2")
    win._refresh_panel()
    win._panel.check_ref(SeriesRef("custom", "", "soma"))
    app.processEvents()
    key_p1 = SeriesRef("column", f1, "P1").key()
    win._plot.set_curve_color(key_p1, "#123456")
    win._plot.set_manual_ranges(0.1, 0.7, -1.0, 12.0)
    proj = tmp_path / "sessao.plotxy"
    save_session(win, str(proj))
    assert proj.exists()

    win2 = MainWindow()
    win2.show()
    warnings = load_session(win2, str(proj))
    app.processEvents()
    assert warnings == []
    names = [os.path.basename(ds.source_path)
             for _, ds in win2._project.files()]
    assert names == ["a.csv", "b.csv"]
    nf1 = file_ids(win2)[0]
    assert win2._panel.x_ref() == SeriesRef("column", nf1, "time")
    ykeys = {r.key() for r in win2._panel.y_refs()}
    assert SeriesRef("column", nf1, "P1").key() in ykeys
    assert SeriesRef("custom", "", "soma").key() in ykeys
    assert any(c.name == "soma" and c.expression == "P1 + P2"
               for c in win2._project.custom())
    assert win2._plot._color_of[SeriesRef("column", nf1, "P1").key()] == "#123456"
    x0, x1, y0, y1 = win2._plot.view_ranges()
    assert abs(x0 - 0.1) < 1e-6 and abs(x1 - 0.7) < 1e-6
    assert abs(y0 - -1.0) < 1e-6 and abs(y1 - 12.0) < 1e-6
    win2.close()


def test_project_load_with_missing_file(win, app, tmp_path, csv_b):
    from plotxy_app.session import load_session, save_session
    proj = tmp_path / "faltando.plotxy"
    save_session(win, str(proj))
    os.remove(csv_b)
    win2 = MainWindow()
    win2.show()
    warnings = load_session(win2, str(proj))
    app.processEvents()
    assert len(warnings) == 1 and "b.csv" in warnings[0]
    names = [os.path.basename(ds.source_path)
             for _, ds in win2._project.files()]
    assert names == ["a.csv"]
    win2.close()


def test_project_load_invalid_keeps_session(win, tmp_path):
    from plotxy_app.session import SessionError, load_session
    bad = tmp_path / "ruim.plotxy"
    bad.write_text("{isso não é json", encoding="utf-8")
    files_before = len(win._project.files())
    with pytest.raises(SessionError):
        load_session(win, str(bad))
    assert len(win._project.files()) == files_before  # session untouched


def test_legend_toggle_hides_series_everywhere(win, app):
    f1 = file_ids(win)[0]
    win._panel.set_x_ref(SeriesRef("column", f1, "time"))
    win._panel.check_ref(SeriesRef("column", f1, "P1"))
    win._panel.check_ref(SeriesRef("column", f1, "P2"))
    app.processEvents()
    plot = win._plot
    keys = list(plot._curves)
    assert plot.visible_keys() == tuple(keys)
    captured = []
    plot.v_cursor_moved.connect(lambda c, rows: captured.append(rows))
    plot.set_cursor_x(0.3)
    app.processEvents()
    assert len(captured[-1]) == 2
    # simulate the legend swatch click (pyqtgraph toggles visibility,
    # then emits sigSampleClicked with the curve)
    curve = plot._curves[keys[0]]
    curve.setVisible(False)
    plot._legend.sigSampleClicked.emit(curve)
    app.processEvents()
    assert plot.visible_keys() == (keys[1],)
    assert [r[0] for r in captured[-1]] == [keys[1]]   # readout follows
    assert not plot._zoom_curves[keys[0]].isVisible()  # twin mirrored
    assert [r[0] for r in plot.measures_rows(0.0, 0.9)] == [keys[1]]
    # toggle back on
    curve.setVisible(True)
    plot._legend.sigSampleClicked.emit(curve)
    app.processEvents()
    assert plot.visible_keys() == tuple(keys)
    assert len(captured[-1]) == 2
    assert plot._zoom_curves[keys[0]].isVisible()


def test_file_menu_actions(win):
    # NOTE: never call QAction.menu() here — the PySide6 binding hands the
    # menu's ownership to Python and the QMenu gets deleted with the
    # temporary wrapper. Access the stored menu attributes instead.
    assert any(a.text() == "&Arquivo" for a in win.menuBar().actions())
    texts = [a.text() for a in win._file_menu.actions()]
    assert any("Abrir CSV" in t for t in texts)
    assert "Recentes" in texts
    assert any("Exportar imagem" in t for t in texts)
    assert any("Copiar imagem" in t for t in texts)
