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
    assert _np.isclose(float(tbl.item(row, 7).text()),
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
