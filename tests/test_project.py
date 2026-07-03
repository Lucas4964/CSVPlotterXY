import numpy as np
import pytest

from plotxy_app.data_model import DataSet
from plotxy_app.project import INDEX_NAME, Project, ProjectError, SeriesRef


def make_ds(names, nrows, path="a.csv", start=0.0):
    cols = np.arange(nrows * len(names), dtype=np.float64)
    cols = cols.reshape(nrows, len(names)) + start
    return DataSet(names=list(names), columns=cols, source_path=path)


@pytest.fixture
def proj():
    return Project()


def test_add_files_and_index(proj):
    f1 = proj.add_file(make_ds(["time", "v"], 5, "a.csv"))
    refs = proj.file_refs(f1)
    assert refs[0] == SeriesRef("index", f1, INDEX_NAME)
    assert [r.name for r in refs[1:]] == ["time", "v"]
    assert np.allclose(proj.values(refs[0]), [0, 1, 2, 3, 4])


def test_label_qualification(proj):
    f1 = proj.add_file(make_ds(["time", "v"], 5, "a.csv"))
    assert proj.label(SeriesRef("column", f1, "time")) == "time"
    f2 = proj.add_file(make_ds(["time", "w"], 5, "b.csv"))
    assert proj.label(SeriesRef("column", f1, "time")) == "time (a.csv)"
    assert proj.label(SeriesRef("column", f2, "time")) == "time (b.csv)"
    assert proj.label(SeriesRef("column", f2, "w")) == "w"  # still unique
    # index qualifies with >= 2 files
    assert "(a.csv)" in proj.label(SeriesRef("index", f1, INDEX_NAME))


def test_ref_by_name(proj):
    f1 = proj.add_file(make_ds(["time", "v"], 5, "a.csv"))
    assert proj.ref_by_name("v") == SeriesRef("column", f1, "v")
    proj.add_file(make_ds(["time", "w"], 5, "b.csv"))
    with pytest.raises(ProjectError, match="ambíguo"):
        proj.ref_by_name("time")
    assert proj.ref_by_name("time (a.csv)") == SeriesRef("column", f1, "time")
    with pytest.raises(KeyError):
        proj.ref_by_name("nope")


def test_resolve_plot_truncation(proj):
    f1 = proj.add_file(make_ds(["time", "v"], 5, "a.csv"))
    f2 = proj.add_file(make_ds(["t2", "w"], 3, "b.csv"))
    rp = proj.resolve_plot(SeriesRef("column", f1, "time"),
                           [SeriesRef("column", f2, "w")])
    assert rp.truncated and rp.n_points == 3
    assert len(rp.x) == 3 and len(rp.series[0][2]) == 3


def test_custom_create_and_use(proj):
    proj.add_file(make_ds(["time", "P1", "P2"], 4, "a.csv"))
    cs, truncated = proj.add_custom("total", "P1 + P2")
    assert not truncated
    expected = proj.values(proj.ref_by_name("P1")) + proj.values(proj.ref_by_name("P2"))
    assert np.allclose(cs.values, expected)
    # custom usable inside another custom
    cs2, _ = proj.add_custom("dobro", "total * 2")
    assert np.allclose(cs2.values, expected * 2)
    assert "total" in cs2.dep_customs
    assert cs2.dep_files == cs.dep_files


def test_custom_name_validation(proj):
    proj.add_file(make_ds(["time", "P1"], 4, "a.csv"))
    with pytest.raises(ProjectError, match="nome"):
        proj.add_custom("  ", "P1")
    with pytest.raises(ProjectError, match="Já existe"):
        proj.add_custom("P1", "P1 * 2")
    with pytest.raises(ProjectError, match="Já existe"):
        proj.add_custom(INDEX_NAME, "P1")
    proj.add_custom("ok", "P1 * 2")
    with pytest.raises(ProjectError, match="Já existe"):
        proj.add_custom("ok", "P1")


def test_remove_file_cascades_customs(proj):
    f1 = proj.add_file(make_ds(["time", "P1"], 4, "a.csv"))
    f2 = proj.add_file(make_ds(["t", "Q"], 4, "b.csv"))
    proj.add_custom("cA", "P1 * 2")        # depends on f1
    proj.add_custom("cB", "Q + 1")         # depends on f2
    proj.add_custom("cAB", "cA + cB")      # depends on both (transitively)
    assert set(proj.dependents_on_file(f1)) == {"cA", "cAB"}
    removed = proj.remove_file(f1)
    assert set(removed) == {"cA", "cAB"}
    assert [c.name for c in proj.custom()] == ["cB"]
    assert len(proj.files()) == 1 and proj.files()[0][0] == f2


def test_remove_custom_cascades(proj):
    proj.add_file(make_ds(["time", "P1"], 4, "a.csv"))
    proj.add_custom("a1", "P1 * 2")
    proj.add_custom("a2", "a1 + 1")
    proj.add_custom("a3", "a2 + 1")
    proj.add_custom("solo", "P1 - 1")
    removed = proj.remove_custom("a1")
    assert set(removed) == {"a1", "a2", "a3"}
    assert [c.name for c in proj.custom()] == ["solo"]


def test_edit_custom_recomputes_dependents(proj):
    proj.add_file(make_ds(["time", "P1"], 4, "a.csv"))
    proj.add_custom("base", "P1 * 2")
    proj.add_custom("dep", "base + 1")
    p1 = proj.values(proj.ref_by_name("P1"))
    recomputed = proj.edit_custom("base", "base", "P1 * 10")
    assert recomputed == ["dep"]
    assert np.allclose(proj.values(proj.ref_by_name("base")), p1 * 10)
    assert np.allclose(proj.values(proj.ref_by_name("dep")), p1 * 10 + 1)


def test_edit_cycle_rejected(proj):
    proj.add_file(make_ds(["time", "P1"], 4, "a.csv"))
    proj.add_custom("a1", "P1 * 2")
    proj.add_custom("a2", "a1 + 1")
    with pytest.raises(ProjectError, match="circular"):
        proj.edit_custom("a1", "a1", "a2 + 1")
    with pytest.raises(ProjectError, match="circular"):
        proj.edit_custom("a1", "a1", "a1 + 1")


def test_rename_blocked_when_referenced(proj):
    proj.add_file(make_ds(["time", "P1"], 4, "a.csv"))
    proj.add_custom("base", "P1 * 2")
    proj.add_custom("dep", "base + 1")
    with pytest.raises(ProjectError, match="renomear"):
        proj.edit_custom("base", "novo_nome", "P1 * 2")
    # renaming when nothing depends on it is fine
    proj.edit_custom("dep", "dep2", "base + 5")
    assert {c.name for c in proj.custom()} == {"base", "dep2"}


def test_custom_truncation_flag(proj):
    proj.add_file(make_ds(["time", "P1"], 5, "a.csv"))
    proj.add_file(make_ds(["t", "Q"], 3, "b.csv"))
    cs, truncated = proj.add_custom("mix", "P1 + Q")
    assert truncated and len(cs.values) == 3


def test_qualified_names_in_expressions(proj):
    proj.add_file(make_ds(["time", "v"], 4, "a.csv"))
    proj.add_file(make_ds(["time", "w"], 4, "b.csv"))
    cs, _ = proj.add_custom("soma", '"time (a.csv)" + "time (b.csv)"')
    assert len(cs.values) == 4
    with pytest.raises(ProjectError, match="ambíguo"):
        proj.add_custom("ruim", "time * 2")
