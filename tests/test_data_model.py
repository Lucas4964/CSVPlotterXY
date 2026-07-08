"""load_csv parsing: delimiter auto-detection and decimal-comma support."""

import numpy as np
import pytest

from plotxy_app.data_model import DataLoadError, load_csv


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_classic_comma_csv_unchanged(tmp_path):
    ds = load_csv(_write(tmp_path, "a.csv",
                         '"time","P1"\n0.0,1.5\n0.1,2.5\n'))
    assert ds.names == ["time", "P1"]
    assert np.allclose(ds.column("P1"), [1.5, 2.5])


def test_semicolon_decimal_comma(tmp_path):
    ds = load_csv(_write(tmp_path, "br.csv",
                         "tempo;valor\n0,0;1,5\n0,1;2,75\n1;3\n"))
    assert ds.names == ["tempo", "valor"]
    assert np.allclose(ds.column("tempo"), [0.0, 0.1, 1.0])
    assert np.allclose(ds.column("valor"), [1.5, 2.75, 3.0])


def test_semicolon_with_point_decimals(tmp_path):
    ds = load_csv(_write(tmp_path, "eu.csv",
                         "t;v\n0.0;1.5\n0.5;2.5\n"))
    assert np.allclose(ds.column("v"), [1.5, 2.5])


def test_tab_decimal_comma(tmp_path):
    ds = load_csv(_write(tmp_path, "tab.csv",
                         "t\tv\n0,0\t1,5\n0,5\t2,5\n"))
    assert ds.names == ["t", "v"]
    assert np.allclose(ds.column("v"), [1.5, 2.5])


def test_headerless_semicolon_decimal_comma(tmp_path):
    ds = load_csv(_write(tmp_path, "raw.csv",
                         "0,0;1,5\n0,5;2,5\n1,0;3,5\n"))
    assert ds.names == ["col_1", "col_2"]  # first line is data, not header
    assert ds.n_rows == 3
    assert np.allclose(ds.column("col_1"), [0.0, 0.5, 1.0])


def test_semicolon_all_nan_column_dropped(tmp_path):
    ds = load_csv(_write(tmp_path, "drop.csv",
                         "t;texto;v\n0,0;abc;1,5\n0,5;def;2,5\n"))
    assert ds.dropped_columns == ["texto"]
    assert ds.names == ["t", "v"]


def test_empty_and_header_only(tmp_path):
    with pytest.raises(DataLoadError, match="vazio"):
        load_csv(_write(tmp_path, "e.csv", "\n"))
    with pytest.raises(DataLoadError):
        load_csv(_write(tmp_path, "h.csv", "a;b\n"))
