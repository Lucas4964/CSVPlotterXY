"""Data layer: DataSet container and file loaders.

The UI only ever sees DataSet instances, so future loaders for other
formats (.mat, .pl4, .lvm, .adf) just need to return a DataSet.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

import numpy as np


class DataLoadError(Exception):
    """Raised when a file cannot be loaded into a DataSet."""


@dataclass(frozen=True)
class DataSet:
    names: list[str]
    columns: np.ndarray  # 2-D float64, shape (nrows, ncols)
    source_path: str
    dropped_columns: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return self.columns.shape[0]

    @property
    def n_cols(self) -> int:
        return self.columns.shape[1]

    def column(self, name: str) -> np.ndarray:
        return self.columns[:, self.names.index(name)]


def _dedup_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append(f"{name} ({seen[name]})")
        else:
            seen[name] = 1
            out.append(name)
    return out


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _detect_delimiter(line: str) -> str:
    """Sniff the field delimiter from the first line. Semicolon or tab
    present means a European/Brazilian export (where the comma is the
    decimal separator); otherwise the classic comma CSV."""
    if ";" in line:
        return ";"
    if "\t" in line:
        return "\t"
    return ","


def load_csv(path: str) -> DataSet:
    """Load a CSV file where each column is an independent data series.

    The delimiter is auto-detected (`,`, `;` or tab). With `;`/tab files
    the decimal comma is converted to a point (Brazilian/European Excel
    and instrument exports). Header row is auto-detected: if any token on
    the first line is not parseable as a float, it is treated as a
    header; otherwise names col_1..col_N are synthesized. Non-numeric
    cells become NaN; columns that are entirely NaN are dropped and
    reported in dropped_columns.

    Loading is synchronous; if very large files ever become a use case,
    this is the call to move onto a QThread.
    """
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            text = f.read()
    except OSError as e:
        raise DataLoadError(f"Não foi possível ler o arquivo:\n{e}") from e

    first_line, _, rest = text.partition("\n")
    if not first_line.strip():
        raise DataLoadError("O arquivo está vazio.")
    delim = _detect_delimiter(first_line)
    tokens = next(csv.reader(io.StringIO(first_line), delimiter=delim))
    # with ; or tab, "0,5" is a decimal-comma number, not a header token
    check = ([t.replace(",", ".") for t in tokens] if delim != ","
             else tokens)
    has_header = not all(_is_float(t) for t in check if t.strip())
    if has_header:
        names = _dedup_names([t.strip() or f"col_{i + 1}"
                              for i, t in enumerate(tokens)])
        body = rest
    else:
        names = [f"col_{i + 1}" for i in range(len(tokens))]
        body = text
    if delim != ",":
        body = body.replace(",", ".")  # decimal comma -> point (data only)
    if not body.strip():
        raise DataLoadError("O arquivo não contém linhas de dados numéricos.")

    try:
        data = np.loadtxt(io.StringIO(body), delimiter=delim, ndmin=2)
    except ValueError:
        try:
            data = np.genfromtxt(io.StringIO(body), delimiter=delim,
                                 filling_values=np.nan, ndmin=2)
        except ValueError:
            raise DataLoadError(
                "O arquivo não contém linhas de dados numéricos.") from None

    if data.size == 0 or data.shape[0] == 0:
        raise DataLoadError("O arquivo não contém linhas de dados numéricos.")
    if data.shape[1] != len(names):
        raise DataLoadError(
            f"Número de colunas inconsistente: cabeçalho tem {len(names)}, "
            f"dados têm {data.shape[1]}.")

    all_nan = np.all(np.isnan(data), axis=0)
    dropped = [n for n, bad in zip(names, all_nan) if bad]
    if dropped:
        data = data[:, ~all_nan]
        names = [n for n, bad in zip(names, all_nan) if not bad]

    if data.shape[1] < 2:
        raise DataLoadError(
            "O arquivo precisa de pelo menos 2 colunas numéricas "
            "(uma para o eixo X e uma para o eixo Y).")

    # Fortran (column-major) order: each column() view is contiguous in
    # memory, so per-series vectorized ops avoid strided access/copies
    return DataSet(names=names, columns=np.asfortranarray(data, dtype=np.float64),
                   source_path=path, dropped_columns=dropped)
