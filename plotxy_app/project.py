"""Project model: multiple files, index series, custom expression series.

Qt-free and fully unit-testable. MainWindow is the single mutator and
refreshes the UI explicitly after each mutation — no observer machinery.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from . import expressions
from .data_model import DataSet

INDEX_NAME = "index"


class ProjectError(Exception):
    """User-facing pt-BR error message."""


@dataclass(frozen=True)
class SeriesRef:
    kind: str      # "column" | "index" | "custom"
    file_id: str   # "f1", "f2", ... ; "" for custom
    name: str      # column name; INDEX_NAME for kind="index"; custom name

    def key(self) -> str:
        """Stable string identity for plot-curve dicts (never displayed)."""
        return f"{self.kind}|{self.file_id}|{self.name}"


@dataclass
class CustomSeries:
    name: str
    expression: str
    values: np.ndarray
    deps: frozenset[SeriesRef]       # direct references
    dep_files: frozenset[str]        # transitive file_ids
    dep_customs: frozenset[str] = field(default_factory=frozenset)  # transitive


@dataclass(frozen=True)
class ResolvedPlot:
    x_key: str
    x_label: str
    x: np.ndarray
    series: list[tuple[str, str, np.ndarray]]  # (key, label, y)
    truncated: bool
    n_points: int


class Project:
    def __init__(self) -> None:
        self._files: dict[str, DataSet] = {}   # insertion-ordered
        self._custom: dict[str, CustomSeries] = {}
        self._index_cache: dict[str, np.ndarray] = {}
        self._next_id = 1

    # -------------------------------------------------------------- files

    def add_file(self, dataset: DataSet) -> str:
        file_id = f"f{self._next_id}"
        self._next_id += 1
        self._files[file_id] = dataset
        return file_id

    def remove_file(self, file_id: str) -> list[str]:
        """Remove a file and cascade-remove custom series depending on it.
        Returns the names of removed custom series."""
        if file_id not in self._files:
            return []
        removed = self.dependents_on_file(file_id)
        for name in removed:
            self._custom.pop(name, None)
        del self._files[file_id]
        self._index_cache.pop(file_id, None)
        return removed

    def files(self) -> list[tuple[str, DataSet]]:
        return list(self._files.items())

    def file_refs(self, file_id: str) -> list[SeriesRef]:
        ds = self._files[file_id]
        refs = [SeriesRef("index", file_id, INDEX_NAME)]
        refs.extend(SeriesRef("column", file_id, n) for n in ds.names)
        return refs

    def dependents_on_file(self, file_id: str) -> list[str]:
        return [c.name for c in self._custom.values()
                if file_id in c.dep_files]

    # ------------------------------------------------------------- custom

    def custom(self) -> list[CustomSeries]:
        return list(self._custom.values())

    def check_custom(self, name: str, expression: str,
                     ignore: str | None = None) -> None:
        """Validate name + expression without mutating. Raises ProjectError."""
        name = name.strip()
        if not name:
            raise ProjectError("Informe um nome para a série.")
        taken = self._all_taken_names(exclude_custom=ignore)
        if name in taken:
            raise ProjectError(f'Já existe uma série chamada "{name}".')
        # cycle check when editing: expression must not (transitively)
        # reference the series being edited
        deps = self._resolve_deps(expression)
        if ignore is not None:
            transitive_customs = set()
            for ref in deps:
                if ref.kind == "custom":
                    transitive_customs.add(ref.name)
                    transitive_customs |= self._custom[ref.name].dep_customs
            if ignore in transitive_customs:
                raise ProjectError("A expressão criaria uma referência circular.")

    def add_custom(self, name: str, expression: str) -> tuple[CustomSeries, bool]:
        """Create a custom series. Returns (series, truncated)."""
        self.check_custom(name, expression)
        cs, truncated = self._compute(name.strip(), expression)
        self._custom[cs.name] = cs
        return cs, truncated

    def edit_custom(self, old_name: str, new_name: str,
                    expression: str) -> list[str]:
        """Edit a custom series, recomputing transitive dependents.
        Returns the names of recomputed dependents."""
        if old_name not in self._custom:
            raise ProjectError(f'Série não encontrada: "{old_name}"')
        self.check_custom(new_name, expression, ignore=old_name)
        new_name = new_name.strip()

        dependents = [c.name for c in self._custom.values()
                      if old_name in c.dep_customs]
        renamed = new_name != old_name
        if renamed and dependents:
            raise ProjectError(
                "Não é possível renomear: outras séries usam esta na "
                f"expressão ({', '.join(dependents)}).")

        # replace, preserving insertion order
        cs, _ = self._compute(new_name, expression, skip_custom=old_name)
        new_dict: dict[str, CustomSeries] = {}
        for k, v in self._custom.items():
            new_dict[new_name if k == old_name else k] = cs if k == old_name else v
        self._custom = new_dict

        # recompute dependents in stored (creation) order
        for dep_name in dependents:
            dep = self._custom[dep_name]
            recomputed, _ = self._compute(dep_name, dep.expression)
            self._custom[dep_name] = recomputed
        return dependents

    def remove_custom(self, name: str) -> list[str]:
        """Remove a custom series and, in cascade, its transitive
        dependents. Returns all removed names (including `name`)."""
        if name not in self._custom:
            return []
        to_remove = {name} | {c.name for c in self._custom.values()
                              if name in c.dep_customs}
        for n in to_remove:
            self._custom.pop(n, None)
        return sorted(to_remove)

    def dependents_on_custom(self, name: str) -> list[str]:
        return [c.name for c in self._custom.values() if name in c.dep_customs]

    # ---------------------------------------------------- labels & lookup

    def label(self, ref: SeriesRef) -> str:
        if ref.kind == "custom":
            return ref.name
        count = self._name_count(ref.name)
        if count <= 1:
            return ref.name
        base = os.path.basename(self._files[ref.file_id].source_path)
        return f"{ref.name} ({base})"

    def ref_by_name(self, name: str) -> SeriesRef:
        """Resolve an expression name: exact qualified label first, then
        unique bare name."""
        matches = [r for r in self._all_refs() if self.label(r) == name]
        if len(matches) == 1:
            return matches[0]
        bare = [r for r in self._all_refs() if r.name == name]
        if len(bare) == 1:
            return bare[0]
        if len(bare) > 1:
            example = self.label(bare[0])
            raise ProjectError(
                f'Nome ambíguo: "{name}" existe em mais de um arquivo. '
                f'Use o nome qualificado, por exemplo "{example}".')
        raise KeyError(name)

    def values(self, ref: SeriesRef) -> np.ndarray:
        if ref.kind == "column":
            return self._files[ref.file_id].column(ref.name)
        if ref.kind == "index":
            if ref.file_id not in self._index_cache:
                n = self._files[ref.file_id].n_rows
                self._index_cache[ref.file_id] = np.arange(n, dtype=np.float64)
            return self._index_cache[ref.file_id]
        return self._custom[ref.name].values

    def resolve_plot(self, x_ref: SeriesRef,
                     y_refs: list[SeriesRef]) -> ResolvedPlot:
        x = self.values(x_ref)
        ys = [(r.key(), self.label(r), self.values(r)) for r in y_refs]
        lengths = [len(x)] + [len(y) for _, _, y in ys]
        n = min(lengths) if lengths else 0
        truncated = any(ln > n for ln in lengths)
        return ResolvedPlot(
            x_key=x_ref.key(), x_label=self.label(x_ref), x=x[:n],
            series=[(k, lbl, y[:n]) for k, lbl, y in ys],
            truncated=truncated, n_points=n)

    # ------------------------------------------------------------ internal

    def _all_refs(self) -> list[SeriesRef]:
        refs: list[SeriesRef] = []
        for fid in self._files:
            refs.extend(self.file_refs(fid))
        refs.extend(SeriesRef("custom", "", n) for n in self._custom)
        return refs

    def _name_count(self, name: str) -> int:
        count = sum(1 for fid, ds in self._files.items()
                    for n in ds.names if n == name)
        count += sum(1 for _ in self._files) if name == INDEX_NAME else 0
        count += 1 if name in self._custom else 0
        return count

    def _all_taken_names(self, exclude_custom: str | None = None) -> set[str]:
        taken: set[str] = set()
        for ds in self._files.values():
            taken.update(ds.names)
        if self._files:
            taken.add(INDEX_NAME)
        taken.update(n for n in self._custom if n != exclude_custom)
        # qualified labels too, so a custom can't shadow "time (a.csv)"
        taken.update(self.label(r) for r in self._all_refs()
                     if not (r.kind == "custom" and r.name == exclude_custom))
        return taken

    def _resolve_deps(self, expression: str) -> set[SeriesRef]:
        names = expressions.collect_series_names(expression)
        deps: set[SeriesRef] = set()
        for n in names:
            try:
                deps.add(self.ref_by_name(n))
            except KeyError:
                raise ProjectError(f'Série não encontrada: "{n}"') from None
        return deps

    def _compute(self, name: str, expression: str,
                 skip_custom: str | None = None) -> tuple[CustomSeries, bool]:
        """Evaluate expression and build the CustomSeries with transitive
        dependency sets. `skip_custom` avoids self-reference when editing."""
        def resolver(nm: str) -> np.ndarray:
            ref = self.ref_by_name(nm)
            if ref.kind == "custom" and ref.name == skip_custom:
                raise ProjectError("A expressão criaria uma referência circular.")
            return self.values(ref)

        try:
            values, used_names, truncated = expressions.evaluate(
                expression, resolver)
        except expressions.ExpressionError as e:
            raise ProjectError(str(e)) from e

        deps = frozenset(self.ref_by_name(n) for n in used_names)
        dep_files: set[str] = set()
        dep_customs: set[str] = set()
        for ref in deps:
            if ref.kind == "custom":
                child = self._custom[ref.name]
                dep_customs.add(ref.name)
                dep_customs |= child.dep_customs
                dep_files |= child.dep_files
            else:
                dep_files.add(ref.file_id)
        return CustomSeries(
            name=name, expression=expression, values=values, deps=deps,
            dep_files=frozenset(dep_files),
            dep_customs=frozenset(dep_customs)), truncated
