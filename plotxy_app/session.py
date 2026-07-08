"""Save/load of a .plotxy project file (JSON).

The file stores everything needed to rebuild a session: the CSV paths
(in load order), the custom series (name + expression, recomputed on
load), the X/Y selection, user-picked colors, the view ranges and the
theme. Series references embed the *index* of their file in the list —
file ids (f1, f2, …) are reassigned by the Project on every load.
"""

from __future__ import annotations

import json
import os

from PySide6.QtWidgets import QApplication

from .project import ProjectError, SeriesRef

FORMAT_VERSION = 1


class SessionError(Exception):
    """Raised when a .plotxy file cannot be read or is malformed."""


def _ref_to_dict(ref: SeriesRef, file_index: dict[str, int]) -> dict:
    return {"kind": ref.kind,
            "file": file_index.get(ref.file_id, -1),
            "name": ref.name}


def _ref_from_dict(d: dict, fids: list[str | None]) -> SeriesRef | None:
    kind, name = d.get("kind"), d.get("name")
    if not kind or name is None:
        return None
    if kind == "custom":
        return SeriesRef("custom", "", name)
    idx = d.get("file", -1)
    if not isinstance(idx, int) or not (0 <= idx < len(fids)):
        return None
    fid = fids[idx]
    if fid is None:   # its file was missing on load
        return None
    return SeriesRef(kind, fid, name)


def save_session(win, path: str) -> None:
    """Serialize the MainWindow session to `path` (JSON)."""
    files = list(win._project.files())
    file_index = {fid: i for i, (fid, _) in enumerate(files)}
    x_ref = win._panel.x_ref()

    colors = []
    for key, color in win._plot._color_override.items():
        kind, fid, name = key.split("|", 2)
        if kind != "custom" and fid not in file_index:
            continue
        colors.append({"kind": kind, "file": file_index.get(fid, -1),
                       "name": name, "color": color})

    data = {
        "version": FORMAT_VERSION,
        "app": "CSVPlotterXY",
        "theme": win._theme.name,
        "files": [os.path.abspath(ds.source_path) for _, ds in files],
        "customs": [{"name": c.name, "expression": c.expression}
                    for c in win._project.custom()],
        "x": _ref_to_dict(x_ref, file_index) if x_ref else None,
        "ys": [_ref_to_dict(r, file_index) for r in win._panel.y_refs()],
        "colors": colors,
        "view": list(win._plot.view_ranges()),
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as e:
        raise SessionError(f"Não foi possível salvar o projeto:\n{e}") from e


def load_session(win, path: str) -> list[str]:
    """Rebuild the session described by `path` into the MainWindow.

    Replaces the current session. Returns a list of warnings (missing
    files, failed custom series) — loading continues without them.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SessionError(f"Não foi possível abrir o projeto:\n{e}") from e
    if not isinstance(data, dict) or data.get("version") != FORMAT_VERSION:
        raise SessionError(
            "Arquivo de projeto inválido ou de versão não suportada.")

    warnings: list[str] = []
    win.reset_session()

    # files, in saved order; a missing file leaves a None placeholder so
    # the saved indices keep pointing to the right datasets
    fids: list[str | None] = []
    for p in data.get("files", []):
        before = {fid for fid, _ in win._project.files()}
        if os.path.exists(p):
            win.open_path(p)
        new = [fid for fid, _ in win._project.files() if fid not in before]
        fids.append(new[0] if new else None)
        if fids[-1] is None:
            warnings.append(f"Arquivo não encontrado: {p}")

    # X axis first (D()/I() snapshot it when customs are recomputed)
    x_ref = _ref_from_dict(data["x"], fids) if data.get("x") else None
    if x_ref is not None:
        win._project.set_x_axis(x_ref)

    for c in data.get("customs", []):
        try:
            win._project.add_custom(str(c["name"]), str(c["expression"]))
        except (ProjectError, KeyError, TypeError) as e:
            warnings.append(f'Série personalizada "{c.get("name")}": {e}')
    win._refresh_panel()

    if x_ref is not None:
        win._panel.set_x_ref(x_ref)
    for d in data.get("ys", []):
        ref = _ref_from_dict(d, fids)
        if ref is not None:
            win._panel.check_ref(ref)

    for c in data.get("colors", []):
        ref = _ref_from_dict(c, fids)
        color = c.get("color")
        if ref is not None and isinstance(color, str):
            win._plot.set_curve_color(ref.key(), color)

    theme = data.get("theme")
    if theme in ("dark", "light") and theme != win._theme.name:
        win.set_theme(theme)

    # let the debounced selection plot the curves, then restore the view
    QApplication.processEvents()
    view = data.get("view")
    if (isinstance(view, list) and len(view) == 4
            and all(isinstance(v, (int, float)) for v in view)):
        win._plot.set_manual_ranges(*view)
    return warnings
