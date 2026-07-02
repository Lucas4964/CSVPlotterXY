"""Light/dark themes: palettes, Qt stylesheet builder and apply function."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Theme:
    name: str
    window: str          # main window background
    panel: str           # side panels / toolbar background
    base: str            # input / list background
    text: str
    text_muted: str
    border: str
    accent: str
    hover: str
    plot_bg: str
    axis_color: str
    grid_alpha: float
    cursor_color: str
    curve_palette: tuple[str, ...]


# Tableau-style 12-color palettes: bright for dark bg, desaturated for light bg.
_DARK_CURVES = (
    "#4f8cff", "#ff7f0e", "#2ecc71", "#e74c3c", "#b285ff", "#f1c40f",
    "#1abc9c", "#ff6eb4", "#a3e635", "#38bdf8", "#fb923c", "#c084fc",
)
_LIGHT_CURVES = (
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
    "#17becf", "#e377c2", "#7f7f7f", "#bcbd22", "#0f766e", "#7c3aed",
)

DARK = Theme(
    name="dark",
    window="#12141a", panel="#1b1e27", base="#232733",
    text="#e8eaf0", text_muted="#9aa1b5", border="#2e3342",
    accent="#4f8cff", hover="#2a2f3e",
    plot_bg="#161922", axis_color="#9aa1b5", grid_alpha=0.15,
    cursor_color="#ffd166", curve_palette=_DARK_CURVES,
)

LIGHT = Theme(
    name="light",
    window="#f5f6f8", panel="#ffffff", base="#ffffff",
    text="#24292f", text_muted="#6b7280", border="#d5d9e0",
    accent="#2563eb", hover="#eaeef5",
    plot_bg="#ffffff", axis_color="#4b5563", grid_alpha=0.25,
    cursor_color="#d97706", curve_palette=_LIGHT_CURVES,
)

THEMES = {"dark": DARK, "light": LIGHT}


def build_qss(t: Theme) -> str:
    return f"""
QMainWindow, QDialog {{ background: {t.window}; }}
QWidget {{ color: {t.text}; font-size: 10pt; }}

QToolBar {{
    background: {t.panel}; border: none; padding: 4px; spacing: 6px;
    border-bottom: 1px solid {t.border};
}}
QToolBar QLabel {{ color: {t.text_muted}; padding: 0 6px; }}

QPushButton, QToolButton {{
    background: {t.base}; border: 1px solid {t.border}; border-radius: 6px;
    padding: 5px 12px; color: {t.text};
}}
QPushButton:hover, QToolButton:hover {{ background: {t.hover}; border-color: {t.accent}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {t.accent}; color: #ffffff; }}

QComboBox {{
    background: {t.base}; border: 1px solid {t.border}; border-radius: 6px;
    padding: 4px 8px;
}}
QComboBox:hover {{ border-color: {t.accent}; }}
QComboBox QAbstractItemView {{
    background: {t.base}; border: 1px solid {t.border};
    selection-background-color: {t.accent}; selection-color: #ffffff;
}}

QLineEdit {{
    background: {t.base}; border: 1px solid {t.border}; border-radius: 6px;
    padding: 5px 8px;
}}
QLineEdit:focus {{ border-color: {t.accent}; }}

QListView, QTableWidget {{
    background: {t.base}; border: 1px solid {t.border}; border-radius: 6px;
    outline: none; alternate-background-color: {t.hover};
}}
QListView::item {{ padding: 3px 4px; border-radius: 4px; }}
QListView::item:hover {{ background: {t.hover}; }}
QListView::item:selected {{ background: {t.accent}; color: #ffffff; }}

QHeaderView::section {{
    background: {t.panel}; color: {t.text_muted}; border: none;
    border-bottom: 1px solid {t.border}; padding: 5px;
}}
QTableWidget {{ gridline-color: {t.border}; }}

QGroupBox {{
    border: 1px solid {t.border}; border-radius: 6px; margin-top: 12px;
    background: {t.panel}; padding-top: 4px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 8px; padding: 0 4px;
    color: {t.text_muted};
}}

QSplitter::handle {{ background: {t.window}; width: 4px; }}
QSplitter::handle:hover {{ background: {t.accent}; }}

QStatusBar {{
    background: {t.panel}; color: {t.text_muted};
    border-top: 1px solid {t.border};
}}

QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {t.border}; border-radius: 5px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {t.accent}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{
    background: transparent; height: 10px; margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {t.border}; border-radius: 5px; min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{ background: {t.accent}; }}

QCheckBox::indicator, QListView::indicator {{
    width: 14px; height: 14px; border: 1px solid {t.border};
    border-radius: 4px; background: {t.base};
}}
QCheckBox::indicator:checked, QListView::indicator:checked {{
    background: {t.accent}; border-color: {t.accent};
}}
"""


def build_palette(t: Theme) -> QPalette:
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(t.window))
    p.setColor(QPalette.ColorRole.WindowText, QColor(t.text))
    p.setColor(QPalette.ColorRole.Base, QColor(t.base))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(t.hover))
    p.setColor(QPalette.ColorRole.Text, QColor(t.text))
    p.setColor(QPalette.ColorRole.Button, QColor(t.panel))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(t.text))
    p.setColor(QPalette.ColorRole.Highlight, QColor(t.accent))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(t.panel))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(t.text))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(t.text_muted))
    return p


def apply_app_theme(app: QApplication, theme: Theme) -> None:
    """Apply palette + stylesheet to the whole application.

    Plot-specific recoloring is done by PlotArea.apply_theme, which the
    main window calls right after this.
    """
    app.setPalette(build_palette(theme))
    app.setStyleSheet(build_qss(theme))
