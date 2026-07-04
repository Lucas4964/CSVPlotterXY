"""Entry point: python -m plotxy_app [optional_csv_path]"""

from __future__ import annotations

import sys

from PySide6.QtCore import QLocale
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main() -> int:
    # period is the only decimal separator across the whole app
    QLocale.setDefault(QLocale(QLocale.Language.C))
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.show()

    if len(sys.argv) > 1:
        window.open_path(sys.argv[1])

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
