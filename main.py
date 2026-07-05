"""Frozen-executable entry point (used by PyInstaller).

Kept separate from ``plotxy_app/__main__.py`` because PyInstaller needs a
plain top-level script; this one uses an absolute import so it works both
frozen and when run directly (``python main.py``).
"""

import sys

from plotxy_app.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
