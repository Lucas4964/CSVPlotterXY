# PyInstaller spec for CSVPlotterXY — single-file, windowed Windows build.
# Build from the repo root:  python -m PyInstaller packaging/CSVPlotterXY.spec
#
# PySide6/pyqtgraph/numpy are handled by PyInstaller's bundled hooks; the
# excludes below drop Qt modules the app never imports, shrinking the
# one-file executable from ~200 MB to ~80-110 MB.

import os

block_cipher = None

_here = os.path.dirname(os.path.abspath(SPECPATH))  # repo root
_icon = os.path.join(_here, "packaging", "icon.ico")
_version = os.path.join(_here, "packaging", "version_info.txt")

_EXCLUDES = [
    # heavy Qt modules the app never touches
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick", "PySide6.QtWebChannel", "PySide6.QtWebSockets",
    "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQuickWidgets",
    "PySide6.QtQml", "PySide6.QtQmlModels",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets",
    "PySide6.QtNetwork", "PySide6.QtNetworkAuth",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DInput",
    "PySide6.Qt3DAnimation", "PySide6.Qt3DExtras",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtGraphs",
    "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtSql",
    "PySide6.QtTest", "PySide6.QtBluetooth", "PySide6.QtPositioning",
    "PySide6.QtSensors", "PySide6.QtSerialPort", "PySide6.QtNfc",
    "PySide6.QtLocation", "PySide6.QtRemoteObjects", "PySide6.QtScxml",
    "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtUiTools",
    # non-Qt libraries never used
    "tkinter", "matplotlib", "scipy", "pandas", "PyQt5", "PyQt6",
    "IPython", "pytest",
]

a = Analysis(
    [os.path.join(_here, "main.py")],
    pathex=[_here],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=_EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="CSVPlotterXY",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,        # GUI app: no console window
    disable_windowed_traceback=False,
    icon=_icon,
    version=_version,
)
