"""Generate packaging/icon.ico — an axes + curve glyph on a dark tile.

Run once (or after changing the design):  python packaging/make_icon.py
Requires PySide6 (rendering) and Pillow (multi-size .ico writing).
The resulting icon.ico is committed so CI builds don't need to run this.
"""

from __future__ import annotations

import io
import os

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (QBrush, QColor, QImage, QPainter, QPainterPath, QPen)
from PySide6.QtWidgets import QApplication

_ACCENT = "#4f8cff"
_BG = "#161922"
_AXIS = "#9aa1b5"


def _render(size: int) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # rounded dark tile
    r = size * 0.16
    p.setBrush(QBrush(QColor(_BG)))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(0, 0, size, size, r, r)

    m = size * 0.18          # margin
    axis_pen = QPen(QColor(_AXIS))
    axis_pen.setWidthF(size * 0.028)
    axis_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(axis_pen)
    # Y axis and X axis
    p.drawLine(QPointF(m, m * 0.7), QPointF(m, size - m))
    p.drawLine(QPointF(m, size - m), QPointF(size - m * 0.7, size - m))

    # a rising, curving data line
    path = QPainterPath()
    path.moveTo(m, size - m * 1.4)
    path.cubicTo(size * 0.42, size * 0.86,
                 size * 0.5, size * 0.30,
                 size - m * 0.8, m * 0.95)
    curve_pen = QPen(QColor(_ACCENT))
    curve_pen.setWidthF(size * 0.05)
    curve_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    curve_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(curve_pen)
    p.drawPath(path)
    p.end()
    return img


def main() -> None:
    app = QApplication.instance() or QApplication([])
    from PIL import Image

    here = os.path.dirname(os.path.abspath(__file__))
    frames = []
    for s in (256, 128, 64, 48, 32, 16):
        qimg = _render(s)
        buf = io.BytesIO()
        qimg.save_ok = qimg.save  # keep ref (avoid GC quirk)
        # Qt -> PNG bytes -> Pillow
        from PySide6.QtCore import QBuffer, QByteArray
        ba = QByteArray()
        qbuf = QBuffer(ba)
        qbuf.open(QBuffer.OpenModeFlag.WriteOnly)
        qimg.save(qbuf, "PNG")
        qbuf.close()
        frames.append(Image.open(io.BytesIO(bytes(ba.data()))).convert("RGBA"))

    out = os.path.join(here, "icon.ico")
    frames[0].save(out, format="ICO",
                   sizes=[(f.width, f.height) for f in frames])
    print("wrote", out)
    _ = app


if __name__ == "__main__":
    main()
