# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
SettlingPlotWidget — small live line plot of relative LED-brightness change
during the red-stabilization wait.

Fed by parsed "Red settling sample: t=..s mean=.. rel=..%" lines from
intrinsic_imaging.py's red-stabilization wait (opt-in via
--red-settling-sample-interval-s). Pure QPainter — no charting library — to
stay within the GUI's PySide6-only dependency footprint.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

_LINE_COLOR = QColor("#27ae60")


class SettlingPlotWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[tuple[float, float]] = []  # (elapsed_s, rel_pct)
        self.setMinimumHeight(90)
        self.setToolTip(
            "Relative LED-brightness change vs. the first sample, taken "
            "periodically (within the calibrated ROI) during red stabilization."
        )

    def clear_points(self) -> None:
        self._points = []
        self.update()

    def add_point(self, elapsed_s: float, rel_pct: float) -> None:
        self._points.append((elapsed_s, rel_pct))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override name)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(6, 6, -6, -6)
        text_color = self.palette().windowText().color()

        if not self._points:
            painter.setPen(text_color)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No settling samples yet.")
            return

        xs = [p[0] for p in self._points]
        ys = [p[1] for p in self._points]
        x_min, x_max = 0.0, max(xs) if max(xs) > 0 else 1.0
        y_min, y_max = min(0.0, min(ys)), max(0.0, max(ys))
        if y_max - y_min < 1e-6:
            y_min, y_max = y_min - 1.0, y_max + 1.0
        pad = (y_max - y_min) * 0.1
        y_min -= pad
        y_max += pad

        def to_px(x: float, y: float) -> QPointF:
            px = rect.left() + (x - x_min) / (x_max - x_min) * rect.width()
            py = rect.bottom() - (y - y_min) / (y_max - y_min) * rect.height()
            return QPointF(px, py)

        grid_color = QColor(text_color)
        grid_color.setAlpha(60)
        zero_pen = QPen(grid_color)
        zero_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(zero_pen)
        zero_y = to_px(x_min, 0.0).y()
        painter.drawLine(QPointF(rect.left(), zero_y), QPointF(rect.right(), zero_y))

        line_pen = QPen(_LINE_COLOR)
        line_pen.setWidthF(1.8)
        painter.setPen(line_pen)
        prev: QPointF | None = None
        for x, y in self._points:
            pt = to_px(x, y)
            if prev is not None:
                painter.drawLine(prev, pt)
            prev = pt

        last_x, last_y = self._points[-1]
        painter.setPen(text_color)
        painter.drawText(
            rect, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
            f"{last_y:+.2f}% @ {last_x:.0f}s",
        )
