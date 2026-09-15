# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Application-wide fix for the classic "scrolled past a spin box / dropdown
and it silently changed value" accident. QAbstractSpinBox and QComboBox
both react to the mouse wheel by default even when the cursor is just
passing over them on the way down a long form (e.g. Advanced settings
panels, the Utilities tab's scroll area) — this filter blocks that and
forwards the wheel event to the widget's parent instead, so the page
scrolls normally and the value underneath the cursor never changes.

That alone isn't quite enough: both widget classes also default to
Qt.WheelFocus, which grants keyboard focus as a side effect of a wheel
event reaching them (handled by Qt before the event ever reaches our
filter above), so the text cursor visibly jumps into the box even though
its value doesn't change. strip_wheel_focus() downgrades them to
StrongFocus (keeps tab/click focus, drops the wheel-triggered grab).

Install once on the QApplication instance and run the sweep once the
window is built (see ioi_control_panel.py):
    app.installEventFilter(NoScrollWheelFilter(app))
    ...
    strip_wheel_focus(window)
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QWidget


class NoScrollWheelFilter(QObject):
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Wheel and isinstance(watched, (QAbstractSpinBox, QComboBox)):
            parent = watched.parentWidget()
            if parent is not None:
                QApplication.sendEvent(parent, event)
            return True
        return super().eventFilter(watched, event)


def strip_wheel_focus(root: QWidget) -> None:
    for widget in list(root.findChildren(QAbstractSpinBox)) + list(root.findChildren(QComboBox)):
        widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
