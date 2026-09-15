# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
PortIndicator — read-only display of the app-wide Arduino port.

The SafetyBar at the top of the window owns the single Arduino-port control.
Tabs that need the port embed a PortIndicator instead of their own dropdown,
so there is exactly one place to change it and every tab follows along.
"""
from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from gui.safety_bar import SafetyBar

_STYLE = "QLabel { font-weight: bold; }"


class PortIndicator(QWidget):
    def __init__(self, safety_bar: SafetyBar, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._safety_bar = safety_bar

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel()
        self._label.setStyleSheet(_STYLE)
        self._label.setToolTip("Set by the Arduino port control at the top of the window.")
        layout.addWidget(self._label)
        layout.addStretch()

        safety_bar.portChanged.connect(self._on_port_changed)
        self._on_port_changed(safety_bar.selected_port() or "")

    def _on_port_changed(self, port: str) -> None:
        self._label.setText(port if port else "(none selected — set at top)")

    def selected_port(self) -> str | None:
        return self._safety_bar.selected_port()
