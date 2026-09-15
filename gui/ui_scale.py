# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Runtime UI zoom — scales the whole app by adjusting the global font point
size. Qt's Fusion+QSS layout in gui/theme.py is font-metric-driven (group
boxes, buttons, form spacing all resize from the app font automatically),
so re-deriving that one font size is the lever that cascades through nearly
the entire app on a zoom change.

Persisted as a small standalone JSON file (ui_settings.json, next to
presets/) via gui/paths.py::project_root() -- the same plain json.dumps/
loads convention gui/presets.py uses, not QSettings (which is registry-based
on Windows and would be a second, inconsistent persistence mechanism).
"""
from __future__ import annotations

import json

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from gui.paths import project_root
from gui.theme import apply_dark_theme, apply_zoom_font

MIN_ZOOM = 0.5
MAX_ZOOM = 2.0
ZOOM_STEP = 0.1
DEFAULT_ZOOM = 1.0

_SETTINGS_PATH = project_root() / "ui_settings.json"


def _clamp(value: float) -> float:
    return max(MIN_ZOOM, min(MAX_ZOOM, round(value, 2)))


def load_zoom() -> float:
    try:
        data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        return _clamp(float(data.get("zoom", DEFAULT_ZOOM)))
    except (OSError, ValueError, json.JSONDecodeError):
        return DEFAULT_ZOOM


def save_zoom(value: float) -> None:
    try:
        _SETTINGS_PATH.write_text(json.dumps({"zoom": value}, indent=2), encoding="utf-8")
    except OSError:
        pass  # best-effort -- a failed save just means next launch uses the default


class ScaleController(QObject):
    """Owns the app's current zoom level; applies it via theme.apply_dark_theme
    and persists it to ui_settings.json on every change."""

    zoom_changed = Signal(float)

    def __init__(self, app: QApplication, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._app = app
        self._zoom = load_zoom()
        # Full theme setup (style/palette/stylesheet/font) happens exactly
        # once here, before any widgets exist. Later zoom changes only ever
        # touch the font -- see set_zoom().
        apply_dark_theme(self._app, self._zoom)

    @property
    def zoom(self) -> float:
        return self._zoom

    def set_zoom(self, value: float) -> None:
        value = _clamp(value)
        if value == self._zoom:
            return
        self._zoom = value
        apply_zoom_font(self._app, value)
        save_zoom(value)
        self.zoom_changed.emit(value)

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom + ZOOM_STEP)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom - ZOOM_STEP)

    def reset(self) -> None:
        self.set_zoom(DEFAULT_ZOOM)
