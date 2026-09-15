# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
IOI Control Panel — entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.no_scroll_filter import NoScrollWheelFilter, strip_wheel_focus
from gui.ui_scale import ScaleController


def _icon_path() -> Path:
    # In a PyInstaller --onefile bundle the ico is extracted to sys._MEIPASS.
    # In normal Python it lives next to this file in the project root.
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "ioi_icon.ico"  # type: ignore[attr-defined]
    return Path(__file__).parent / "ioi_icon.ico"


def _set_windows_app_id() -> None:
    # Started through pythonw.exe (the shortcut setup.ps1 creates), Windows
    # would group the window under Python's own taskbar identity and show
    # Python's icon. An explicit AppUserModelID makes the taskbar treat it as
    # its own app, using the window icon set below. Must run before any window.
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("GonzalesLab.IOIControlPanel")
    except (AttributeError, OSError):
        pass


def main() -> int:
    _set_windows_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName("IOI Control Panel")
    scale = ScaleController(app)
    app.installEventFilter(NoScrollWheelFilter(app))

    ico = _icon_path()
    if ico.exists():
        app.setWindowIcon(QIcon(str(ico)))

    window = MainWindow(scale)
    strip_wheel_focus(window)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
