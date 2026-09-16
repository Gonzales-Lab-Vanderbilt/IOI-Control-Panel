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


APP_USER_MODEL_ID = "GonzalesLab.IOIControlPanel"


def _set_windows_app_id() -> None:
    # Started through pythonw.exe (the shortcut setup.ps1 creates), Windows
    # would group the window under Python's own taskbar identity and show
    # Python's icon. An explicit AppUserModelID makes the taskbar treat it as
    # its own app, using the window icon set below. Must run before any window.
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except (AttributeError, OSError):
        pass


def _relaunch_details() -> tuple[str, str]:
    """(command line, icon resource) the taskbar should use for a pinned copy."""
    if getattr(sys, "frozen", False):
        # The exe embeds the icon; the extracted _MEIPASS copy is gone on exit.
        return f'"{sys.executable}"', f"{sys.executable},0"
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    if pythonw.exists():
        # Launched from a terminal with python.exe, still pin the console-less one.
        exe = pythonw
    script = Path(__file__).resolve()
    return f'"{exe}" "{script}"', f"{_icon_path().resolve()},0"


def _set_taskbar_relaunch(hwnd: int) -> None:
    # The AppUserModelID alone isn't enough for pinning: no shortcut carries
    # that ID, so "Pin to taskbar" falls back to the process -- a bare
    # pythonw.exe with Python's icon and no script argument. These window
    # properties tell the taskbar what to pin instead.
    if sys.platform != "win32":
        return
    try:
        import ctypes
        import uuid
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                        ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

        class PROPERTYKEY(ctypes.Structure):
            _fields_ = [("fmtid", GUID), ("pid", wintypes.DWORD)]

        class PROPVARIANT(ctypes.Structure):
            # Only the VT_LPWSTR case; the trailing pointer pads to the real size.
            _fields_ = [("vt", wintypes.USHORT), ("reserved1", wintypes.WORD),
                        ("reserved2", wintypes.WORD), ("reserved3", wintypes.WORD),
                        ("pwszVal", ctypes.c_wchar_p), ("pad", ctypes.c_void_p)]

        def guid(text: str) -> GUID:
            return GUID.from_buffer_copy(uuid.UUID(text).bytes_le)

        VT_LPWSTR = 31
        app_model = guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3")
        command, icon = _relaunch_details()
        values = [
            (2, command),             # PKEY_AppUserModel_RelaunchCommand
            (3, icon),                 # PKEY_AppUserModel_RelaunchIconResource
            (4, "IOI Control Panel"),  # PKEY_AppUserModel_RelaunchDisplayNameResource
            (5, APP_USER_MODEL_ID),    # PKEY_AppUserModel_ID
        ]

        shell32 = ctypes.windll.shell32
        shell32.SHGetPropertyStoreForWindow.argtypes = [
            wintypes.HWND, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
        shell32.SHGetPropertyStoreForWindow.restype = ctypes.HRESULT
        store = ctypes.c_void_p()
        iid_property_store = guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")
        shell32.SHGetPropertyStoreForWindow(hwnd, ctypes.byref(iid_property_store), ctypes.byref(store))

        # IPropertyStore vtable: 2 Release, 6 SetValue, 7 Commit.
        vtable = ctypes.cast(store, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        set_value = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, ctypes.POINTER(PROPERTYKEY),
                                       ctypes.POINTER(PROPVARIANT))(vtable[6])
        commit = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p)(vtable[7])
        release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])
        try:
            for pid, text in values:
                key = PROPERTYKEY(app_model, pid)
                value = PROPVARIANT(vt=VT_LPWSTR, pwszVal=text)
                set_value(store, ctypes.byref(key), ctypes.byref(value))
            commit(store)
        finally:
            release(store)
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
    _set_taskbar_relaunch(int(window.winId()))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
