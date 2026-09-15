# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for IOI Control Panel.
#
# Build with:  powershell -ExecutionPolicy Bypass -File build_exe.ps1
#              (build environment: INSTALL.md, "Building the standalone exe")
#
# Produces:  ioi_control_panel.exe in the project root (single-file, no console
#            window). build_exe.ps1 passes --distpath . to put it there, next
#            to red.py, green.py, intrinsic_*.py, etc. -- the exe resolves all
#            script paths relative to its own location via gui/paths.py.
#            Running PyInstaller on this spec directly writes to dist\ instead,
#            and the exe must then be moved next to the scripts.
#
# The acquisition scripts (intrinsic_*.py, red.py, green.py, etc.) are NOT
# bundled — they must be present in the same directory as the exe at runtime.
# The gui/ package IS bundled.  PySpin and any Spinnaker SDK code are never
# imported into the GUI and must not be added here.

block_cipher = None

a = Analysis(
    ['ioi_control_panel.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('ioi_icon.ico', '.'),
        ('gonzales_lab_logo.png', '.'),
        ('gui/icons/*.svg', 'gui/icons'),
    ],
    hiddenimports=[
        # pyserial: list_ports uses platform-specific sub-modules loaded at runtime
        'serial',
        'serial.tools',
        'serial.tools.list_ports',
        'serial.tools.list_ports_windows',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Hard constraint: PySpin / Spinnaker SDK must never enter the GUI process
        'PySpin',
        # Analysis-side dependencies not needed in the GUI
        'matplotlib',
        'cv2',
        'tifffile',
        'skimage',
    ],
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
    name='ioi_control_panel',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,        # no black terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='ioi_icon.ico',
)
