# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Frozen-aware project root resolver.

When running as a PyInstaller --onefile bundle, __file__ points into a temp
extraction directory, not the project folder.  sys.executable is always the
.exe itself, so its parent is the directory the user placed the exe in — which
must be the project root (next to red.py, green.py, intrinsic_*.py, etc.).

When running as plain Python, fall back to the repo root two levels up from
this file (gui/paths.py → gui/ → project root).
"""
from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def bundled_asset_path(filename: str) -> Path:
    """Resolve a PyInstaller-bundled runtime UI asset (icon, gif, ...) --
    NOT a sibling script. Bundled `datas` files extract into sys._MEIPASS
    (a temp dir) when frozen, not next to the exe, unlike project_root()
    above (which resolves loose sibling scripts the exe expects to find
    beside itself). Plain-Python and frozen both resolve to the project
    root when not frozen."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / filename  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent / filename
