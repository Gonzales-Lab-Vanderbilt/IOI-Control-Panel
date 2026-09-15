# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Preset I/O — JSON form-state snapshots for the Run Session tab.

Keys are argparse attribute names (dashes → underscores, no leading --).
Internal metadata keys are prefixed with underscore (_name, _description).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gui.paths import project_root

_PROJECT_ROOT = project_root()
PRESETS_DIR = _PROJECT_ROOT / "presets"


def list_presets() -> list[Path]:
    """Return sorted list of .json preset files in the presets folder."""
    if not PRESETS_DIR.exists():
        return []
    return sorted(PRESETS_DIR.glob("*.json"))


def load_preset(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_preset(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def display_name(path: Path, data: dict[str, Any]) -> str:
    return data.get("_name", path.stem)
