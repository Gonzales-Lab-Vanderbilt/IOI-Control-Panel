# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Per-computer rig settings that must not travel with presets or the public
release: currently just the spatial calibration (um per full-frame pixel)
used for figure scale bars, and whether someone has verified it on this rig.

Stored as rig_settings.json next to ui_settings.json (same plain-JSON
convention as gui/ui_scale.py). A fresh install has no value, which means
figures are made without a scale bar.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from gui.paths import project_root

_SETTINGS_PATH = project_root() / "rig_settings.json"


@dataclass(frozen=True)
class SpatialCalibration:
    um_per_px: float | None
    verified: bool


def load_spatial_calibration() -> SpatialCalibration:
    try:
        data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        value = data.get("um_per_px")
        um_per_px = float(value) if value is not None and float(value) > 0 else None
        return SpatialCalibration(um_per_px, bool(data.get("um_per_px_verified")) and um_per_px is not None)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return SpatialCalibration(None, False)


def save_spatial_calibration(cal: SpatialCalibration) -> None:
    try:
        data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError, json.JSONDecodeError):
        data = {}
    data["um_per_px"] = cal.um_per_px
    data["um_per_px_verified"] = cal.verified and cal.um_per_px is not None
    try:
        _SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass  # best-effort, like ui_settings.json
