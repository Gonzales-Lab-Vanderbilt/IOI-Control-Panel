# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
ioi_landmarks.py -- the per-session skull-landmark file, <session>/landmarks.json.

One small, stdlib-only module shared by the GUI (gui/landmark_selector.py,
which writes the file) and the analysis scripts (session_poster_figures.py,
session_timelapse.py, which read it). Stdlib only on purpose: the GUI process
must never import numpy (see README, "What it is not").

What the file records, all in FULL-RESOLUTION SENSOR PIXELS (1920 x 1200,
x right, y down, the same frame as the green reference and every crop):

  orientation      which image edge is anterior, which image side is the
                   animal's LEFT (this fixes handedness, so no mirror-image
                   ambiguity), and which hemisphere(s) are in view
  points           bregma / lambda, each with how it was obtained:
                     "visible"        the suture junction can be seen
                     "estimated"      judged from partial sutures / anatomy
                   plus up to two points on the sagittal midline
  reference_point  optional: any clicked point whose stereotaxic position is
                   known from the surgical record (e.g. the craniotomy or
                   window centre at AP -3.5, ML +2.5 from bregma). Lets an
                   image with no visible landmark still be placed on the map,
                   honestly flagged as "from_surgical_record".
  calibration      micrometres per full-resolution pixel FOR THIS SESSION
                   (calibration changes whenever the optics do, so it is
                   stored per session, not per rig)

stereotaxic() turns any image point into (AP, ML) millimetres:
  AP  + = anterior of the origin, along the midline
  ML  + = animal's RIGHT, measured perpendicular to the midline
The origin is bregma when available, else the reference point's implied
bregma, else lambda (reported as lambda-relative). Every result says which
origin it used and how each input was obtained, so a map can show
visible-landmark and estimated sessions differently.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

SCHEMA = "ioi-landmarks/1"
FILENAME = "landmarks.json"
SENSOR_W, SENSOR_H = 1920, 1200

SIDES = ("left", "right", "top", "bottom")
OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
HEMISPHERES = ("left", "right", "both")
POINT_STATUSES = ("visible", "estimated")
_SIDE_VECTORS = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "top": (0.0, -1.0), "bottom": (0.0, 1.0)}


class LandmarkError(ValueError):
    pass


# ── construction / validation ────────────────────────────────────────────────

def empty(session_name: str = "") -> dict:
    return {
        "schema": SCHEMA,
        "session": session_name,
        "coordinate_frame": "full-resolution sensor pixels, x right, y down",
        "image": {"width": SENSOR_W, "height": SENSOR_H},
        "orientation": {"anterior_side": None, "animal_left_side": None, "hemispheres": None},
        "points": {"bregma": None, "lambda": None, "midline": []},
        "reference_point": None,
        "calibration": {"um_per_px": None, "source": ""},
        "annotator": "",
        "annotated_at": None,
        "notes": "",
    }


def perpendicular_sides(side: str) -> tuple[str, str]:
    if side in ("top", "bottom"):
        return ("left", "right")
    return ("top", "bottom")


def validate(lm: dict) -> list[str]:
    """Human-readable problems; empty list means the file is usable as far as
    it goes. Incomplete is allowed (an orientation-only file is still useful
    for figure compasses); inconsistent is not."""
    problems: list[str] = []
    if lm.get("schema") != SCHEMA:
        problems.append(f"schema is {lm.get('schema')!r}, expected {SCHEMA!r}")
    o = lm.get("orientation") or {}
    a, left, hemi = o.get("anterior_side"), o.get("animal_left_side"), o.get("hemispheres")
    if a is not None and a not in SIDES:
        problems.append(f"anterior_side {a!r} is not one of {SIDES}")
    if left is not None:
        if left not in SIDES:
            problems.append(f"animal_left_side {left!r} is not one of {SIDES}")
        elif a in SIDES and left not in perpendicular_sides(a):
            problems.append("animal_left_side must be perpendicular to anterior_side")
    if hemi is not None and hemi not in HEMISPHERES:
        problems.append(f"hemispheres {hemi!r} is not one of {HEMISPHERES}")
    pts = lm.get("points") or {}
    for name in ("bregma", "lambda"):
        p = pts.get(name)
        if p is None:
            continue
        if not _is_xy(p):
            problems.append(f"{name} needs numeric x and y")
        if p.get("status") not in POINT_STATUSES:
            problems.append(f"{name} status must be one of {POINT_STATUSES}")
    mid = pts.get("midline") or []
    if len(mid) not in (0, 2):
        problems.append("midline needs exactly two points (or none)")
    elif len(mid) == 2:
        if not all(_is_xy(p) for p in mid):
            problems.append("midline points need numeric x and y")
        elif math.dist((mid[0]["x"], mid[0]["y"]), (mid[1]["x"], mid[1]["y"])) < 20:
            problems.append("midline points are too close together to define a direction")
    rp = lm.get("reference_point")
    if rp is not None:
        if not _is_xy(rp):
            problems.append("reference_point needs numeric x and y")
        for k in ("ap_mm", "ml_mm"):
            if not isinstance(rp.get(k), (int, float)):
                problems.append(f"reference_point needs numeric {k}")
    um = (lm.get("calibration") or {}).get("um_per_px")
    if um is not None and not (isinstance(um, (int, float)) and um > 0):
        problems.append("calibration um_per_px must be a positive number")
    return problems


def _is_xy(p) -> bool:
    return isinstance(p, dict) and isinstance(p.get("x"), (int, float)) and isinstance(p.get("y"), (int, float))


# ── file I/O ─────────────────────────────────────────────────────────────────

def path_for(session_dir) -> Path:
    return Path(session_dir) / FILENAME


def load(session_dir) -> dict | None:
    """The session's landmarks, or None if it has no landmarks.json.
    Raises LandmarkError on an unreadable or wrong-schema file rather than
    silently ignoring it -- a figure drawn with the wrong orientation is
    worse than one that refuses to draw."""
    p = path_for(session_dir)
    if not p.exists():
        return None
    try:
        lm = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LandmarkError(f"{p}: unreadable ({exc})") from exc
    if lm.get("schema") != SCHEMA:
        raise LandmarkError(f"{p}: schema {lm.get('schema')!r}, expected {SCHEMA!r}")
    return lm


def save(session_dir, lm: dict) -> Path:
    """Validate, stamp the time, and write. Refuses an inconsistent file."""
    problems = validate(lm)
    if problems:
        raise LandmarkError("; ".join(problems))
    lm = dict(lm)
    lm["annotated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    p = path_for(session_dir)
    p.write_text(json.dumps(lm, indent=2), encoding="utf-8")
    return p


# ── orientation helpers (for figure compasses) ───────────────────────────────

def medial_side(lm: dict) -> str | None:
    """The image edge the midline lies toward, for a one-hemisphere image.
    A left-hemisphere image has the midline on the animal's right, and vice
    versa. None for bilateral images (use the midline-centred compass) or
    when orientation is incomplete."""
    o = lm.get("orientation") or {}
    left, hemi = o.get("animal_left_side"), o.get("hemispheres")
    if left not in SIDES or hemi not in ("left", "right"):
        return None
    return OPPOSITE[left] if hemi == "left" else left


def compass_settings(lm: dict) -> dict | None:
    """{'anterior_side', 'medial_side' | None, 'midline_centered'} for
    session_poster_figures.py's compass, or None if anterior is unknown."""
    o = lm.get("orientation") or {}
    a = o.get("anterior_side")
    if a not in SIDES:
        return None
    both = o.get("hemispheres") == "both"
    return {"anterior_side": a, "medial_side": None if both else medial_side(lm), "midline_centered": both}


# ── stereotaxic coordinates ──────────────────────────────────────────────────

def _unit(vx: float, vy: float) -> tuple[float, float]:
    n = math.hypot(vx, vy)
    if n == 0:
        raise LandmarkError("zero-length direction")
    return vx / n, vy / n


def axes(lm: dict) -> dict:
    """Unit vectors (image px space) for anterior and for the animal's right.

    Anterior comes from the midline points when present (the real AP axis,
    however the head was rotated in the frame), with its sign taken from
    anterior_side; otherwise it is the anterior_side edge direction itself,
    which assumes the head was square to the frame. 'source' says which."""
    o = lm.get("orientation") or {}
    a_side, left_side = o.get("anterior_side"), o.get("animal_left_side")
    if a_side not in SIDES or left_side not in SIDES:
        raise LandmarkError("orientation needs anterior_side and animal_left_side")
    ref = _SIDE_VECTORS[a_side]
    mid = [m for m in ((lm.get("points") or {}).get("midline") or []) if _is_xy(m)]
    if len(mid) == 2:
        ax_, ay_ = _unit(mid[1]["x"] - mid[0]["x"], mid[1]["y"] - mid[0]["y"])
        if ax_ * ref[0] + ay_ * ref[1] < 0:
            ax_, ay_ = -ax_, -ay_
        source = "midline"
    else:
        ax_, ay_ = ref
        source = "anterior_side (head assumed square to frame)"
    # the two perpendiculars of the anterior axis; pick the one pointing to the animal's RIGHT
    lv = _SIDE_VECTORS[left_side]
    px, py = -ay_, ax_
    if px * lv[0] + py * lv[1] > 0:   # that one points to the animal's left
        px, py = -px, -py
    return {"anterior": (ax_, ay_), "right": (px, py), "source": source}


def stereotaxic(lm: dict, x: float, y: float) -> dict:
    """(AP, ML) in mm for image point (x, y), full-resolution pixels.

    ML is measured from the midline line when two midline points exist
    (robust to a slightly off-midline bregma click), otherwise from the
    origin. Returns the numbers plus provenance: origin used, how it was
    obtained, and how the axes were set."""
    um = (lm.get("calibration") or {}).get("um_per_px")
    if not (isinstance(um, (int, float)) and um > 0):
        raise LandmarkError("calibration um_per_px is not set")
    ax_ = axes(lm)
    (anx, any_), (rx, ry) = ax_["anterior"], ax_["right"]
    pts = lm.get("points") or {}
    rp = lm.get("reference_point")

    if pts.get("bregma"):
        ox, oy = pts["bregma"]["x"], pts["bregma"]["y"]
        origin, origin_status = "bregma", pts["bregma"]["status"]
    elif rp is not None:
        # bregma implied by a point of known stereotaxic position
        s = 1000.0 / um  # px per mm
        ox = rp["x"] - (rp["ap_mm"] * anx + rp["ml_mm"] * rx) * s
        oy = rp["y"] - (rp["ap_mm"] * any_ + rp["ml_mm"] * ry) * s
        origin, origin_status = "bregma", "from_surgical_record"
    elif pts.get("lambda"):
        ox, oy = pts["lambda"]["x"], pts["lambda"]["y"]
        origin, origin_status = "lambda", pts["lambda"]["status"]
    else:
        raise LandmarkError("no bregma, lambda or reference point to anchor coordinates")

    mm_per_px = um / 1000.0
    dx, dy = x - ox, y - oy
    ap = (dx * anx + dy * any_) * mm_per_px
    mid = [m for m in (pts.get("midline") or []) if _is_xy(m)]
    if len(mid) == 2:
        mx, my = mid[0]["x"], mid[0]["y"]
        ml = ((x - mx) * rx + (y - my) * ry) * mm_per_px
        ml_ref = "midline"
    else:
        ml = (dx * rx + dy * ry) * mm_per_px
        ml_ref = origin
    return {
        "ap_mm": ap, "ml_mm": ml,
        "origin": origin, "origin_status": origin_status,
        "ml_measured_from": ml_ref, "axes_from": ax_["source"],
        "um_per_px": um,
    }
