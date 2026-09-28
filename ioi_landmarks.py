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
                   plus up to two points on the sagittal midline.
                   Points may lie OUTSIDE the image (negative, or past
                   1920 / 1200) when the landmark is off-frame and its
                   position is extrapolated; such a point is always
                   "estimated", and save() stamps "in_field": false on it
                   so maps can tell extrapolated landmarks apart.
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
origin it used, how each input was obtained, and whether the origin lies
on the image, so a map can show visible-landmark, estimated and
extrapolated sessions differently.
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
# how far outside the frame a point may be placed, in frame widths/heights
# (one frame ~= 7 x 4.4 mm at 3.682 um/px; further than that is almost
# certainly a typo, not an extrapolation)
MAX_OUTSIDE_FRAMES = 1.0
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
        elif p.get("status") == "visible" and not in_field(p):
            problems.append(f"{name} is outside the image, so it can't be 'visible': mark it Estimated")
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
    named = [(n, pts.get(n)) for n in ("bregma", "lambda")] + [("reference_point", rp)]
    named += [(f"midline point {i + 1}", m) for i, m in enumerate(mid if isinstance(mid, list) else [])]
    for name, p in named:
        if _is_xy(p) and not _within_limits(p["x"], p["y"]):
            problems.append(f"{name} is more than one image width/height outside the frame: check its x/y")
    um = (lm.get("calibration") or {}).get("um_per_px")
    if um is not None and not (isinstance(um, (int, float)) and um > 0):
        problems.append("calibration um_per_px must be a positive number")
    return problems


def _is_xy(p) -> bool:
    return isinstance(p, dict) and isinstance(p.get("x"), (int, float)) and isinstance(p.get("y"), (int, float))


def _within_limits(x: float, y: float) -> bool:
    mx, my = MAX_OUTSIDE_FRAMES * SENSOR_W, MAX_OUTSIDE_FRAMES * SENSOR_H
    return -mx <= x <= SENSOR_W - 1 + mx and -my <= y <= SENSOR_H - 1 + my


# ── inside / outside the image ───────────────────────────────────────────────

FULL_FRAME_BOX = (0, 0, SENSOR_W - 1, SENSOR_H - 1)


def in_field(p, box=FULL_FRAME_BOX) -> bool:
    """True if point p ({'x', 'y'}) lies on the image (or inside box =
    (x0, y0, x1, y1), inclusive pixel coordinates)."""
    return _is_xy(p) and box[0] <= p["x"] <= box[2] and box[1] <= p["y"] <= box[3]


def outside_edges(x: float, y: float, box=FULL_FRAME_BOX) -> dict:
    """How far (px) the point lies beyond each image edge it is past, e.g.
    {'top': 212.0} or {'top': 40.0, 'left': 95.0}; {} when inside."""
    out = {}
    if y < box[1]:
        out["top"] = box[1] - y
    elif y > box[3]:
        out["bottom"] = y - box[3]
    if x < box[0]:
        out["left"] = box[0] - x
    elif x > box[2]:
        out["right"] = x - box[2]
    return out


def edge_pointer(x: float, y: float, box=FULL_FRAME_BOX, inset: float = 0.0) -> dict | None:
    """For drawing an off-image point as 'it is that way' at the image edge.
    None when (x, y) is inside box. Otherwise: the nearest point on the box
    (pulled in by `inset` so a marker fits), the unit direction from there to
    the real point, and its distance outside the box in px."""
    x0, y0, x1, y1 = box
    cx, cy = min(max(x, x0), x1), min(max(y, y0), y1)
    dist = math.hypot(x - cx, y - cy)
    if dist == 0:
        return None
    return {
        "x": min(max(x, x0 + inset), x1 - inset), "y": min(max(y, y0 + inset), y1 - inset),
        "ux": (x - cx) / dist, "uy": (y - cy) / dist, "outside_px": dist,
    }


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
    lm = json.loads(json.dumps(lm))           # deep copy: don't stamp the caller's dict
    for p in ((lm.get("points") or {}).get("bregma"), (lm.get("points") or {}).get("lambda"),
              lm.get("reference_point")):
        if p:
            p["in_field"] = in_field(p)       # false = extrapolated beyond the image
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
        # False = the origin lies outside the image (extrapolated / implied)
        "origin_in_field": in_field({"x": ox, "y": oy}),
        "ml_measured_from": ml_ref, "axes_from": ax_["source"],
        "um_per_px": um,
    }
