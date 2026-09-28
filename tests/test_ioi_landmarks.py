# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""Tests for ioi_landmarks: orientation/handedness, stereotaxic maths, file round trip.
Run: py -3.10 -m pytest tests"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ioi_landmarks as lmk  # noqa: E402

UM = 3.682
PX_PER_MM = 1000.0 / UM


def make(anterior="top", left="left", hemi="both", bregma=(960, 300), status="visible",
         midline=((960, 100), (960, 1100)), lam=None, ref=None, um=UM):
    lm = lmk.empty("s")
    lm["orientation"] = {"anterior_side": anterior, "animal_left_side": left, "hemispheres": hemi}
    lm["points"]["bregma"] = {"x": bregma[0], "y": bregma[1], "status": status} if bregma else None
    lm["points"]["lambda"] = {"x": lam[0], "y": lam[1], "status": "estimated"} if lam else None
    lm["points"]["midline"] = [{"x": x, "y": y} for x, y in midline] if midline else []
    lm["reference_point"] = ref
    lm["calibration"] = {"um_per_px": um, "source": "test"}
    return lm


def test_dorsal_view_anterior_up_left_is_left():
    lm = make()
    assert lmk.validate(lm) == []
    s = lmk.stereotaxic(lm, 960 + 2 * PX_PER_MM, 300)          # 2 mm toward image right
    assert s["ap_mm"] == pytest.approx(0, abs=1e-9)
    assert s["ml_mm"] == pytest.approx(+2.0)                  # image right = animal's right here
    s = lmk.stereotaxic(lm, 960, 300 + PX_PER_MM)              # 1 mm down the image = posterior
    assert s["ap_mm"] == pytest.approx(-1.0)
    assert s["origin"] == "bregma" and s["origin_status"] == "visible"
    assert s["ml_measured_from"] == "midline" and s["axes_from"] == "midline"


def test_handedness_flips_when_left_side_flips():
    a = lmk.stereotaxic(make(left="left"), 960 + PX_PER_MM, 300)["ml_mm"]
    b = lmk.stereotaxic(make(left="right"), 960 + PX_PER_MM, 300)["ml_mm"]
    assert a == pytest.approx(1.0) and b == pytest.approx(-1.0)


def test_anterior_left_head_facing_left():
    # Facing image-left seen from above, the animal's right is image-top, so its left is image-bottom.
    lm = make(anterior="left", left="bottom", bregma=(900, 600), midline=((100, 600), (1800, 600)))
    assert lmk.validate(lm) == []
    s = lmk.stereotaxic(lm, 900 - PX_PER_MM, 600 - 0.5 * PX_PER_MM)
    assert s["ap_mm"] == pytest.approx(1.0)       # toward image left = anterior
    assert s["ml_mm"] == pytest.approx(0.5)       # toward image top = animal's right


def test_rotated_midline_defines_the_axes():
    th = math.radians(12.0)
    ux, uy = math.sin(th), -math.cos(th)          # "anterior", tilted 12 deg off image-up
    b = (1000.0, 700.0)
    mid = ((b[0] - 500 * ux, b[1] - 500 * uy), (b[0] + 500 * ux, b[1] + 500 * uy))
    lm = make(bregma=b, midline=mid)
    s = lmk.stereotaxic(lm, b[0] + 2 * PX_PER_MM * ux, b[1] + 2 * PX_PER_MM * uy)
    assert s["ap_mm"] == pytest.approx(2.0)
    assert s["ml_mm"] == pytest.approx(0.0, abs=1e-9)
    # without a midline the axes fall back to the image edge and the tilt shows up as ML error
    s2 = lmk.stereotaxic(make(bregma=b, midline=None), b[0] + 2 * PX_PER_MM * ux, b[1] + 2 * PX_PER_MM * uy)
    assert s2["axes_from"].startswith("anterior_side")
    assert s2["ml_mm"] == pytest.approx(2.0 * math.sin(th))


def test_ml_is_measured_from_the_midline_not_an_off_midline_bregma():
    lm = make(bregma=(990, 300))                  # bregma clicked 30 px right of the midline
    assert lmk.stereotaxic(lm, 960, 500)["ml_mm"] == pytest.approx(0.0, abs=1e-9)


def test_reference_point_implies_bregma():
    ref = {"x": 1200, "y": 700, "ap_mm": -3.5, "ml_mm": 2.5, "what": "window centre"}
    lm = make(bregma=None, midline=None, ref=ref)
    s = lmk.stereotaxic(lm, 1200, 700)
    assert s["ap_mm"] == pytest.approx(-3.5) and s["ml_mm"] == pytest.approx(2.5)
    assert s["origin"] == "bregma" and s["origin_status"] == "from_surgical_record"
    s = lmk.stereotaxic(lm, 1200, 700 - PX_PER_MM)   # 1 mm anterior of the window centre
    assert s["ap_mm"] == pytest.approx(-2.5)


def test_lambda_only_is_labelled_lambda():
    lm = make(bregma=None, lam=(960, 900))
    s = lmk.stereotaxic(lm, 960, 900 - PX_PER_MM)
    assert s["origin"] == "lambda" and s["ap_mm"] == pytest.approx(1.0)


def test_no_anchor_and_no_calibration_raise():
    with pytest.raises(lmk.LandmarkError):
        lmk.stereotaxic(make(bregma=None), 0, 0)
    with pytest.raises(lmk.LandmarkError):
        lmk.stereotaxic(make(um=None), 0, 0)


def test_validate_catches_inconsistency():
    assert any("perpendicular" in p for p in lmk.validate(make(anterior="top", left="bottom")))
    bad = make()
    bad["points"]["midline"] = [{"x": 1, "y": 2}]
    assert any("exactly two" in p for p in lmk.validate(bad))
    bad = make()
    bad["points"]["bregma"]["status"] = "sure"
    assert any("status" in p for p in lmk.validate(bad))
    half = make()
    half["points"]["midline"] = [{"x": 960, "y": 100}, None]
    assert lmk.validate(half)                      # a half-placed midline can't be saved ...
    lmk.stereotaxic(half, 960, 300)                # ... but doesn't crash the live readout


def test_compass_settings():
    assert lmk.compass_settings(make(hemi="both")) == {"anterior_side": "top", "medial_side": None,
                                                        "midline_centered": True}
    # left hemisphere: the midline is toward the animal's right, i.e. the side opposite animal-left
    assert lmk.compass_settings(make(anterior="left", left="bottom", hemi="left",
                                     midline=((100, 600), (1800, 600))))["medial_side"] == "top"
    assert lmk.compass_settings(make(anterior="left", left="bottom", hemi="right",
                                     midline=((100, 600), (1800, 600))))["medial_side"] == "bottom"
    assert lmk.compass_settings(lmk.empty()) is None


def test_save_load_round_trip(tmp_path):
    lm = make()
    p = lmk.save(tmp_path, lm)
    assert p.name == "landmarks.json"
    back = lmk.load(tmp_path)
    assert back["points"] == lm["points"] and back["annotated_at"]
    assert lmk.load(tmp_path / "nowhere") is None


def test_save_refuses_inconsistent_and_load_refuses_foreign(tmp_path):
    with pytest.raises(lmk.LandmarkError):
        lmk.save(tmp_path, make(anterior="top", left="top"))
    (tmp_path / "landmarks.json").write_text(json.dumps({"schema": "something-else"}))
    with pytest.raises(lmk.LandmarkError):
        lmk.load(tmp_path)
