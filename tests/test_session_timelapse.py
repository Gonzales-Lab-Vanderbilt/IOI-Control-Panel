# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""Tests for session_timelapse's timing and display logic on synthetic logs.
Run: py -3.10 -m pytest tests"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import session_timelapse as st  # noqa: E402

DT_NS = 100_400_000  # 100.4 ms camera interval


def write_logs(d: Path, blocks):
    """blocks: list of (host_prefix, cam0_ns, stim_ms) -- each a full acquisition of trial 1."""
    rows, marks = [], []
    for host, cam0, stim_ms in blocks:
        trig = 0
        def add(phase, fname, k):
            rows.append(dict(trial_index=1, phase=phase, frame_index_in_phase=0, global_frame_count=0,
                             host_timestamp_iso=f"{host}:{k:02d}.000", camera_timestamp=cam0 + k * DT_NS,
                             filename=fname, last_serial_marker=""))
        # triggers 1..10 baseline (saved 1..8), 9..10 cushion unsaved, 11..14 gap, 15..18 post; stim at trigger 12
        for k in range(1, 9):
            add("baseline_collect", f"baseline_{k:05d}.raw", k)
        for k in (9, 10):
            add("baseline_full", "", k)
        for i, k in enumerate(range(11, 15), 1):
            add("gap", f"gap_{i:05d}.raw", k)
        for i, k in enumerate(range(15, 19), 1):
            add("post_collect", f"post_{i:05d}.raw", k)
        # Arduino: stim at trigger 12, first post frame at trigger 15 => +3 intervals
        for m, ms in (("STIM_START", stim_ms), ("POST_FIRST_FRAME_TRIGGER", stim_ms + 301), ("STIM_END", stim_ms + 5004)):
            marks.append(dict(trial_index=1, host_timestamp_iso=f"{host}:15.000", marker=m,
                              arduino_millis=ms, raw_text=f"{m},{ms}"))
    with open(d / "session_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(d / "marker_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(marks[0].keys()))
        w.writeheader(); w.writerows(marks)


def test_frame_times_anchor_on_arduino_and_use_camera_spacing(tmp_path):
    write_logs(tmp_path, [("2026-09-09T11:00", 5_000_000_000, 100_000)])
    table, rep = st.frame_table(tmp_path, [1])
    frames = {f[1]: f[2] for f in table[1]}
    assert frames["post_00001.raw"] == pytest.approx(0.301)                 # Arduino anchor
    assert frames["post_00002.raw"] - frames["post_00001.raw"] == pytest.approx(0.1004)   # camera spacing
    assert frames["baseline_00008.raw"] == pytest.approx(0.301 - 7 * 0.1004)
    assert rep["stimulus_duration_s"]["median"] == pytest.approx(5.004)
    assert len(table[1]) == 16                                               # unsaved cushion rows ignored


def test_rerun_into_same_trial_keeps_the_later_block(tmp_path):
    # the first attempt is overwritten on disk by the second; timing must follow the second
    write_logs(tmp_path, [("2026-09-09T11:00", 5_000_000_000, 100_000),
                          ("2026-09-09T11:05", 9_000_000_000, 400_000)])
    table, _ = st.frame_table(tmp_path, [1])
    frames = {f[1]: f[2] for f in table[1]}
    assert len(table[1]) == 16
    assert frames["post_00001.raw"] == pytest.approx(0.301)
    assert frames["gap_00001.raw"] == pytest.approx(0.301 - 4 * 0.1004)


def test_missing_spans_are_reported_not_bridged():
    dt = 0.1
    t = np.arange(-5, 6) * dt
    present = np.ones_like(t, bool); present[3:5] = False
    spans = st.missing_spans(t, present, dt)
    assert len(spans) == 1 and spans[0] == pytest.approx((t[3] - dt / 2, t[5] - dt / 2))

    res = {"mean": np.zeros((6, 4, 4), np.float32)}
    res["mean"][:] = np.arange(6)[:, None, None]
    res["mean"][2] = np.nan                                   # a missing frame
    valid = np.ones((4, 4), bool)
    out = st.display_stack(res, valid, b=4, demean=False, smooth_frames=3)
    assert np.isnan(out[2]).all()
    assert out[1, 0, 0] == pytest.approx(100.0)               # next to the gap: left unsmoothed
    assert out[4, 0, 0] == pytest.approx(400.0)               # 3,4,5 averaged = 4 (x100 %)


def test_median_demean_ignores_a_focal_response():
    res = {"mean": np.zeros((1, 20, 20), np.float32)}
    res["mean"][0, :5, :5] = -0.01                            # 1/16 of the field responds
    out = st.display_stack(res, np.ones((20, 20), bool), b=4, demean=True, smooth_frames=1)
    assert abs(np.nanmedian(out[0])) < 1e-12                  # background stays at zero
    assert out[0, 19, 19] == pytest.approx(0.0, abs=1e-3)


def test_landmark_outside_the_crop_becomes_an_edge_arrow():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lm = {"points": {"bregma": {"x": 900, "y": -410, "status": "estimated"},      # off-image
                     "lambda": {"x": 700, "y": 600, "status": "visible"}}}         # inside the crop
    fig, ax = plt.subplots()
    ax.set_xlim(0, 250); ax.set_ylim(175, 0)
    st.draw_landmarks(ax, lm, (400, 300, 1000, 700), 4, 8, um=3.682)
    texts = [t.get_text() for t in ax.texts]
    assert "B? 2.6 mm" in texts and "L" in texts                                  # (300 + 410) px * 3.682 um
    arrow = [a for a in ax.texts if a.get_text() == "" and a.arrow_patch is not None][0]
    tip_x, tip_y = arrow.xy
    assert 0 <= tip_x <= 250 and 0 <= tip_y <= 175                                 # drawn at the panel edge
    plt.close(fig)
