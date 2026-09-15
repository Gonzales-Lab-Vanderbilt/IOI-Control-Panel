#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""Audit acquisition timing for one IOI session folder.

Reads session_log.csv and marker_log.csv and reports, per trial:

  * mean and SD of the camera inter-frame interval (the real frame clock)
  * mean host inter-frame interval (how fast the consumer actually ran)
  * backlog -- host elapsed minus camera elapsed -- at each phase boundary and
    at trial end. This is the queue depth in frames: it is how far behind the
    camera the consumer had fallen, and it is what used to push phase labels
    onto the wrong frames.
  * count and location of dropped frames
  * offset, in triggers, between each phase-boundary marker's trigger index and
    the first frame actually carrying that phase label. This is the phase
    labeling error, and it should be exactly 0.

Acceptance criteria for a healthy session:
    zero dropped frames, phase-boundary offset 0 for every trial, and
    end-of-trial backlog under 3 frames.

Usage:
    py -3.10 check_frame_timing.py <session_dir>
    py -3.10 check_frame_timing.py --session-log <path> --marker-log <path>

Standard library only, so it runs under any interpreter -- no PySpin, no numpy.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Optional

# Phase label -> the CSV `phase` values that belong to that segment.
# baseline_full / post_full are the same block past its saved-frame target.
PHASE_LABELS = {
    "baseline_collect": {"baseline_collect", "baseline_full"},
    "gap": {"gap"},
    "post_collect": {"post_collect", "post_full"},
    "post_trailing": {"post_trailing"},
}

# Phase-boundary marker -> the segment whose first frame it should name.
# Both marker spellings are accepted; the firmware emits the first of each pair.
BOUNDARY_MARKERS = [
    ("RED_BASELINE_START", "baseline_collect"),
    ("BASELINE_FIRST_FRAME_TRIGGER", "baseline_collect"),
    ("GAP_START", "gap"),
    ("POST_START", "post_collect"),
    ("POST_FIRST_FRAME_TRIGGER", "post_collect"),
    ("POST_TRAILING_START", "post_trailing"),
]

# Candidate camera-timestamp tick rates, used to auto-detect the scale.
CANDIDATE_HZ = (1e9, 1e6, 1e3, 8e7, 1.25e8)

BACKLOG_LIMIT_FRAMES = 3.0


def _to_int(value: str) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_iso_seconds(value: str) -> Optional[float]:
    """Host timestamps are local ISO strings with millisecond precision."""
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(value).strip()).timestamp()
    except (TypeError, ValueError):
        return None


def load_session_log(path: Path) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))

    frames = []
    for row in rows:
        trial = _to_int(row.get("trial_index", ""))
        if trial is None:
            continue
        frames.append({
            "trial_index": trial,
            "phase": (row.get("phase") or "").strip(),
            "frame_index_in_phase": _to_int(row.get("frame_index_in_phase", "")),
            "camera_timestamp": _to_int(row.get("camera_timestamp", "")),
            "host_s": _parse_iso_seconds(row.get("host_timestamp_iso", "")),
            "filename": (row.get("filename") or "").strip(),
            "trigger_index": _to_int(row.get("trigger_index", "")),
            "dropped_before": _to_int(row.get("dropped_before", "")) or 0,
        })
    return frames


def load_marker_log(path: Path) -> list[dict]:
    """Marker trigger indices come out of raw_text, which holds the full line.

    marker_log.csv keeps its original columns; the third serial field
    (MARKER,millis,triggerIndex) is only present inside raw_text.
    """
    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))

    markers = []
    for row in rows:
        trial = _to_int(row.get("trial_index", ""))
        marker = (row.get("marker") or "").strip()
        if trial is None or not marker:
            continue

        trigger_index = None
        parts = [p.strip() for p in (row.get("raw_text") or "").split(",")]
        if len(parts) >= 3:
            trigger_index = _to_int(parts[2])

        markers.append({
            "trial_index": trial,
            "marker": marker,
            "arduino_millis": _to_int(row.get("arduino_millis", "")),
            "trigger_index": trigger_index,
        })
    return markers


def detect_timestamp_hz(frames: list[dict], nominal_period_s: float) -> tuple[float, float]:
    """Pick the tick rate that makes the median inter-frame delta look nominal.

    Returns (hz, median_delta_in_seconds_at_that_hz).
    """
    deltas = []
    by_trial: dict[int, list[dict]] = {}
    for f in frames:
        by_trial.setdefault(f["trial_index"], []).append(f)

    for trial_frames in by_trial.values():
        stamps = [f["camera_timestamp"] for f in trial_frames if f["camera_timestamp"] is not None]
        deltas.extend(
            b - a for a, b in zip(stamps, stamps[1:]) if b > a
        )

    if not deltas:
        return 1e9, 0.0

    median_ticks = statistics.median(deltas)
    best_hz = min(CANDIDATE_HZ, key=lambda hz: abs(median_ticks / hz - nominal_period_s))
    return best_hz, median_ticks / best_hz


def analyze_trial(
    trial_index: int,
    frames: list[dict],
    markers: list[dict],
    hz: float,
    nominal_period_s: float,
) -> dict:
    result: dict = {"trial_index": trial_index, "frame_count": len(frames), "problems": []}

    cam_s = [
        f["camera_timestamp"] / hz if f["camera_timestamp"] is not None else None
        for f in frames
    ]
    host_s = [f["host_s"] for f in frames]

    cam_deltas = [
        (b - a) for a, b in zip(cam_s, cam_s[1:]) if a is not None and b is not None
    ]
    host_deltas = [
        (b - a) for a, b in zip(host_s, host_s[1:]) if a is not None and b is not None
    ]

    result["cam_mean_ms"] = statistics.fmean(cam_deltas) * 1e3 if cam_deltas else None
    result["cam_sd_ms"] = (
        statistics.pstdev(cam_deltas) * 1e3 if len(cam_deltas) > 1 else 0.0
    )
    result["host_mean_ms"] = statistics.fmean(host_deltas) * 1e3 if host_deltas else None
    result["host_sd_ms"] = (
        statistics.pstdev(host_deltas) * 1e3 if len(host_deltas) > 1 else 0.0
    )

    # ---- backlog ---------------------------------------------------------
    # Host elapsed minus camera elapsed, in frames. The camera clock is the
    # ground truth for when a frame was exposed; the host clock is when it was
    # dequeued. The difference, divided by the frame period, is queue depth.
    def backlog_at(i: int) -> Optional[float]:
        # i == 0 is the reference point, so its backlog is 0 by construction.
        if i < 0 or i >= len(frames):
            return None
        if None in (cam_s[0], cam_s[i], host_s[0], host_s[i]):
            return None
        host_elapsed = host_s[i] - host_s[0]
        cam_elapsed = cam_s[i] - cam_s[0]
        return (host_elapsed - cam_elapsed) / nominal_period_s

    # ---- phase-boundary offsets -----------------------------------------
    marker_ti: dict[str, int] = {}
    for m in markers:
        if m["trigger_index"] is not None:
            marker_ti[m["marker"]] = m["trigger_index"]

    first_frame_of_phase: dict[str, dict] = {}
    first_index_of_phase: dict[str, int] = {}
    for i, f in enumerate(frames):
        for segment, labels in PHASE_LABELS.items():
            if f["phase"] in labels and segment not in first_frame_of_phase:
                first_frame_of_phase[segment] = f
                first_index_of_phase[segment] = i

    boundaries = []
    seen_segments: set[str] = set()
    for marker_name, segment in BOUNDARY_MARKERS:
        if marker_name not in marker_ti:
            continue
        # Prefer the primary spelling; don't report the alias twice.
        if segment in seen_segments:
            continue
        seen_segments.add(segment)

        frame = first_frame_of_phase.get(segment)
        offset = None
        if frame is not None and frame["trigger_index"] is not None:
            offset = frame["trigger_index"] - marker_ti[marker_name]

        idx = first_index_of_phase.get(segment)
        boundaries.append({
            "marker": marker_name,
            "segment": segment,
            "marker_trigger_index": marker_ti[marker_name],
            "first_frame_trigger_index": frame["trigger_index"] if frame else None,
            "offset_triggers": offset,
            "backlog_frames": backlog_at(idx) if idx is not None else None,
        })
        if offset is None:
            result["problems"].append(
                f"{segment}: no trigger index available to check the {marker_name} boundary"
            )
        elif offset != 0:
            result["problems"].append(
                f"{segment}: phase label is off by {offset:+d} trigger(s) "
                f"(marker at {marker_ti[marker_name]}, first labeled frame at "
                f"{frame['trigger_index']})"
            )

    result["boundaries"] = boundaries
    result["end_backlog_frames"] = backlog_at(len(frames) - 1)

    if result["end_backlog_frames"] is not None and result["end_backlog_frames"] >= BACKLOG_LIMIT_FRAMES:
        result["problems"].append(
            f"end-of-trial backlog {result['end_backlog_frames']:.1f} frames "
            f"(limit {BACKLOG_LIMIT_FRAMES:.0f})"
        )

    # ---- dropped frames --------------------------------------------------
    drops = []
    have_drop_column = any(f["trigger_index"] is not None for f in frames)
    for f in frames:
        if f["dropped_before"]:
            drops.append({
                "trigger_index": f["trigger_index"],
                "dropped": f["dropped_before"],
                "phase": f["phase"],
                "filename": f["filename"],
            })

    # Independent check straight off the camera clock, so a session logged by
    # an older build (no dropped_before column) is still audited.
    inferred = []
    for i in range(1, len(frames)):
        if cam_s[i] is None or cam_s[i - 1] is None:
            continue
        gap = cam_s[i] - cam_s[i - 1]
        if gap > 1.5 * nominal_period_s:
            missing = int(round(gap / nominal_period_s)) - 1
            if missing > 0:
                inferred.append({
                    "trigger_index": frames[i]["trigger_index"],
                    "dropped": missing,
                    "phase": frames[i]["phase"],
                    "filename": frames[i]["filename"],
                })

    result["dropped_logged"] = drops
    result["dropped_inferred"] = inferred
    result["dropped_total"] = sum(d["dropped"] for d in (drops if have_drop_column else inferred))

    if result["dropped_total"]:
        result["problems"].append(f"{result['dropped_total']} dropped frame(s)")
    if not have_drop_column:
        result["problems"].append(
            "session_log.csv has no trigger_index column (written by an older build); "
            "drop counts below are inferred from camera timestamps only"
        )

    return result


def print_trial(result: dict) -> None:
    print(f"\n--- Trial {result['trial_index']} ({result['frame_count']} frames) ---")

    def fmt(value, suffix=""):
        return "n/a" if value is None else f"{value:.2f}{suffix}"

    print(
        f"  camera interval : {fmt(result['cam_mean_ms'])} ms "
        f"(SD {fmt(result['cam_sd_ms'])} ms)"
    )
    print(
        f"  host interval   : {fmt(result['host_mean_ms'])} ms "
        f"(SD {fmt(result['host_sd_ms'])} ms)"
    )

    if result["boundaries"]:
        print("  phase boundaries:")
        print(
            f"    {'segment':<18}{'marker ti':>10}{'frame ti':>10}"
            f"{'offset':>8}{'backlog':>10}"
        )
        for b in result["boundaries"]:
            offset = "n/a" if b["offset_triggers"] is None else f"{b['offset_triggers']:+d}"
            backlog = "n/a" if b["backlog_frames"] is None else f"{b['backlog_frames']:.1f}"
            frame_ti = "n/a" if b["first_frame_trigger_index"] is None else b["first_frame_trigger_index"]
            print(
                f"    {b['segment']:<18}{b['marker_trigger_index']:>10}"
                f"{frame_ti:>10}{offset:>8}{backlog:>10}"
            )
    else:
        print("  phase boundaries: no trigger indices in marker_log.csv (older firmware)")

    end_backlog = result["end_backlog_frames"]
    print(f"  end-of-trial backlog: {'n/a' if end_backlog is None else f'{end_backlog:.1f} frames'}")

    drops = result["dropped_logged"] or result["dropped_inferred"]
    if drops:
        print(f"  dropped frames  : {result['dropped_total']}")
        for d in drops:
            where = d["filename"] or "(not saved)"
            print(
                f"    -{d['dropped']} before trigger {d['trigger_index']} "
                f"in {d['phase']} [{where}]"
            )
    else:
        print("  dropped frames  : 0")

    for problem in result["problems"]:
        print(f"  PROBLEM: {problem}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Report acquisition timing, backlog, drops, and phase-label offsets."
    )
    parser.add_argument(
        "session_dir", type=Path, nargs="?", default=None,
        help="Session folder containing session_log.csv and marker_log.csv.",
    )
    parser.add_argument("--session-log", type=Path, default=None)
    parser.add_argument("--marker-log", type=Path, default=None)
    parser.add_argument(
        "--trigger-period-ms", type=float, default=100.0,
        help="Nominal trigger period; must match the Arduino's triggerPeriodMs.",
    )
    parser.add_argument(
        "--camera-timestamp-hz", type=float, default=None,
        help="Camera timestamp ticks per second. Auto-detected when omitted.",
    )
    args = parser.parse_args(argv)

    if args.session_dir is not None:
        session_log = args.session_log or args.session_dir / "session_log.csv"
        marker_log = args.marker_log or args.session_dir / "marker_log.csv"
    else:
        session_log = args.session_log
        marker_log = args.marker_log

    if session_log is None or not Path(session_log).exists():
        parser.error("session_log.csv not found; pass a session folder or --session-log")

    nominal_period_s = args.trigger_period_ms / 1000.0

    frames = load_session_log(Path(session_log))
    markers = load_marker_log(Path(marker_log)) if marker_log else []

    if args.camera_timestamp_hz:
        hz = args.camera_timestamp_hz
        median_s = 0.0
    else:
        hz, median_s = detect_timestamp_hz(frames, nominal_period_s)

    print(f"session log : {session_log}")
    print(f"marker log  : {marker_log if markers else '(none)'}")
    print(f"nominal frame period: {args.trigger_period_ms:.1f} ms")
    print(
        f"camera timestamp scale: {hz:.3g} ticks/s"
        + ("" if args.camera_timestamp_hz else f" (auto-detected; median delta {median_s * 1e3:.2f} ms)")
    )

    # trial_index 0 is the session green reference block, not a timed trial.
    trial_indices = sorted({f["trial_index"] for f in frames if f["trial_index"] > 0})
    if not trial_indices:
        print("\nNo timed trials found in session_log.csv.")
        return 1

    results = []
    for trial_index in trial_indices:
        trial_frames = [f for f in frames if f["trial_index"] == trial_index]
        trial_markers = [m for m in markers if m["trial_index"] == trial_index]
        result = analyze_trial(trial_index, trial_frames, trial_markers, hz, nominal_period_s)
        results.append(result)
        print_trial(result)

    total_drops = sum(r["dropped_total"] for r in results)
    all_boundaries = [b for r in results for b in r["boundaries"]]
    bad_offsets = sum(1 for b in all_boundaries if b["offset_triggers"] not in (0, None))
    # Counted separately and never folded into "0 bad offsets": an offset that
    # could not be computed is an unverified boundary, not a passing one.
    unverified = sum(1 for b in all_boundaries if b["offset_triggers"] is None)
    worst_backlog = max(
        (r["end_backlog_frames"] for r in results if r["end_backlog_frames"] is not None),
        default=None,
    )

    print("\n=== Session summary ===")
    print(f"  trials analyzed          : {len(results)}")
    print(f"  total dropped frames     : {total_drops}   (target 0)")
    print(f"  phase boundaries checked : {len(all_boundaries) - unverified} of {len(all_boundaries)}")
    print(f"  non-zero phase offsets   : {bad_offsets}   (target 0)")
    if unverified or not all_boundaries:
        print(f"  UNVERIFIED boundaries    : {unverified}   (need trigger_index in both logs)")
    print(
        f"  worst end-trial backlog  : "
        f"{'n/a' if worst_backlog is None else f'{worst_backlog:.1f} frames'}"
        f"   (target < {BACKLOG_LIMIT_FRAMES:.0f})"
    )

    passed = (
        total_drops == 0
        and bad_offsets == 0
        and unverified == 0
        and bool(all_boundaries)
        and worst_backlog is not None
        and worst_backlog < BACKLOG_LIMIT_FRAMES
    )
    print(f"\n  ACCEPTANCE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
