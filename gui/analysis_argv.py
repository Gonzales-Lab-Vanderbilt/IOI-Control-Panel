# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Analysis argv builders — pure functions that turn a frozen snapshot of the
Statistics form into the argv lists statistical_analyses.py /
session_poster_figures.py expect.

Split out of gui/statistics_tab.py so a queued AnalysisJob can build its
command line from a snapshot taken at "Run analysis" time, instead of
re-reading live widget state that may have changed by the time the job's
turn comes up. resolve_out_dir() is the single source of truth for a job's
output directory — used both to build --out-dir and, before any subprocess
is launched, by AnalysisQueue's same-output-folder collision guard, so the
two can never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gui.paths import project_root

_PROJECT_ROOT = project_root()
STATS_SCRIPT = str(_PROJECT_ROOT / "statistical_analyses.py")
POSTER_SCRIPT = str(_PROJECT_ROOT / "session_poster_figures.py")
DEFAULT_OUT_SUBFOLDER = "session_stats"


@dataclass(frozen=True)
class StatsFormSnapshot:
    session_dir: str
    full_frame: bool
    neg_only: bool
    condition: str  # "all" | "stim" | "catch"
    excluded_trial_ids: tuple[int, ...]
    n_perm: int
    crop_rect: tuple[int, int, int, int] | None
    no_crop: bool
    roi_within_region: bool
    array_source_dir: str
    log_dir: str
    log_time_min: str
    log_time_max: str
    out_dir_override: str
    out_subfolder: str
    reuse_timecourse_cache: bool
    save_loo_masks: bool


@dataclass(frozen=True)
class PosterFormSnapshot:
    compare_session: str
    compare_label: str
    um_per_px: float | None  # None: rig calibration not set -> no scale bar
    anterior_side: str
    medial_side: str
    midline_centered: bool
    fixed_vmax: float
    reuse_extraction_cache: bool
    suppress_title: bool
    show_amplitude_labels: bool


def resolve_out_dir(snap: StatsFormSnapshot) -> Path | None:
    """Deterministic --out-dir for this snapshot, or None if no session
    folder is set yet. Mirrors statistical_analyses.py's own --out-suffix
    default being dead from the GUI's perspective since --out-dir always
    wins when given."""
    if snap.out_dir_override:
        return Path(snap.out_dir_override)
    if not snap.session_dir:
        return None
    subfolder = snap.out_subfolder or DEFAULT_OUT_SUBFOLDER
    if snap.condition != "all":
        subfolder = f"{subfolder}_{snap.condition}"
    return Path(snap.session_dir) / subfolder


def build_stats_argv(snap: StatsFormSnapshot) -> list[str]:
    args: list[str] = []

    if snap.session_dir:
        args.append(snap.session_dir)

    if snap.full_frame:
        args.append("--full-frame")

    if snap.neg_only:
        args.append("--neg-only")

    if snap.condition != "all":
        args += ["--condition", snap.condition]

    if snap.excluded_trial_ids:
        args += ["--exclude-trials", ",".join(str(t) for t in snap.excluded_trial_ids)]

    args += ["--n-perm", str(snap.n_perm)]

    if snap.crop_rect is not None:
        x, y, w, h = snap.crop_rect
        args += ["--shared-crop", f"{x},{y},{w},{h}"]

    if snap.no_crop:
        args.append("--no-crop")

    if snap.roi_within_region:
        args.append("--roi-within-region")

    if snap.array_source_dir:
        args += ["--array-source-dir", snap.array_source_dir]

    if snap.log_dir:
        args += ["--log-dir", snap.log_dir]

    if snap.log_time_min:
        args += ["--log-time-min", snap.log_time_min]

    if snap.log_time_max:
        args += ["--log-time-max", snap.log_time_max]

    out_dir = resolve_out_dir(snap)
    if out_dir is not None:
        args += ["--out-dir", str(out_dir)]

    if snap.reuse_timecourse_cache:
        args.append("--reuse-timecourse-cache")

    if snap.save_loo_masks:
        args.append("--save-loo-masks")

    return args


def build_poster_argv(
    session_dir: str, output_dir: str, poster: PosterFormSnapshot, *,
    compare_session_override: str | None = None,
    compare_stats_dir_override: str | None = None,
    compare_label_override: str | None = None,
) -> list[str]:
    """--stats-dir / --out-dir both point at output_dir -- the folder
    statistical_analyses.py itself just reported via "Done -> ...", so this
    targets exactly where the just-completed run wrote, whatever
    resolve_out_dir() resolved to, without duplicating that logic here.

    The compare_*_override args let a caller (see AnalysisJob.compare_job)
    wire an interleaved session's stim job to compare against its own catch
    job's statistics, which live in a condition-suffixed subfolder rather
    than COMPARE_SESSION/session_stats -- so --compare-stats-dir has to be
    given explicitly. They take priority over the form's own (blank, in
    that case) Compare session field."""
    args = [session_dir]
    args += ["--stats-dir", output_dir]
    args += ["--out-dir", output_dir]

    compare_session = compare_session_override or poster.compare_session
    if compare_session:
        args += ["--compare-session", compare_session]
        if compare_stats_dir_override:
            args += ["--compare-stats-dir", compare_stats_dir_override]
        compare_label = compare_label_override or poster.compare_label
        if compare_label:
            args += ["--compare-label", compare_label]

    if poster.um_per_px is None:
        args.append("--no-scale-bar")
    else:
        args += ["--um-per-px", str(poster.um_per_px)]
    args += ["--anterior-side", poster.anterior_side]
    if poster.midline_centered:
        args.append("--midline-centered")
    else:
        args += ["--medial-side", poster.medial_side]
    args += ["--fixed-vmax", str(poster.fixed_vmax)]

    if poster.reuse_extraction_cache:
        args.append("--reuse-extraction-cache")

    if poster.suppress_title:
        args.append("--no-title")

    if poster.show_amplitude_labels:
        args.append("--show-amplitude-labels")

    return args
