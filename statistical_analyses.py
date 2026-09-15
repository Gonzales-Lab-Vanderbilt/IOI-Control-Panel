#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
General version of the ROI time-course / LOO-ROI / pixelwise-t-map-with-
permutation pipeline, parameterized by session folder so it can be pointed
at any session that has per-trial trial_XXX/analysis/{baseline_reference,
post_mean_analysis_window}.npy already saved (i.e. run_analysis was on).

Usage:
    python3 statistical_analyses.py /path/to/session_dir

Robustness differences vs the earlier session-specific version:
  - Trial count is auto-detected from trial_XXX folders on disk.
  - Per-frame timestamps still come from session_log.csv, but STIM_START for
    each trial is picked as whichever marker_log.csv STIM_START row (for that
    trial_index) has the smallest time gap to that trial's own baseline
    frames -- this is robust to any duplicated/appended log blocks (seen in
    at least one prior session) without needing a manually-picked cutoff.
  - Trials with no matched STIM_START (e.g. an aborted last trial) are
    skipped from the time-course/LOO analysis, with a printed warning.

Writes ONLY to a new sibling folder '<session_name>_reanalysis'; never
touches anything in the session folder itself.
"""

import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.ndimage import gaussian_filter, label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 1920, 1200
SMOOTH_SIGMA = 5.0
N_PERM = 1000
CLUSTER_T_THRESH = 3.0
# PROVISIONAL ONLY -- used for the per-trial progress print during the streaming
# LOO pass. The authoritative amplitude window is frame-indexed (post frames
# analysis_start_frame..analysis_end_frame-1) and is applied after that pass;
# it overwrites every value computed with this constant.
PEAK_WINDOW_S = (1.0, 4.0)


def _camera_frame_interval(log_dir, t_min=None) -> float:
    """Median camera inter-frame interval (s), from the CAMERA hardware clock.
    Host timestamps carry logging latency (up to ~19% inflation) and must not be
    used for anything quantitative."""
    c = []
    with open(Path(log_dir) / "session_log.csv") as fp:
        for r in csv.DictReader(fp):
            if t_min and r["host_timestamp_iso"] < t_min:
                continue
            if r["phase"] == "baseline_collect":
                c.append(int(r["camera_timestamp"]))
            elif c:
                break
    return float(np.median(np.diff(c)) / 1e9)


def _phase_offsets(log_dir, t_min=None):
    """Median (last baseline frame, first post frame) offsets in seconds relative
    to STIM_START, taken from the Arduino marker clock."""
    per = {}
    with open(Path(log_dir) / "marker_log.csv") as fp:
        for r in csv.DictReader(fp):
            if t_min and r["host_timestamp_iso"] < t_min:
                continue
            if r["trial_index"] == "0":
                continue
            millis_str = r["arduino_millis"]
            if not millis_str:
                # Writer quirk seen in some sessions: arduino_millis is left blank
                # for MARKER,millis,trigger_index lines (3-field firmware format),
                # but the value is always still there as raw_text's 2nd field.
                millis_str = r["raw_text"].split(",")[1].strip()
            per.setdefault(r["trial_index"], {})[r["marker"]] = int(millis_str)
    need = {"STIM_START", "BASELINE_LAST_FRAME_TRIGGER", "POST_FIRST_FRAME_TRIGGER"}
    bl, pf = [], []
    for m in per.values():
        if need <= set(m):
            bl.append((m["BASELINE_LAST_FRAME_TRIGGER"] - m["STIM_START"]) / 1000.0)
            pf.append((m["POST_FIRST_FRAME_TRIGGER"] - m["STIM_START"]) / 1000.0)
    return float(np.median(bl)), float(np.median(pf))


def parse_ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def load_raw(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint16)
    assert data.size == WIDTH * HEIGHT, f"bad size {data.size} for {path}"
    return data.reshape(HEIGHT, WIDTH).astype(np.float64)


def main(session_dir: Path, full_frame: bool = False, shared_crop=None, out_suffix: str = "_reanalysis",
         log_dir: Path = None, log_time_min: str = None, log_time_max: str = None,
         no_crop: bool = False, array_source_dir: Path = None, n_perm: int = None,
         reuse_timecourse_cache: bool = False, save_loo_masks: bool = False,
         out_dir_override=None, neg_only=False, condition: str = "all",
         roi_within_region: bool = False, exclude_trials=None):
    """
    log_dir: directory containing session_log.csv / marker_log.csv, if
        different from session_dir (some sessions' camera logs got appended
        into the PRECEDING session's log files instead of their own -- point
        this at that folder in that case).
    log_time_min / log_time_max: optional ISO timestamp strings to restrict
        which rows of the (possibly shared/multi-session) log files are
        considered, in case one log file contains more than one session's
        worth of rows back to back.
    no_crop: if True, ignore any manual active_roi crop recorded in each
        trial's metadata entirely and treat the analysis region as the full
        1920x1200 sensor capture. Requires array_source_dir to point at
        pre-computed, uncropped baseline_reference.npy/post_mean_analysis_
        window.npy per trial (see compute_fullframe_arrays.py), since the
        arrays saved by the original acquisition pipeline are already
        cropped and can't be un-cropped after the fact.
    array_source_dir: directory containing trial_XXX/{baseline_reference,
        post_mean_analysis_window}.npy to use INSTEAD of the ones under
        session_dir/trial_XXX/analysis/. Use with no_crop for a true
        full-uncropped-frame re-analysis.
    n_perm: override for the number of sign-flip permutations (default
        module-level N_PERM if not given).
    full_frame: if True, the ROI for the time-course/amplitude metric is the
        ENTIRE imaged window (or shared_crop, if given) instead of a
        data-driven top-5%-most-negative-pixel mask. This is a completely
        assumption-free, non-circular metric by construction (no pixel
        selection at all), useful for comparing stim vs. no-stim sessions
        on equal footing.
    condition: "all" (default, today's behavior), "stim", or "catch". For
        sessions with interleaved trials (trial_metadata.json's
        trial_condition field set per trial by --trial-conditions at
        acquisition time), restricts usable_trials to that condition before
        any ROI/dR/R/t-map computation -- so a catch trial can never
        contaminate the stim condition's ROI selection or vice versa.
        Raises SystemExit if any usable trial's metadata predates this field.
        Non-"all" values append "_{condition}" to out_suffix so running a
        session twice (stim, catch) doesn't collide.
    shared_crop: optional (x, y, w, h) in FULL-RESOLUTION pixel coordinates.
        When given, both the ROI mask (time-course) and the pixelwise
        t-map/permutation test are restricted to this rectangle instead of
        this session's own analysis crop. Use this to force two sessions
        recorded on the same animal/day but with slightly different manual
        crops to be compared over the exact same physical field of view
        (pass the intersection of their two crops).
    exclude_trials: optional iterable of trial numbers to drop after the
        usual frames/STIM_START/--condition filtering, before any ROI/dR/R/
        t-map computation -- e.g. a trial contaminated by a bubble under the
        cranial window. Trial numbers not present in the usable set are
        ignored with a warning rather than raising, so a stale exclusion
        list from an earlier run of this session doesn't break a re-run.
    """
    session_dir = Path(session_dir)
    log_dir = Path(log_dir) if log_dir is not None else session_dir
    # Normalized here (rather than down where usable_trials is filtered) so the
    # requested exclusion set can also disambiguate out_dir below -- otherwise a
    # run with trials excluded and a run without would silently share an output
    # folder (like --condition already avoids), which then lets a stale
    # --reuse-timecourse-cache hit from the other run leak pre-exclusion numbers
    # back into this run's summary.
    exclude_trials = set(exclude_trials) if exclude_trials else set()
    effective_out_suffix = out_suffix if condition == "all" else f"{out_suffix}_{condition}"
    if exclude_trials:
        effective_out_suffix += "_excl" + "-".join(str(t) for t in sorted(exclude_trials))
    out_dir = Path(out_dir_override) if out_dir_override else \
        session_dir.parent / f"{session_dir.name}{effective_out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if log_dir != session_dir:
        print(f"Reading session_log.csv / marker_log.csv from: {log_dir}")
    if log_time_min or log_time_max:
        print(f"Log time window filter: [{log_time_min}, {log_time_max})")
    array_source_dir = Path(array_source_dir) if array_source_dir is not None else None
    n_perm_actual = int(n_perm) if n_perm is not None else N_PERM
    if no_crop:
        print("no_crop=True: ignoring any manual active_roi crop, using full 1920x1200 sensor frame.")
    if array_source_dir is not None:
        print(f"Reading baseline_reference/post_mean_analysis_window arrays from: {array_source_dir}")

    # trial_[0-9]* rather than trial_* -- a session with interleaved/manual
    # trial conditions also has a trial_conditions.json manifest sitting
    # directly in session_dir, which "trial_*" matches too, crashing the
    # int(p.name.split("_")[1]) below on "conditions.json".
    trial_dirs = sorted(session_dir.glob("trial_[0-9]*"))
    n_trials_on_disk = len(trial_dirs)
    trial_ids = [int(p.name.split("_")[1]) for p in trial_dirs]
    print(f"Session: {session_dir.name}  trial folders on disk: {n_trials_on_disk}")

    # Only keep trials that actually have the saved per-trial analysis arrays
    usable_trials = []
    for t in trial_ids:
        td = (array_source_dir / f"trial_{t:03d}") if array_source_dir is not None \
            else (session_dir / f"trial_{t:03d}" / "analysis")
        if (td / "baseline_reference.npy").exists() and (td / "post_mean_analysis_window.npy").exists():
            usable_trials.append(t)
    if len(usable_trials) < n_trials_on_disk:
        missing = sorted(set(trial_ids) - set(usable_trials))
        print(f"WARNING: {len(missing)} trial(s) missing saved analysis arrays, skipped: {missing}")

    # -----------------------------------------------------------------
    # frame -> (timestamp, filename) tables per trial, from session_log.csv
    # -----------------------------------------------------------------
    frames_by_trial = {t: {"baseline": [], "post": [], "gap": []} for t in usable_trials}
    with open(log_dir / "session_log.csv", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                trial = int(row["trial_index"])
            except ValueError:
                continue
            if trial not in frames_by_trial:
                continue
            fname = row["filename"].strip()
            if not fname:
                continue
            phase = row["phase"]
            ts = row["host_timestamp_iso"]
            if log_time_min and ts < log_time_min:
                continue
            if log_time_max and ts >= log_time_max:
                continue
            if phase == "baseline_collect" and fname.startswith("baseline_"):
                frames_by_trial[trial]["baseline"].append((ts, fname))
            elif phase == "post_collect" and fname.startswith("post_"):
                frames_by_trial[trial]["post"].append((ts, fname))
            elif phase == "gap" and fname.startswith("gap_"):
                frames_by_trial[trial]["gap"].append((ts, fname))

    for t in list(frames_by_trial.keys()):
        b = sorted(set(frames_by_trial[t]["baseline"]), key=lambda x: x[0])
        p = sorted(set(frames_by_trial[t]["post"]), key=lambda x: x[0])
        g = sorted(set(frames_by_trial[t]["gap"]), key=lambda x: x[0])
        # de-dupe by filename in case of an appended/duplicated log block:
        # keep the timestamp closest to the OTHER frames of this same trial
        def dedupe_by_filename(rows):
            by_fname = {}
            for ts, fname in rows:
                by_fname.setdefault(fname, []).append(ts)
            out = []
            for fname, ts_list in by_fname.items():
                # if duplicated, pick median timestamp's row (robust-ish);
                # with a single dup block this just disambiguates deterministically
                ts_list.sort()
                out.append((ts_list[len(ts_list) // 2], fname))
            out.sort(key=lambda x: x[1])  # filename order == chronological order
            return out
        b2 = dedupe_by_filename(b)
        p2 = dedupe_by_filename(p)
        g2 = dedupe_by_filename(g)
        frames_by_trial[t]["baseline"] = sorted(b2, key=lambda x: x[0])
        frames_by_trial[t]["post"] = sorted(p2, key=lambda x: x[0])
        frames_by_trial[t]["gap"] = sorted(g2, key=lambda x: x[0])

    # drop trials that don't have exactly the expected 40/40 frames logged
    for t in list(frames_by_trial.keys()):
        nb = len(frames_by_trial[t]["baseline"])
        npst = len(frames_by_trial[t]["post"])
        if nb == 0 or npst == 0:
            print(f"  trial {t}: no baseline/post frame log rows found, dropping ({nb}/{npst})")
            del frames_by_trial[t]

    # -----------------------------------------------------------------
    # STIM_START per trial: pick, among all marker_log.csv STIM_START rows
    # for that trial_index, the one closest in time to this trial's own
    # baseline frames (robust to duplicated/appended log blocks).
    # -----------------------------------------------------------------
    stim_candidates = {}
    with open(log_dir / "marker_log.csv", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            ts = row["host_timestamp_iso"]
            if log_time_min and ts < log_time_min:
                continue
            if log_time_max and ts >= log_time_max:
                continue
            if row["marker"] == "STIM_START":
                trial = int(row["trial_index"])
                stim_candidates.setdefault(trial, []).append(ts)

    stim_start = {}
    for t, frames in frames_by_trial.items():
        if t not in stim_candidates:
            continue
        baseline_ref_time = parse_ts(frames["baseline"][-1][0])  # last baseline frame time
        best = min(stim_candidates[t], key=lambda ts: abs(parse_ts(ts) - baseline_ref_time))
        stim_start[t] = parse_ts(best)

    usable_trials = sorted(set(frames_by_trial.keys()) & set(stim_start.keys()))
    dropped = sorted(set(trial_ids) - set(usable_trials))
    if dropped:
        print(f"Dropped trials (no matched frames/STIM_START): {dropped}")
    print(f"Usable trials (frames + STIM_START matched): {len(usable_trials)} -> {usable_trials}")

    if condition != "all":
        filtered = []
        missing_field = []
        for t in usable_trials:
            meta_path = session_dir / f"trial_{t:03d}" / "meta" / "trial_metadata.json"
            try:
                trial_cond = json.loads(meta_path.read_text()).get("trial_condition")
            except (OSError, ValueError):
                trial_cond = None
            if trial_cond is None:
                missing_field.append(t)
            elif trial_cond == condition:
                filtered.append(t)
        if missing_field:
            raise SystemExit(
                f"--condition {condition} requires trial_condition in every usable "
                f"trial's trial_metadata.json, but trial(s) {missing_field} don't "
                f"have it -- this session predates interleaved-trial support. "
                f"Re-run with --condition all for this session."
            )
        print(f"Condition filter '{condition}': {len(filtered)}/{len(usable_trials)} "
              f"usable trials kept -> {filtered}")
        usable_trials = filtered

    excluded_present: list[int] = []
    if exclude_trials:
        excluded_present = sorted(exclude_trials & set(usable_trials))
        excluded_absent = sorted(exclude_trials - set(usable_trials))
        if excluded_absent:
            print(f"NOTE: --exclude-trials names trial(s) not in the usable set, ignored: {excluded_absent}")
        if excluded_present:
            print(f"Manually excluded trial(s) (e.g. bubble contamination): {excluded_present}")
            usable_trials = sorted(set(usable_trials) - set(excluded_present))

    n = len(usable_trials)
    if n < 3:
        print("Too few usable trials, aborting.")
        return

    # -----------------------------------------------------------------
    # per-trial dR/R maps from saved arrays (fast, no raw frame reads needed)
    # -----------------------------------------------------------------
    def trial_drr_map(trial: int) -> np.ndarray:
        src = (array_source_dir / f"trial_{trial:03d}") if array_source_dir is not None \
            else (session_dir / f"trial_{trial:03d}" / "analysis")
        b = np.load(src / "baseline_reference.npy")
        p = np.load(src / "post_mean_analysis_window.npy")
        safe_b = np.where(b > 0, b, np.nan)
        drr = (p - b) / safe_b
        return gaussian_filter(np.nan_to_num(drr, nan=0.0), sigma=SMOOTH_SIGMA)

    print("Computing per-trial dR/R maps from saved arrays...")
    all_maps = {t: trial_drr_map(t) for t in usable_trials}
    map_shape = next(iter(all_maps.values())).shape  # (Hb, Wb) at pipeline's analysis_binning

    # Some sessions crop to a software ROI *before* binning (active_roi in
    # trial_metadata.json). Read the crop bounds so the binned mask gets
    # placed back at the right offset in the full 1200x1920 raw frame.
    def get_crop_bounds(trial):
        meta_path = session_dir / f"trial_{trial:03d}" / "meta" / "trial_metadata.json"
        x0, y0, w, h = 0, 0, WIDTH, HEIGHT
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            active_roi = meta.get("active_roi")
            if active_roi:
                crop = active_roi.get("analysis_crop") or active_roi.get("requested")
                if crop:
                    x0, y0, w, h = int(crop["x"]), int(crop["y"]), int(crop["width"]), int(crop["height"])
        return x0, y0, w, h

    if no_crop:
        crop_x0, crop_y0, crop_w, crop_h = 0, 0, WIDTH, HEIGHT
    else:
        crop_x0, crop_y0, crop_w, crop_h = get_crop_bounds(usable_trials[0])
    print(f"Analysis crop (full-res px): x={crop_x0} y={crop_y0} w={crop_w} h={crop_h}")

    # The rectangle actually used for ROI/analysis purposes: either this
    # session's own crop, or an externally-supplied shared rectangle
    # (e.g. the intersection with a same-day/same-mouse paired session),
    # clipped to lie within this session's own available crop.
    if shared_crop is not None:
        sx0, sy0, sw, sh = shared_crop
        rx0 = max(sx0, crop_x0)
        ry0 = max(sy0, crop_y0)
        rx1 = min(sx0 + sw, crop_x0 + crop_w)
        ry1 = min(sy0 + sh, crop_y0 + crop_h)
        region_x0, region_y0 = rx0, ry0
        region_w, region_h = max(0, rx1 - rx0), max(0, ry1 - ry0)
        print(f"Shared-crop region (full-res px, clipped to own crop): "
              f"x={region_x0} y={region_y0} w={region_w} h={region_h}")
    else:
        region_x0, region_y0, region_w, region_h = crop_x0, crop_y0, crop_w, crop_h

    def make_roi_from_maps(maps, top_frac=0.05):
        if full_frame:
            full_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
            full_mask[region_y0:region_y0 + region_h, region_x0:region_x0 + region_w] = True
            return full_mask
        mean_map = np.mean(maps, axis=0)
        if roi_within_region:
            # Restrict ROI *candidate pixels* to the analysis
            # region. Without this, --shared-crop restricts only the pixelwise
            # t-map/permutation test; the top-5%-most-negative ROI is still
            # selected over the whole binned map, so a shared crop applied to
            # exclude an artifact (e.g. a bubble) does not keep the ROI out of
            # it, and the amplitude/time-course numbers come out bit-identical
            # to the uncropped run. When shared_crop is None the region equals
            # the session's own acquisition crop, which is exactly the extent of
            # mean_map, so this branch is a no-op and reproduces historical
            # results exactly. Off by default; enable with --roi-within-region.
            _by0, _by1, _bx0, _bx1 = local_binned_slice_for_region()
            sub = mean_map[_by0:_by1, _bx0:_bx1]
            thresh = np.percentile(sub, top_frac * 100.0)
            mask_binned = np.zeros(mean_map.shape, dtype=bool)
            mask_binned[_by0:_by1, _bx0:_bx1] = sub <= thresh
        else:
            thresh = np.percentile(mean_map, top_frac * 100.0)
            mask_binned = mean_map <= thresh
        by = crop_h // map_shape[0]
        bx = crop_w // map_shape[1]
        mask_cropres = np.repeat(np.repeat(mask_binned, by, axis=0), bx, axis=1)
        full_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        h_avail, w_avail = mask_cropres.shape
        full_mask[crop_y0:crop_y0 + h_avail, crop_x0:crop_x0 + w_avail] = mask_cropres
        return full_mask

    def local_binned_slice_for_region():
        """Map the full-res `region_*` rectangle into this session's own
        binned dR/R map coordinates (map_shape), for restricting the
        pixelwise t-map/permutation to the same region."""
        by_factor = crop_h // map_shape[0]
        bx_factor = crop_w // map_shape[1]
        lx0 = max(region_x0 - crop_x0, 0)
        ly0 = max(region_y0 - crop_y0, 0)
        lx1 = min(region_x0 + region_w - crop_x0, crop_w)
        ly1 = min(region_y0 + region_h - crop_y0, crop_h)
        bx0, by0 = lx0 // bx_factor, ly0 // by_factor
        bx1, by1 = lx1 // bx_factor, ly1 // by_factor
        by1 = min(by1, map_shape[0]); bx1 = min(bx1, map_shape[1])
        return by0, by1, bx0, bx1

    def roi_mean_timecourse(trial: int, roi_mask: np.ndarray):
        """Returns (times, values, gap_times, gap_values). The first pair is
        baseline+post only, in the exact order/length the frame-indexed
        amplitude/FWHM code below depends on -- unaffected by whether this
        trial has gap frames. The second pair is the inter-phase gap frames
        (if any were saved), dR/R against the same r_bar_baseline; kept
        separate rather than spliced into the first pair so a session with
        no gap frames, or a trial with a partial gap capture, can't silently
        change that array's length/index alignment."""
        trial_dir = session_dir / f"trial_{trial:03d}"
        baseline_files = frames_by_trial[trial]["baseline"]
        post_files = frames_by_trial[trial]["post"]
        gap_files = frames_by_trial[trial]["gap"]
        t0 = stim_start[trial]

        baseline_roi_vals = []
        for ts, fname in baseline_files:
            img = load_raw(trial_dir / "baseline" / fname)
            baseline_roi_vals.append(img[roi_mask].mean())
        r_bar_baseline = float(np.mean(baseline_roi_vals))

        times, values = [], []
        for (ts, fname), r in zip(baseline_files, baseline_roi_vals):
            times.append(parse_ts(ts) - t0)
            values.append(100.0 * (r - r_bar_baseline) / r_bar_baseline)
        for ts, fname in post_files:
            img = load_raw(trial_dir / "post" / fname)
            r = img[roi_mask].mean()
            times.append(parse_ts(ts) - t0)
            values.append(100.0 * (r - r_bar_baseline) / r_bar_baseline)

        gap_times, gap_values = [], []
        for ts, fname in gap_files:
            img = load_raw(trial_dir / "gap" / fname)
            r = img[roi_mask].mean()
            gap_times.append(parse_ts(ts) - t0)
            gap_values.append(100.0 * (r - r_bar_baseline) / r_bar_baseline)

        order = np.argsort(times)
        gap_order = np.argsort(gap_times)
        return (np.array(times)[order], np.array(values)[order],
                np.array(gap_times)[gap_order], np.array(gap_values)[gap_order])

    def upsample_map_to_fullres(binned_img: np.ndarray) -> np.ndarray:
        """Same placement logic as make_roi_from_maps, but for a continuous-
        valued image (e.g. an anatomy reference) instead of a boolean mask."""
        by = crop_h // map_shape[0]
        bx = crop_w // map_shape[1]
        img_cropres = np.repeat(np.repeat(binned_img, by, axis=0), bx, axis=1)
        full_img = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
        h_avail, w_avail = img_cropres.shape
        full_img[crop_y0:crop_y0 + h_avail, crop_x0:crop_x0 + w_avail] = img_cropres
        return full_img

    if save_loo_masks:
        masks_dir = out_dir / "loo_masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving LOO masks for {len(usable_trials)} folds to {masks_dir} ...")
        mask_stack = np.zeros((len(usable_trials), HEIGHT, WIDTH), dtype=bool)
        for i, held_out in enumerate(usable_trials):
            training_maps = [all_maps[t] for t in usable_trials if t != held_out]
            mask_stack[i] = make_roi_from_maps(training_maps, top_frac=0.05)
        np.save(masks_dir / "loo_mask_stack.npy", mask_stack)
        np.save(masks_dir / "loo_mask_trial_ids.npy", np.array(usable_trials))

        freq_map = mask_stack.mean(axis=0)  # fraction of folds (0..1) each pixel was selected in
        np.save(masks_dir / "loo_mask_fold_frequency.npy", freq_map)

        def src_for(trial):
            return (array_source_dir / f"trial_{trial:03d}") if array_source_dir is not None \
                else (session_dir / f"trial_{trial:03d}" / "analysis")
        anatomy_binned = np.mean([np.load(src_for(t) / "baseline_reference.npy") for t in usable_trials], axis=0)
        anatomy_full = upsample_map_to_fullres(anatomy_binned)
        anatomy_disp = np.clip((anatomy_full - np.percentile(anatomy_full, 1)) /
                                (np.percentile(anatomy_full, 99) - np.percentile(anatomy_full, 1) + 1e-9), 0, 1)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(anatomy_disp, cmap="gray")
        im = ax.imshow(freq_map, cmap="viridis", alpha=0.55, vmin=0, vmax=1)
        ax.set_title(f"{session_dir.name}: LOO ROI selection frequency across {len(usable_trials)} folds")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="fraction of folds this pixel was in the top-5% mask")
        fig.tight_layout()
        fig.savefig(masks_dir / "loo_mask_fold_frequency.png", dpi=200)
        plt.close(fig)

        n_show = min(6, len(usable_trials))
        show_idx = np.linspace(0, len(usable_trials) - 1, n_show).astype(int)
        fig, axes = plt.subplots(1, n_show, figsize=(3.2 * n_show, 3.6))
        if n_show == 1:
            axes = [axes]
        for ax, idx in zip(axes, show_idx):
            t = usable_trials[idx]
            ax.imshow(anatomy_disp, cmap="gray")
            ax.imshow(np.ma.masked_where(~mask_stack[idx], mask_stack[idx]), cmap="autumn", alpha=0.6)
            ax.set_title(f"held-out trial {t}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{session_dir.name}: example individual LOO fold masks (top 5% of the OTHER "
                     f"{len(usable_trials)-1} trials' mean map)")
        fig.tight_layout()
        fig.savefig(masks_dir / "loo_mask_example_folds.png", dpi=200)
        plt.close(fig)
        print(f"  saved loo_mask_stack.npy ({mask_stack.shape}), fold_frequency map, and example-fold figure")

    loo_results = {}
    raw_curves = {}
    gap_curves = {}
    common_grid = np.linspace(-6, 8, 280)
    interp_curves = []

    cache_npz = out_dir / "roi_timecourse_raw.npz"
    cache_csv = out_dir / "loo_roi_amplitude_summary.csv"
    reused_cache = False
    if reuse_timecourse_cache and cache_npz.exists() and cache_csv.exists():
        print(f"Reusing cached LOO time-course results from {out_dir} "
              "(skipping raw-frame streaming; only rebuilding the t-map/permutation below).")
        reused_cache = True
        cached = np.load(cache_npz)
        common_grid = cached["common_grid"]
        interp_curves = cached["interp_curves"]

        # NOTE: loo_roi_amplitude_summary.csv exists in two schemas on disk.
        # This script writes
        #     trial,roi_size_px,peak_drr_percent,time_to_peak_s,fwhm_s
        # while finalize_amplitude_and_timecourse.py rewrites the same filename as
        #     trial,roi_size_px,mean_drr_percent,time_to_min_s
        # with no fwhm column. The two amplitude columns hold the same quantity
        # (mean dR/R over the fixed analysis window) and the two time columns are
        # both the time of the minimum inside that window, so either can be read.
        # Previously this branch hard-coded the first schema and died with
        # KeyError: 'peak_drr_percent' on any session that finalize had touched.
        def _pick(row, names, cast=float, default=None):
            for nm in names:
                if nm in row and row[nm] not in ("", None):
                    return cast(row[nm])
            if default is None:
                raise KeyError(
                    f"none of {names} found in {cache_csv.name}; "
                    f"columns present: {list(row)}"
                )
            return default

        with open(cache_csv, newline="") as fp:
            for row in csv.DictReader(fp):
                t = int(row["trial"])
                loo_results[t] = dict(
                    roi_size_px=_pick(row, ["roi_size_px"], int),
                    peak_drr_percent=_pick(row, ["peak_drr_percent", "mean_drr_percent"]),
                    time_to_peak_s=_pick(row, ["time_to_peak_s", "time_to_min_s"]),
                    fwhm_s=_pick(row, ["fwhm_s"], default=float("nan")),
                )
                if f"trial_{t:02d}_t" in cached:
                    raw_curves[t] = (cached[f"trial_{t:02d}_t"], cached[f"trial_{t:02d}_v"])
                if f"trial_{t:02d}_gap_t" in cached:
                    gap_curves[t] = (cached[f"trial_{t:02d}_gap_t"], cached[f"trial_{t:02d}_gap_v"])
    else:
        print("Running leave-one-out ROI + time course extraction (streaming raw frames)...")
        for held_out in usable_trials:
            training_maps = [all_maps[t] for t in usable_trials if t != held_out]
            roi_mask = make_roi_from_maps(training_maps, top_frac=0.05)
            roi_size = int(roi_mask.sum())

            t_arr, v_arr, gap_t_arr, gap_v_arr = roi_mean_timecourse(held_out, roi_mask)
            raw_curves[held_out] = (t_arr.copy(), v_arr.copy())
            gap_curves[held_out] = (gap_t_arr.copy(), gap_v_arr.copy())

            post_mask = (t_arr >= 0) & (t_arr <= 8)
            t_post, v_post = t_arr[post_mask], v_arr[post_mask]
            if len(v_post) == 0:
                continue
            # provisional (see PEAK_WINDOW_S note); overwritten by the frame-indexed pass below
            win = (t_arr >= PEAK_WINDOW_S[0]) & (t_arr <= PEAK_WINDOW_S[1])
            if not win.any():
                continue
            peak_val = float(np.nanmean(v_arr[win]))
            _wt, _wv = t_arr[win], v_arr[win]
            time_to_peak = float(_wt[int(np.argmin(_wv))])

            half = peak_val / 2.0
            below_half = v_post <= half if peak_val < 0 else v_post >= half
            if below_half.any():
                idxs = np.where(below_half)[0]
                fwhm = float(t_post[idxs[-1]] - t_post[idxs[0]])
            else:
                fwhm = float("nan")

            loo_results[held_out] = dict(roi_size_px=roi_size, peak_drr_percent=peak_val,
                                          time_to_peak_s=time_to_peak, fwhm_s=fwhm)
            interp = np.interp(common_grid, t_arr, v_arr, left=np.nan, right=np.nan)
            interp_curves.append(interp)
            print(f"  trial {held_out:2d}: ROI={roi_size}px  peak={peak_val:+.3f}%  "
                  f"t2p={time_to_peak:.1f}s  FWHM={fwhm:.1f}s")

    # === MERGED from finalize_amplitude_and_timecourse.py ===================
    # Amplitude is defined on POST FRAME INDICES -- the identical window used to
    # build post_mean_analysis_window.npy -- so the amplitude and the t-map
    # describe the same epoch in every session, regardless of timestamp latency.
    # The plotting time axis uses the camera interval for frame spacing and
    # Arduino markers for phase offsets.
    _meta = json.loads((session_dir / f"trial_{usable_trials[0]:03d}" / "meta"
                        / "trial_metadata.json").read_text())
    _cfg = _meta["trial_config"]
    n_base, n_post = int(_cfg["baseline_frames"]), int(_cfg["post_frames"])
    f0, f1 = int(_cfg["analysis_start_frame"]), int(_cfg["analysis_end_frame"])
    cam_dt_s = _camera_frame_interval(log_dir, log_time_min)
    base_last_s, post_first_s = _phase_offsets(log_dir, log_time_min)
    t_axis = np.concatenate([
        base_last_s - (n_base - 1 - np.arange(n_base)) * cam_dt_s,
        post_first_s + np.arange(n_post) * cam_dt_s,
    ])
    amp_w0 = float(t_axis[n_base + f0]); amp_w1 = float(t_axis[n_base + f1 - 1])
    print(f"Camera interval {cam_dt_s*1000:.2f} ms ({1/cam_dt_s:.2f} Hz); "
          f"baseline ends {base_last_s:+.3f} s, post begins {post_first_s:+.3f} s")
    print(f"Amplitude window: post frames {f0}-{f1-1} (~{amp_w0:.2f}-{amp_w1:.2f} s)")

    # Older sessions (and any trial where the camera paused across the gap)
    # have no real frames between base_last_s and post_first_s; those still
    # get the flat NaN gap the "no data (camera paused)" box below marks.
    # Sessions where the camera kept triggering through the gap
    # (save_gap_frames) get their real dR/R plotted through that span
    # instead -- but only when the gap frames actually cover the span
    # end-to-end (no void bigger than a couple of frame periods at either
    # edge or in the middle), so a handful of stray gap frames can't produce
    # a misleadingly-smooth interpolation across a mostly-missing gap.
    gap_fill_tol_s = max(2.5 * cam_dt_s, 0.05)
    n_gap_filled = 0

    def _gap_fills_span(gap_t: np.ndarray) -> bool:
        if len(gap_t) == 0:
            return False
        boundary = np.concatenate([[base_last_s], np.sort(gap_t), [post_first_s]])
        diffs = np.diff(boundary)
        return bool(np.all(diffs >= -1e-9) and np.all(diffs <= gap_fill_tol_s))

    interp_curves = []
    _kept = []
    for t in usable_trials:
        if t not in raw_curves:
            continue
        v = np.asarray(raw_curves[t][1], dtype=float)
        if len(v) != n_base + n_post:
            print(f"  WARNING trial {t}: {len(v)} samples, expected "
                  f"{n_base + n_post}; excluded from amplitude.")
            loo_results.pop(t, None)
            continue
        post_v = v[n_base:]
        amp = float(np.mean(post_v[f0:f1]))
        wv = post_v[f0:f1]
        t2p = float(t_axis[n_base + f0:n_base + f1][int(np.argmin(wv))])
        half = amp / 2.0
        bel = post_v <= half if amp < 0 else post_v >= half
        pt = t_axis[n_base:]
        fwhm = float(pt[np.where(bel)[0][-1]] - pt[np.where(bel)[0][0]]) if bel.any() else float("nan")
        loo_results[t] = dict(roi_size_px=int(loo_results.get(t, {}).get("roi_size_px", 0)),
                              peak_drr_percent=amp, time_to_peak_s=t2p, fwhm_s=fwhm)

        gap_t, gap_v = gap_curves.get(t, (np.array([]), np.array([])))
        if _gap_fills_span(gap_t):
            gorder = np.argsort(gap_t)
            t_axis_plot = np.concatenate([t_axis[:n_base], gap_t[gorder], t_axis[n_base:]])
            v_plot = np.concatenate([v[:n_base], gap_v[gorder], v[n_base:]])
            ip = np.interp(common_grid, t_axis_plot, v_plot, left=np.nan, right=np.nan)
            n_gap_filled += 1
        else:
            ip = np.interp(common_grid, t_axis, v, left=np.nan, right=np.nan)
            ip[(common_grid > base_last_s) & (common_grid < post_first_s)] = np.nan
        interp_curves.append(ip)
        _kept.append(t)
    usable_trials = _kept
    print(f"Gap-frame coverage: {n_gap_filled}/{len(usable_trials)} trial(s) had real "
          f"data spanning the full baseline->post gap (tolerance {gap_fill_tol_s*1000:.0f} ms); "
          "the rest fall back to the no-data placeholder for that span.")
    # =======================================================================

    interp_curves = np.array(interp_curves)

    csv_path = out_dir / "loo_roi_amplitude_summary.csv"
    # NOTE: when the amplitudes were just read back out of the cache, nothing has
    # been recomputed, so rewriting this file would only rename the columns to this
    # script's schema and clobber whatever finalize_amplitude_and_timecourse.py
    # wrote. Leave it alone in that case.
    # Amplitudes are recomputed on every run from the frame-indexed window above,
    # so this file is always authoritative and is always rewritten.
    with open(csv_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["trial", "roi_size_px", "peak_drr_percent", "time_to_peak_s", "fwhm_s"])
        for t in usable_trials:
            if t not in loo_results:
                continue
            r = loo_results[t]
            w.writerow([t, r["roi_size_px"], f"{r['peak_drr_percent']:.4f}",
                        f"{r['time_to_peak_s']:.2f}", f"{r['fwhm_s']:.2f}"])

    # Iterate usable_trials, not loo_results.keys(): with --reuse-timecourse-cache,
    # loo_results is seeded from whatever the on-disk CSV happened to contain
    # (e.g. from an earlier run before a trial was excluded via --condition or
    # --exclude-trials) and is never pruned to the current usable set -- only
    # the CSV rewrite above already does this filtering. Aggregating over
    # loo_results directly would silently leak a since-excluded trial's stale
    # numbers back into the summary even though usable_trial_ids says it's gone.
    peaks = np.array([loo_results[t]["peak_drr_percent"] for t in usable_trials if t in loo_results])
    t2ps = np.array([loo_results[t]["time_to_peak_s"] for t in usable_trials if t in loo_results])
    fwhms = np.array([loo_results[t]["fwhm_s"] for t in usable_trials if t in loo_results])
    peak_mean = float(peaks.mean()) if len(peaks) else float("nan")
    peak_sem = float(peaks.std(ddof=1) / np.sqrt(len(peaks))) if len(peaks) > 1 else float("nan")
    print(f"\nLOO peak dR/R%: mean={peak_mean:.3f}  SEM={peak_sem:.3f}")
    print(f"LOO time-to-peak: mean={np.nanmean(t2ps):.2f}s")
    # NOTE: a reused finalize-schema CSV carries no fwhm column, so this can be all-NaN.
    if len(fwhms) and not np.all(np.isnan(fwhms)):
        print(f"LOO FWHM: mean={np.nanmean(fwhms):.2f}s")
    else:
        print("LOO FWHM: n/a (no fwhm column in the reused cache)")

    # mean +/- SEM time course figure (acquisition gap drawn as a break)
    with np.errstate(invalid="ignore"):
        mean_curve = np.nanmean(interp_curves, axis=0) if len(interp_curves) else np.full_like(common_grid, np.nan)
        n_valid = np.sum(~np.isnan(interp_curves), axis=0) if len(interp_curves) else np.zeros_like(common_grid)
        sem_curve = (np.nanstd(interp_curves, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))
                     if len(interp_curves) else np.full_like(common_grid, np.nan))
    mean_curve = np.where(n_valid >= 3, mean_curve, np.nan)
    sem_curve = np.where(n_valid >= 3, sem_curve, np.nan)

    def _missing_runs(is_missing: np.ndarray, lo: float, hi: float):
        """Contiguous (start, end) time spans within [lo, hi] where
        is_missing is True on common_grid."""
        idx = np.where((common_grid >= lo) & (common_grid <= hi))[0]
        runs, run_start, run_end = [], None, None
        for i in idx:
            if is_missing[i]:
                if run_start is None:
                    run_start = common_grid[i]
                run_end = common_grid[i]
            elif run_start is not None:
                runs.append((run_start, run_end))
                run_start = None
        if run_start is not None:
            runs.append((run_start, run_end))
        return runs

    mean_gap_runs = _missing_runs(np.isnan(mean_curve), base_last_s, post_first_s)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.axvspan(amp_w0, amp_w1, color="#FFD400", alpha=0.22,
               label=f"amplitude window (post frames {f0}-{f1-1})")
    for _run_lo, _run_hi in mean_gap_runs:
        ax.axvspan(_run_lo, _run_hi, color="0.85", alpha=0.9, zorder=0)
    ax.fill_between(common_grid, mean_curve - sem_curve, mean_curve + sem_curve,
                    color="0.55", alpha=0.45, label="+/- SEM")
    ax.plot(common_grid, mean_curve, color="k", lw=1.6,
            label=f"mean (LOO ROI, n={len(interp_curves)} trials)")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="crimson", lw=1.1, ls="--", label="stimulus onset")
    ax.set_xlabel("Time relative to STIM_START (s)")
    ax.set_ylabel("ROI dR/R (%)")
    ax.set_xlim(t_axis.min() - 0.4, t_axis.max() + 0.4)
    _fin = np.isfinite(mean_curve) & np.isfinite(sem_curve)
    if _fin.sum() > 10:
        _lo = np.nanpercentile((mean_curve - sem_curve)[_fin], 1.0)
        _hi = np.nanpercentile((mean_curve + sem_curve)[_fin], 99.0)
        _pad = 0.30 * (_hi - _lo) + 1e-3
        ax.set_ylim(_lo - _pad, _hi + _pad)
    _y0, _y1 = ax.get_ylim()
    if mean_gap_runs:
        _label_lo, _label_hi = max(mean_gap_runs, key=lambda r: r[1] - r[0])
        if (_label_hi - _label_lo) >= 0.3:
            ax.text((_label_lo + _label_hi) / 2, _y1 - 0.26 * (_y1 - _y0),
                    "no data\n(camera paused)", ha="center", va="top", fontsize=7,
                    color="0.30", linespacing=1.35,
                    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.75", lw=0.6, alpha=0.9))
    ax.set_title(f"{session_dir.name}: leave-one-out ROI dR/R(t), n={len(interp_curves)} trials\n"
                 f"mean dR/R (post frames {f0}-{f1-1}, ~{amp_w0:.2f}-{amp_w1:.2f} s) = "
                 f"{peak_mean:+.3f}% +/- {peak_sem:.3f}", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "roi_timecourse_mean_sem.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axvspan(amp_w0, amp_w1, color="#FFD400", alpha=0.20)
    for _run_lo, _run_hi in _missing_runs(n_valid == 0, base_last_s, post_first_s):
        ax.axvspan(_run_lo, _run_hi, color="0.85", alpha=0.9, zorder=0)
    ax.axhline(0, color="gray", lw=0.8)
    cmap = plt.get_cmap("viridis")
    denom = max(len(usable_trials) - 1, 1)
    for i, t in enumerate(usable_trials):
        if i >= len(interp_curves):
            break
        ax.plot(common_grid, interp_curves[i], color=cmap(i / denom), lw=1.1, label=f"trial {t}")
    ax.set_xlabel("Time relative to STIM_START (s)")
    ax.set_ylabel("ROI dR/R (%)")
    ax.set_title(f"{session_dir.name}: per-trial LOO ROI dR/R(t) (dark->light = trial order)")
    ax.legend(ncol=3, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roi_timecourse_per_trial.png", dpi=200)
    plt.close(fig)

    raw_save = {"common_grid": common_grid, "interp_curves": interp_curves,
                "trial_ids": np.array(usable_trials)}
    for t, (ta, va) in raw_curves.items():
        raw_save[f"trial_{t:02d}_t"] = ta
        raw_save[f"trial_{t:02d}_v"] = va
    for t, (gta, gva) in gap_curves.items():
        raw_save[f"trial_{t:02d}_gap_t"] = gta
        raw_save[f"trial_{t:02d}_gap_v"] = gva
    np.savez(out_dir / "roi_timecourse_raw.npz", **raw_save)

    # -----------------------------------------------------------------
    # pixelwise t-map + sign-flip permutation cluster test
    # -----------------------------------------------------------------
    print("\nBuilding pixelwise t-map across trials...")
    by0, by1, bx0, bx1 = local_binned_slice_for_region()
    print(f"Restricting t-map to shared region, local binned slice: "
          f"rows[{by0}:{by1}] cols[{bx0}:{bx1}] (of {map_shape})")
    stack = np.stack([all_maps[t][by0:by1, bx0:bx1] for t in usable_trials], axis=0)
    # NOTE: remove per-trial field-wide DC offset (thermal drift / vasomotion)
    stack = stack - np.nanmean(stack, axis=(1, 2), keepdims=True)
    n = stack.shape[0]
    mean_map = stack.mean(axis=0)
    sd_map = stack.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_map = mean_map / (sd_map / np.sqrt(n))
    t_map = np.nan_to_num(t_map, nan=0.0, posinf=0.0, neginf=0.0)

    def _sig_mask(tmap, thresh):
        return (tmap < -thresh) if neg_only else (np.abs(tmap) > thresh)

    def max_cluster_size(tmap, thresh):
        mask = _sig_mask(tmap, thresh)
        if not mask.any():
            return 0
        lbl, ncomp = label(mask)
        if ncomp == 0:
            return 0
        sizes = np.bincount(lbl.ravel())[1:]
        return int(sizes.max())

    observed_max_cluster = max_cluster_size(t_map, CLUSTER_T_THRESH)

    _dm = np.abs(t_map) > CLUSTER_T_THRESH
    _dl, _dn = label(_dm)
    _rows = []
    for _i in range(1, _dn + 1):
        _sel = _dl == _i
        _sz = int(_sel.sum())
        _mt = float(t_map[_sel].mean())
        _ys, _xs = np.nonzero(_sel)
        _rows.append((_sz, "NEG" if _mt < 0 else "POS", _mt,
                      int(_xs.mean()), int(_ys.mean())))
    _rows.sort(reverse=True)
    print(f"  cluster breakdown (|t|>{CLUSTER_T_THRESH}), top 6 of {_dn}:")
    for _sz, _sg, _mt, _cx, _cy in _rows[:6]:
        print(f"    {_sg}  {_sz:>7d}px  mean_t={_mt:+.2f}  centroid=(x={_cx},y={_cy})")
    _negtot = sum(r[0] for r in _rows if r[1] == "NEG")
    _postot = sum(r[0] for r in _rows if r[1] == "POS")
    print(f"    total NEG px={_negtot}  total POS px={_postot}")
    print(f"  mode: {'NEGATIVE-ONLY' if neg_only else 'two-sided |t|'}")

    print(f"Running {n_perm_actual} sign-flip permutations (n={n} trials)...")
    rng = np.random.default_rng(
        int(hashlib.sha256(session_dir.name.encode()).hexdigest()[:8], 16))
    null_max_clusters = np.empty(n_perm_actual, dtype=int)
    for i in range(n_perm_actual):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_stack = stack * signs[:, None, None]
        pm = perm_stack.mean(axis=0)
        psd = perm_stack.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pt = pm / (psd / np.sqrt(n))
        pt = np.nan_to_num(pt, nan=0.0, posinf=0.0, neginf=0.0)
        null_max_clusters[i] = max_cluster_size(pt, CLUSTER_T_THRESH)

    cluster_p = float((np.sum(null_max_clusters >= observed_max_cluster) + 1) / (n_perm_actual + 1))
    peak_abs_t = float(np.max(np.abs(t_map)))
    print(f"Observed max cluster (|t|>{CLUSTER_T_THRESH}): {observed_max_cluster}px, "
          f"null mean={null_max_clusters.mean():.1f}, p={cluster_p:.4f}, peak|t|={peak_abs_t:.2f} (df={n-1})")

    mask_sig = _sig_mask(t_map, CLUSTER_T_THRESH)
    lbl, ncomp = label(mask_sig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    im0 = axes[0].imshow(t_map, cmap="seismic", vmin=-8, vmax=8)
    axes[0].set_title("Pixelwise t-map (mean/[SD/sqrt(n)])")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label=f"t (df={n-1})")

    axes[1].imshow(mean_map, cmap="gray")
    overlay = np.ma.masked_where(~mask_sig, lbl)
    axes[1].imshow(overlay, cmap="autumn", alpha=0.6)
    axes[1].set_title(("Clusters t<-" if neg_only else "Clusters |t|>") + f"{CLUSTER_T_THRESH} "
                      f"(largest={observed_max_cluster}px, p={cluster_p:.3f})")
    axes[1].axis("off")

    axes[2].hist(null_max_clusters, bins=40, color="gray", alpha=0.8)
    axes[2].axvline(observed_max_cluster, color="red", lw=2, label="observed")
    axes[2].set_xlabel("max cluster size (permutation null)")
    axes[2].set_ylabel("count")
    axes[2].set_title(f"Sign-flip permutation null (n={n_perm_actual})")
    axes[2].legend()
    fig.suptitle(session_dir.name)
    fig.tight_layout()
    fig.savefig(out_dir / "tmap_and_permutation_cluster_test.png", dpi=200)
    plt.close(fig)

    summary = dict(
        session=session_dir.name,
        condition=condition,
        n_trial_folders_on_disk=n_trials_on_disk,
        n_trials_usable=len(usable_trials),
        usable_trial_ids=usable_trials,
        roi_mode="full_frame" if full_frame else "leave_one_out_top5pct",
        roi_definition=(
            f"full imaged window, region x={region_x0} y={region_y0} w={region_w} h={region_h} (full-res px)"
            if full_frame else
            "leave-one-out, top 5% most-negative pixels of training-trial mean dR/R map"
        ),
        shared_crop_fullres=list(shared_crop) if shared_crop is not None else None,
        roi_within_region=bool(roi_within_region),
        manually_excluded_trial_ids=excluded_present,
        analysis_region_fullres=[region_x0, region_y0, region_w, region_h],
        no_crop=bool(no_crop),
        array_source_dir=str(array_source_dir) if array_source_dir is not None else None,
        loo_peak_drr_percent_mean=peak_mean,
        loo_peak_drr_percent_sem=peak_sem,
        loo_time_to_peak_s_mean=float(np.nanmean(t2ps)) if len(t2ps) else None,
        # NOTE: write null rather than a bare NaN (invalid JSON) when fwhm is absent.
        loo_fwhm_s_mean=(float(np.nanmean(fwhms))
                         if len(fwhms) and not np.all(np.isnan(fwhms)) else None),
        tmap_cluster_forming_threshold_t=CLUSTER_T_THRESH,
        tmap_observed_max_cluster_px=observed_max_cluster,
        tmap_permutation_n=n_perm_actual,
        tmap_cluster_corrected_p=cluster_p,
        tmap_peak_abs_t=peak_abs_t,
        tmap_df=n - 1,
        amplitude_window_post_frames=[f0, f1 - 1],
        amplitude_window_s_approx=[round(amp_w0, 3), round(amp_w1, 3)],
        amplitude_estimator=("mean dR/R over post frames "
                             f"{f0}-{f1-1} (identical to the t-map post window)"),
        camera_frame_interval_s=round(cam_dt_s, 5),
        time_axis=("camera interval for frame spacing; phase offsets from "
                   "Arduino markers relative to STIM_START"),
        tmap_sign_mode=("negative_only" if neg_only else "two_sided"),
        gap_fill_tolerance_s=round(gap_fill_tol_s, 4),
        n_trials_gap_filled=n_gap_filled,
        n_trials_gap_placeholder=len(usable_trials) - n_gap_filled,
    )
    # NOTE: with --reuse-timecourse-cache the amplitude/time-course stage was not
    # recomputed, so this run has nothing new to say about those fields. They are
    # owned by finalize_amplitude_and_timecourse.py, which stores them at full
    # precision -- whereas the cached CSV is rounded to 4 dp and has no fwhm column
    # at all. Overwriting would silently degrade the mean/SEM and null out the FWHM.
    # Carry the existing values forward; only the tmap_* fields belong to this run.
    # This script now recomputes the amplitude/time-course fields itself on every
    # run (frame-indexed window above), so nothing needs carrying forward from a
    # previous finalize pass -- doing so would reinstate stale values.
    _FINALIZE_OWNED = ()
    summary_path = out_dir / "analysis_summary.json"
    if reused_cache and summary_path.exists():
        try:
            with open(summary_path) as fp:
                prev = json.load(fp)
        except (ValueError, OSError):
            prev = {}
        carried = []
        for k in _FINALIZE_OWNED:
            if k in prev and prev[k] is not None and summary.get(k) != prev[k]:
                summary[k] = prev[k]
                carried.append(k)
        for k in prev:
            if k not in summary:
                summary[k] = prev[k]
        if carried:
            print(f"Preserved {len(carried)} finalize-owned field(s) from the existing "
                  f"summary: {', '.join(carried)}")

    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    print("Done ->", out_dir)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--full-frame", action="store_true",
                     help="Use the whole imaged window as the ROI instead of the "
                          "data-driven top-5%%-negative-pixel LOO mask.")
    ap.add_argument("--shared-crop", type=str, default=None,
                     help="x,y,w,h in full-resolution pixels; restricts both the "
                          "ROI and the pixelwise t-map to this rectangle (e.g. the "
                          "intersection of two same-day sessions' own crops).")
    ap.add_argument("--out-suffix", type=str, default="_reanalysis")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--neg-only", action="store_true",
                     help="Only negative-going clusters (t < -thresh) count, for observed AND null.")
    ap.add_argument("--condition", type=str, default="all", choices=["all", "stim", "catch"],
                     help="Filter usable trials by their recorded trial_condition "
                          "(interleaved sessions only, from --trial-conditions at "
                          "acquisition time). 'all' (default) matches today's behavior.")
    ap.add_argument("--log-dir", type=str, default=None,
                     help="Folder containing session_log.csv/marker_log.csv, if not session_dir itself.")
    ap.add_argument("--log-time-min", type=str, default=None, help="ISO timestamp, inclusive lower bound.")
    ap.add_argument("--log-time-max", type=str, default=None, help="ISO timestamp, exclusive upper bound.")
    ap.add_argument("--no-crop", action="store_true",
                     help="Ignore any manual active_roi crop; use the full 1920x1200 sensor frame. "
                          "Requires --array-source-dir pointing at pre-computed uncropped arrays.")
    ap.add_argument("--array-source-dir", type=str, default=None,
                     help="Directory with trial_XXX/{baseline_reference,post_mean_analysis_window}.npy "
                          "to use instead of session_dir/trial_XXX/analysis/.")
    ap.add_argument("--n-perm", type=int, default=None, help="Override number of sign-flip permutations.")
    ap.add_argument("--reuse-timecourse-cache", action="store_true",
                     help="If roi_timecourse_raw.npz/loo_roi_amplitude_summary.csv already exist in the "
                          "output folder, reuse them instead of re-streaming raw frames (only rebuilds "
                          "the t-map/permutation section). Useful for re-running with a different "
                          "--n-perm without repeating the slow part.")
    ap.add_argument("--roi-within-region", action="store_true",
                     help="Restrict the top-5%% ROI candidate pixels to the analysis region "
                          "(i.e. make --shared-crop apply to the ROI as well as the t-map). "
                          "No-op when --shared-crop is absent.")
    ap.add_argument("--exclude-trials", type=str, default=None,
                     help="Comma-separated trial numbers to drop before any ROI/dR/R/t-map "
                          "computation (e.g. '3,7,12'), e.g. for bubble-contaminated trials. "
                          "Applied after --condition filtering; trial numbers not in the "
                          "usable set are ignored with a warning.")
    ap.add_argument("--save-loo-masks", action="store_true",
                     help="Save every fold's LOO ROI mask, a fold-selection-frequency heatmap, and an "
                          "example-folds figure into out_dir/loo_masks/. Fast (no raw-frame streaming).")
    args = ap.parse_args()
    shared_crop = tuple(int(v) for v in args.shared_crop.split(",")) if args.shared_crop else None
    exclude_trials = (
        {int(tok) for tok in args.exclude_trials.split(",") if tok.strip()}
        if args.exclude_trials else None
    )
    main(Path(args.session_dir), full_frame=args.full_frame, shared_crop=shared_crop,
         out_suffix=args.out_suffix,
         log_dir=Path(args.log_dir) if args.log_dir else None,
         log_time_min=args.log_time_min, log_time_max=args.log_time_max,
         no_crop=args.no_crop,
         array_source_dir=Path(args.array_source_dir) if args.array_source_dir else None,
         n_perm=args.n_perm,
         reuse_timecourse_cache=args.reuse_timecourse_cache,
         exclude_trials=exclude_trials,
         save_loo_masks=args.save_loo_masks,
         out_dir_override=args.out_dir, neg_only=args.neg_only,
         condition=args.condition, roi_within_region=args.roi_within_region)
