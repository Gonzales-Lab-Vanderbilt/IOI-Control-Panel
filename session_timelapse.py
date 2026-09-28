#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
session_timelapse.py

The intrinsic signal developing over time: a trial-averaged dR/R movie of one
session, aligned to stimulus onset, plus a filmstrip PNG of the same thing for
a poster.

Consumes a session's own statistical_analyses.py output (analysis_summary.json
for the usable trials and the analysis region) and the raw frames on disk.
Writes, into --out-dir (default: the stats folder itself):

    <label>_timelapse.mp4            movie (or .gif if ffmpeg is unavailable)
    <label>_timelapse_filmstrip.png  frames at chosen times, shared scale, with the trace
    <label>_timelapse.json           exactly what was done: trials, timing check,
                                     calibration, scale, processing parameters
    <label>_timelapse_cache_<cond>.npz   the averaged stack, so re-rendering with
                                     a different scale/orientation skips the raw reads

How each movie frame is made
  1. Frame times come from the CAMERA clock (camera_timestamp) for spacing,
     anchored to the ARDUINO clock for the stimulus: t = 0 is STIM_START, and
     the first post frame sits at POST_FIRST_FRAME_TRIGGER - STIM_START. Host
     wall-clock timestamps are never used for timing. Frames are placed on a
     grid of the measured camera interval (~100 ms).
  2. Per trial: dR/R = (F - B) / B per pixel, B = that trial's own baseline mean.
  3. Averaged across the usable trials at each time point, then smoothed in
     space (Gaussian, matched to the stats pipeline's sigma), demeaned over the
     valid field (removes field-wide offsets, as the t-map does), and lightly
     smoothed in time (centred moving average that never bridges missing frames).
  4. Displayed on ONE colour scale for every frame. Per-frame autoscaling would
     make noise look like a growing response.

Pixels too dark to measure (the aperture annulus, out-of-field corners) are
left transparent rather than shown as noise.

The trace under the movie is the same full-session ROI and 180-degree
out-region session_poster_figures.py uses, computed from full-resolution raw
frames exactly as the pipeline does (ROI mean relative to its baseline mean).

Honest gaps: frames the acquisition captured but did not save (the 0.5 s of
baseline cushion before the gap frames; everything after the last post frame;
the whole onset gap in sessions from before gap frames were saved) are shown
as "not saved", never interpolated.

Usage:
    py -3.10 session_timelapse.py <session_dir> [--stats-dir DIR]
        [--compare-stats-dir DIR --compare-label "Catch"] [--um-per-px 3.682]
        [--anterior-side left --medial-side bottom | --midline-centered]
        [--fixed-vmax 0.2] [--bin 4] [--fps 5] [--filmstrip-times -1,0.5,1.5,2.5,3.5,4.5]
        [--reuse-cache] [--compute-only]

If <session_dir>/landmarks.json exists (written by the GUI's Landmarks tab), its
orientation and calibration are used unless overridden on the command line,
and bregma/lambda are marked on the frames.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import textwrap
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle

import ioi_landmarks
import session_poster_figures as spf

W, H = spf.W, spf.H
C_STIM, C_COMPARE, C_OUT = spf.C_STIM, spf.C_COMPARE, spf.C_OUT
SAVED_PHASES = {"baseline_collect": "baseline", "gap": "gap", "post_collect": "post"}
PIPELINE_BINNING = 2          # statistical_analyses.py maps are 2x2 binned ...
PIPELINE_SIGMA = spf.SMOOTH_SIGMA  # ... and smoothed with sigma 5 at that resolution
DARK_FRACTION = 0.25          # pixels with baseline < 25 % of the field's bright level are not measurable
AMP_WINDOW_S = (1.5, 4.4)     # the pipeline's amplitude window (post frames 5-34), for the auto scale
DEFAULT_FILMSTRIP = (-1.0, 0.5, 1.5, 2.5, 3.5, 4.5)


# ───────────────────────────── logs and timing ─────────────────────────────

def _millis(row: dict) -> int | None:
    v = (row.get("arduino_millis") or "").strip()
    if not v:
        parts = (row.get("raw_text") or "").split(",")
        v = parts[1].strip() if len(parts) > 1 else ""
    try:
        return int(v)
    except ValueError:
        return None


def frame_table(session_dir: Path, trial_ids: list[int]) -> tuple[dict, dict]:
    """Per trial: saved frames with their time relative to stimulus onset (s),
    and a timing report. See the module docstring, step 1."""
    want = set(trial_ids)
    rows_by_trial: dict[int, dict[str, dict]] = defaultdict(dict)
    with open(session_dir / "session_log.csv", newline="", encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            try:
                t = int(r["trial_index"])
            except (ValueError, KeyError):
                continue
            if t not in want or r.get("phase") not in SAVED_PHASES:
                continue
            fname = (r.get("filename") or "").strip()
            cam = (r.get("camera_timestamp") or "").strip()
            if not fname or not cam:
                continue
            prev = rows_by_trial[t].get(fname)
            # A re-run into the same trial folder overwrote the files: keep the later block.
            if prev is None or r["host_timestamp_iso"] > prev["host_timestamp_iso"]:
                rows_by_trial[t][fname] = r

    markers: dict[int, list[tuple[str, str, int]]] = defaultdict(list)
    with open(session_dir / "marker_log.csv", newline="", encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            try:
                t = int(r["trial_index"])
            except (ValueError, KeyError):
                continue
            ms = _millis(r)
            if t in want and ms is not None:
                markers[t].append((r["host_timestamp_iso"], r["marker"], ms))

    table, post1_offsets, stim_durs, dts, skipped = {}, [], [], [], {}
    for t in trial_ids:
        rows = rows_by_trial.get(t, {})
        post1 = rows.get("post_00001.raw")
        if post1 is None:
            skipped[t] = "no post_00001 frame in the log"
            continue
        mk = markers.get(t, [])
        pf = [m for m in mk if m[1] == "POST_FIRST_FRAME_TRIGGER"]
        if not pf:
            skipped[t] = "no POST_FIRST_FRAME_TRIGGER marker"
            continue
        # the marker block that belongs to the saved frames: closest in host time to post_00001
        pf_host, _, pf_ms = min(pf, key=lambda m: abs(_iso_s(m[0]) - _iso_s(post1["host_timestamp_iso"])))
        ss = [m for m in mk if m[1] == "STIM_START" and 0 <= pf_ms - m[2] <= 5000]
        if not ss:
            skipped[t] = "no STIM_START within 5 s before the first post frame"
            continue
        stim_ms = max(ss, key=lambda m: m[2])[2]
        se = [m[2] for m in mk if m[1] == "STIM_END" and 0 < m[2] - stim_ms <= 60000]
        if se:
            stim_durs.append((min(se) - stim_ms) / 1000.0)
        anchor = (pf_ms - stim_ms) / 1000.0
        cam0 = int(post1["camera_timestamp"])
        frames = []
        for fname, r in rows.items():
            sub = SAVED_PHASES[r["phase"]]
            if not fname.startswith(sub + "_"):
                continue
            frames.append((sub, fname, (int(r["camera_timestamp"]) - cam0) / 1e9 + anchor))
        frames.sort(key=lambda f: f[2])
        base = [f[2] for f in frames if f[0] == "baseline"]
        if len(base) > 2:
            dts.append(float(np.median(np.diff(base))))
        post1_offsets.append(anchor)
        table[t] = frames

    report = {
        "trials_timed": len(table),
        "trials_skipped": {str(k): v for k, v in skipped.items()},
        "first_post_frame_s": _stats(post1_offsets),
        "stimulus_duration_s": _stats(stim_durs),
        "camera_interval_s": _stats(dts),
    }
    return table, report


def _iso_s(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def _stats(v: list) -> dict | None:
    if not v:
        return None
    a = np.asarray(v, dtype=float)
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


# ───────────────────────────── averaging ───────────────────────────────────

def load_raw(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint16).reshape(H, W)


def bin_region(img: np.ndarray, region: tuple, b: int) -> np.ndarray:
    x0, y0, w, h = region
    h2, w2 = (h // b) * b, (w // b) * b
    sub = img[y0:y0 + h2, x0:x0 + w2].astype(np.float32)
    return sub.reshape(h2 // b, b, w2 // b, b).mean(axis=(1, 3))


def average_condition(session_dir: Path, trial_ids: list[int], region: tuple, b: int,
                      dt: float, roi: np.ndarray, out: np.ndarray, tag: str) -> dict:
    """Trial-averaged per-pixel dR/R on the time grid, plus per-trial ROI and
    out-region traces computed exactly like the pipeline (full-resolution
    mask means relative to the baseline mask mean)."""
    table, report = frame_table(session_dir, trial_ids)
    if not table:
        raise SystemExit(f"No timeable trials for {tag}: {report['trials_skipped']}")
    ks = sorted({int(round(f[2] / dt)) for fr in table.values() for f in fr})
    k0, k1 = ks[0], ks[-1]
    nk = k1 - k0 + 1
    hb, wb = region[3] // b, region[2] // b
    acc = np.zeros((nk, hb, wb), dtype=np.float64)
    cnt = np.zeros(nk, dtype=np.int32)
    base_acc = np.zeros((hb, wb), dtype=np.float64)
    roi_tr = np.full((len(table), nk), np.nan)
    out_tr = np.full((len(table), nk), np.nan)
    used = []
    print(f"Timelapse: streaming {len(table)} {tag} trial(s), {b}x{b} binning, grid {dt*1000:.1f} ms")
    t_start = time.time()
    for i, (trial, frames) in enumerate(sorted(table.items())):
        tdir = session_dir / f"trial_{trial:03d}"
        base_frames = [f for f in frames if f[0] == "baseline"]
        if len(base_frames) < 5:
            print(f"  trial {trial}: only {len(base_frames)} baseline frames, skipped")
            continue
        # baseline frames are held (needed twice); gap/post frames stream one at a time
        present = [(s, f, tt) for s, f, tt in frames if (tdir / s / f).exists()]
        bl = {(s, f): load_raw(tdir / s / f) for s, f, _ in base_frames if (tdir / s / f).exists()}
        B_full = np.mean(list(bl.values()), axis=0, dtype=np.float64)
        B = bin_region(B_full, region, b).astype(np.float64)
        rb, ob = float(B_full[roi].mean()), float(B_full[out].mean())
        safe = np.where(B > 0, B, np.nan)
        seen = set()
        for sub, fname, tt in present:
            img = bl[(sub, fname)] if (sub, fname) in bl else load_raw(tdir / sub / fname)
            k = int(round(tt / dt)) - k0
            if k in seen:          # two frames on one grid slot: keep the first, report it
                print(f"  trial {trial}: duplicate grid slot at {tt:+.3f} s ignored")
                continue
            seen.add(k)
            acc[k] += (bin_region(img, region, b) - B) / safe
            cnt[k] += 1
            roi_tr[i, k] = 100.0 * (img[roi].mean() - rb) / rb
            out_tr[i, k] = 100.0 * (img[out].mean() - ob) / ob
        base_acc += B
        used.append(trial)
        del bl
        print(f"  Timelapse trial {i + 1}/{len(table)} (trial_{trial:03d}, {len(present)} frames, "
              f"{time.time() - t_start:.0f}s)")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = acc / cnt[:, None, None]
    mean[cnt == 0] = np.nan
    keep = [i for i, (trial, _) in enumerate(sorted(table.items())) if trial in used]
    return dict(times=(np.arange(nk) + k0) * dt, mean=mean.astype(np.float32), counts=cnt,
                baseline=(base_acc / max(len(used), 1)).astype(np.float32),
                roi_traces=roi_tr[keep], out_traces=out_tr[keep], trials=np.array(used),
                report=report)


def valid_mask(baseline: np.ndarray) -> np.ndarray:
    bright = np.nanpercentile(baseline, 99)
    return baseline > DARK_FRACTION * bright


def display_stack(res: dict, valid: np.ndarray, b: int, demean: bool, smooth_frames: int) -> np.ndarray:
    """Spatial smoothing (normalised, so edges don't bleed from masked pixels),
    optional per-frame demeaning over the valid field, temporal smoothing that
    never bridges a missing frame. Returns percent dR/R, NaN where invalid."""
    sigma = PIPELINE_SIGMA * PIPELINE_BINNING / b
    m = valid.astype(np.float64)
    norm = gaussian_filter(m, sigma)
    out = np.full(res["mean"].shape, np.nan, dtype=np.float64)
    for k in range(res["mean"].shape[0]):
        f = res["mean"][k].astype(np.float64)
        if not np.isfinite(f).any():
            continue
        f = np.where(valid & np.isfinite(f), f, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = gaussian_filter(f, sigma) / norm
        s[~valid] = np.nan
        if demean:
            # median, not mean: a strong focal response would drag the mean and tint the
            # whole field the opposite colour; the median ignores it while it covers < 50 %
            s = s - np.nanmedian(s)
        out[k] = 100.0 * s
    if smooth_frames > 1:
        half = smooth_frames // 2
        sm = np.full_like(out, np.nan)
        present = np.array([np.isfinite(out[k]).any() for k in range(out.shape[0])])
        for k in range(out.shape[0]):
            if not present[k]:
                continue
            lo, hi = k - half, k + half
            if lo >= 0 and hi < out.shape[0] and present[lo:hi + 1].all():
                sm[k] = np.nanmean(out[lo:hi + 1], axis=0)
            else:
                sm[k] = out[k]      # at an edge or next to a gap: no smoothing rather than bridging
        out = sm
    return out


# ───────────────────────────── drawing helpers ─────────────────────────────

def green_display(session_dir: Path, region: tuple, b: int) -> np.ndarray:
    g = spf.green_reference_mean(session_dir)
    gb = bin_region(g, region, b).astype(np.float64)
    lo, hi = np.percentile(gb, [0.2, 99.8])
    return np.clip((gb - lo) / (hi - lo + 1e-9), 0, 1) ** 0.85


def overlay_rgba(img: np.ndarray, vmax: float) -> np.ndarray:
    norm = Normalize(-vmax, vmax)
    rgba = plt.get_cmap("RdBu")(norm(np.nan_to_num(img)))
    alpha = np.clip(np.abs(np.nan_to_num(img)) / vmax, 0, 1) ** 0.7
    alpha[~np.isfinite(img)] = 0.0
    alpha[alpha < 0.18] = 0.0
    rgba[..., 3] = np.clip(alpha, 0, 0.88)
    return rgba


def draw_scalebar(ax, shape, um_per_binned_px: float, fs: float) -> None:
    hh, ww = shape
    bar = 1000.0 / um_per_binned_px
    m, bh, pad = 0.03 * ww, 0.012 * ww, 0.015 * ww
    ax.add_patch(FancyBboxPatch((ww - m - bar - 2 * pad, hh - m - bh - 2 * pad - 0.09 * hh), bar + 2 * pad,
                                bh + 2 * pad + 0.09 * hh, boxstyle=f"round,pad=0,rounding_size={0.012 * ww}",
                                facecolor="black", alpha=0.5, edgecolor="none", zorder=8))
    ax.add_patch(Rectangle((ww - m - pad - bar, hh - m - pad - bh), bar, bh,
                           facecolor="white", edgecolor="black", lw=0.8, zorder=9))
    ax.text(ww - m - pad - bar / 2, hh - m - pad - bh - 0.012 * hh, "1 mm", ha="center", va="bottom",
            fontsize=fs, fontweight="bold", color="white", zorder=9)


def draw_compass(ax, shape, orient: dict, fs: float) -> None:
    hh, ww = shape
    arm, ext = 0.035 * ww, 0.075 * ww
    cx, cy = 0.03 * ww + ext, hh - 0.03 * ww - ext
    ax.add_patch(FancyBboxPatch((cx - ext, cy - ext), 2 * ext, 2 * ext,
                                boxstyle=f"round,pad=0,rounding_size={0.014 * ww}",
                                facecolor="black", alpha=0.5, edgecolor="none", zorder=8))
    ax.plot([cx - arm, cx + arm], [cy, cy], color="white", lw=1.6, zorder=9)
    ax.plot([cx, cx], [cy - arm, cy + arm], color="white", lw=1.6, zorder=9)
    d = (arm + ext) / 2
    for lx, ly, side in ((cx - d, cy, "left"), (cx + d, cy, "right"), (cx, cy - d, "top"), (cx, cy + d, "bottom")):
        ax.text(lx, ly, orient[side], ha="center", va="center", fontsize=fs, fontweight="bold", color="white", zorder=9)
    if orient.get("center"):
        ax.text(cx, cy, orient["center"], ha="center", va="center", fontsize=fs, fontweight="bold",
                color="white", zorder=10, path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])


def draw_landmarks(ax, lm: dict | None, region: tuple, b: int, fs: float, um: float | None = None) -> None:
    """Bregma / lambda on a cropped, b-binned panel. One outside the crop
    (off-image, or just outside the analysis region) becomes an arrow at the
    panel edge with its distance, instead of being clipped away."""
    if not lm:
        return
    x0, y0, w, h = region[:4]
    box = (x0, y0, x0 + w - 1, y0 + h - 1)
    for key, tag in (("bregma", "B"), ("lambda", "L")):
        p = (lm.get("points") or {}).get(key)
        if not p:
            continue
        x, y = (p["x"] - x0) / b, (p["y"] - y0) / b
        est = p.get("status") != "visible"
        if spf.draw_offframe_pointer(ax, p, tag + ("?" if est else ""), box,
                                     lambda px, py: ((px - x0) / b, (py - y0) / b), um,
                                     fontsize=fs * 0.9, arrow_len=0.16 * min(w, h) / b, lw=1.8):
            continue
        ax.plot(x, y, marker="+", ms=14, mew=2.2, color="white", zorder=11,
                path_effects=[pe.withStroke(linewidth=4, foreground="black")])
        ax.text(x + 6, y - 6, tag + ("?" if est else ""), color="white", fontsize=fs, fontweight="bold",
                zorder=11, path_effects=[pe.withStroke(linewidth=3, foreground="black")])


def missing_spans(times: np.ndarray, present: np.ndarray, dt: float) -> list[tuple[float, float]]:
    spans, start = [], None
    for t, p in zip(times, present):
        if not p and start is None:
            start = t
        if p and start is not None:
            spans.append((start - dt / 2, t - dt / 2))
            start = None
    if start is not None:
        spans.append((start - dt / 2, times[-1] + dt / 2))
    return spans


def plot_traces(ax, series: list, stim_dur: float | None, spans: list, xlim: tuple, fs: float):
    ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    if stim_dur:
        ax.axvspan(0, stim_dur, color="#F2D7D5", alpha=0.6, lw=0, zorder=0)
        ax.text(0.02, 0.96, "stimulus on", transform=ax.get_xaxis_transform(), color="#922B21",
                fontsize=fs * 0.85, va="top", ha="left")
    for a, b_ in spans:
        ax.axvspan(max(a, xlim[0]), min(b_, xlim[1]), facecolor="none", edgecolor="0.55",
                   hatch="///", lw=0, zorder=0)
    for s in series:
        t, tr, color, label, dashed = s
        mu = np.nanmean(tr, axis=0)
        n = np.sum(np.isfinite(tr), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sem = np.nanstd(tr, axis=0, ddof=1) / np.sqrt(n)
        ok = n >= max(2, tr.shape[0] // 2)
        mu_p = np.where(ok, mu, np.nan)
        ax.plot(t, mu_p, color=color, lw=2.0, ls="--" if dashed else "-", label=label, zorder=3)
        if not dashed:
            ax.fill_between(t, mu_p - sem, mu_p + sem, color=color, alpha=0.2, lw=0, zorder=2)
    ax.set_xlim(*xlim)
    ax.set_xlabel("time from stimulus onset (s)", fontsize=fs)
    ax.set_ylabel("ΔR/R (%)", fontsize=fs)
    ax.tick_params(labelsize=fs * 0.85)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=fs * 0.8, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              ncol=len(series), borderaxespad=0.2)


# ───────────────────────────── main ────────────────────────────────────────

def resolve_orientation(args, lm: dict | None) -> tuple[dict, str]:
    if args.anterior_side:
        if args.midline_centered:
            return spf.midline_centered_orientation_map(args.anterior_side), "command line"
        return spf.orientation_map(args.anterior_side, args.medial_side or "bottom"), "command line"
    cs = ioi_landmarks.compass_settings(lm) if lm else None
    if cs:
        if cs["midline_centered"]:
            return spf.midline_centered_orientation_map(cs["anterior_side"]), "landmarks.json"
        if cs["medial_side"]:
            return spf.orientation_map(cs["anterior_side"], cs["medial_side"]), "landmarks.json"
    return spf.orientation_map("left", "bottom"), "default (anterior left, medial bottom), not session metadata"


def resolve_calibration(args, lm: dict | None) -> tuple[float | None, str]:
    lm_um = ((lm or {}).get("calibration") or {}).get("um_per_px")
    if args.no_scale_bar:
        return None, "scale bar off"
    if lm_um:
        if args.um_per_px and abs(args.um_per_px - lm_um) > 1e-6:
            print(f"NOTE: using this session's own landmarks.json calibration {lm_um} um/px, "
                  f"not --um-per-px {args.um_per_px} (calibration travels with the session).")
        return float(lm_um), "landmarks.json"
    if args.um_per_px:
        return float(args.um_per_px), "command line"
    return None, "not set: no scale bar"


def default_stats_dir(session_dir: Path) -> Path:
    for name in ("session_stats", "session_stats_stim"):
        if (session_dir / name / "analysis_summary.json").exists():
            return session_dir / name
    raise SystemExit("No analysis_summary.json found; run statistical_analyses.py first or pass --stats-dir.")


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir")
    ap.add_argument("--stats-dir", help="Folder with this session's analysis_summary.json (default: session_stats[_stim]).")
    ap.add_argument("--compare-stats-dir",
                    help="A second stats folder whose trials are shown side by side, measured in THIS "
                         "session's ROI -- e.g. an interleaved session's session_stats_catch.")
    ap.add_argument("--compare-session", help="Session folder of --compare-stats-dir (default: same session).")
    ap.add_argument("--label", help="Output file prefix (default: session folder name).")
    ap.add_argument("--stim-label", default="Stimulus trials")
    ap.add_argument("--compare-label", default="Catch trials")
    ap.add_argument("--out-dir", help="Default: the stats folder.")
    ap.add_argument("--bin", type=int, default=4, help="Spatial binning for the movie (default 4 -> 480x300 px).")
    ap.add_argument("--smooth-frames", type=int, default=3, help="Centred moving average, in frames (default 3 = 0.3 s; 1 = off).")
    ap.add_argument("--no-demean", action="store_true", help="Keep field-wide offsets (default subtracts the per-frame field mean).")
    ap.add_argument("--fixed-vmax", type=float, help="Colour scale limit in %% dR/R (default: auto from the amplitude window).")
    ap.add_argument("--um-per-px", type=float, help="Full-resolution calibration for the scale bar.")
    ap.add_argument("--no-scale-bar", action="store_true")
    ap.add_argument("--anterior-side", choices=spf._SIDES)
    ap.add_argument("--medial-side", choices=spf._SIDES)
    ap.add_argument("--midline-centered", action="store_true")
    ap.add_argument("--fps", type=float, default=5.0, help="Movie frame rate (default 5 = half real time).")
    ap.add_argument("--format", choices=["auto", "mp4", "gif"], default="auto")
    ap.add_argument("--filmstrip-times", default=",".join(str(t) for t in DEFAULT_FILMSTRIP),
                    help="Comma-separated times (s) for the filmstrip panels.")
    ap.add_argument("--reuse-cache", action="store_true", help="Reuse the averaged stack if already cached.")
    ap.add_argument("--compute-only", action="store_true", help="Build the cache(s) and stop.")
    args = ap.parse_args(argv)

    session_dir = Path(args.session_dir)
    stats_dir = Path(args.stats_dir) if args.stats_dir else default_stats_dir(session_dir)
    out_dir = Path(args.out_dir) if args.out_dir else stats_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or session_dir.name
    summary = json.loads((stats_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    lm = ioi_landmarks.load(session_dir)
    print(f"Session: {session_dir.name}  stats: {stats_dir}  label: {label}")

    print("Timelapse: building ROI and out-region masks (same as the session figures)...")
    masks = spf.build_masks(session_dir, summary)
    region = tuple(int(v) for v in summary["analysis_region_fullres"])
    roi, out = masks["roi"].astype(bool), masks["out"].astype(bool)
    b = max(1, int(args.bin))

    # grid spacing: the measured camera interval (checked again per condition in the report)
    dt = spf.camera_frame_interval(session_dir)

    def get(cond_tag: str, sdir: Path, sess: Path) -> dict:
        cache = out_dir / f"{label}_timelapse_cache_{cond_tag}.npz"
        summ = json.loads((sdir / "analysis_summary.json").read_text(encoding="utf-8"))
        ids = [int(t) for t in summ["usable_trial_ids"]]
        key = json.dumps({"trials": ids, "region": region, "bin": b, "dt": round(dt, 6),
                          "roi_px": int(roi.sum()), "session": str(sess.name)})
        if args.reuse_cache and cache.exists():
            z = np.load(cache, allow_pickle=False)
            if str(z["key"]) == key:
                print(f"Timelapse: reusing cached {cond_tag} stack ({cache.name})")
                return {k: z[k] for k in z.files if k not in ("key", "report")} | {"report": json.loads(str(z["report"]))}
            print(f"Timelapse: cache {cache.name} is for different inputs; recomputing")
        res = average_condition(sess, ids, region, b, dt, roi, out, cond_tag)
        np.savez_compressed(cache, key=key, report=json.dumps(res["report"]),
                            **{k: v for k, v in res.items() if k != "report"})
        return res

    stim = get("main", stats_dir, session_dir)
    comp = None
    if args.compare_stats_dir:
        csess = Path(args.compare_session) if args.compare_session else session_dir
        comp = get("compare", Path(args.compare_stats_dir), csess)
    if args.compute_only:
        print("Done (compute only).")
        return 0

    orient, orient_src = resolve_orientation(args, lm)
    um, um_src = resolve_calibration(args, lm)
    print(f"Orientation: {orient_src}.  Calibration: {um if um else '-'} um/px ({um_src}).")

    valid = valid_mask(stim["baseline"])
    disp = display_stack(stim, valid, b, not args.no_demean, args.smooth_frames)
    disp_c = display_stack(comp, valid, b, not args.no_demean, args.smooth_frames) if comp else None

    # one grid for both conditions
    times = stim["times"]
    if comp is not None:
        t_all = np.union1d(np.round(times / dt), np.round(comp["times"] / dt)).astype(int)
        def regrid(d, tt, arr):
            outa = np.full((len(t_all),) + arr.shape[1:], np.nan)
            idx = np.searchsorted(t_all, np.round(tt / dt).astype(int))
            outa[idx] = arr
            return outa
        disp, disp_c = regrid(stim, stim["times"], disp), regrid(comp, comp["times"], disp_c)
        s_roi = regrid(stim, stim["times"], stim["roi_traces"].T).T
        s_out = regrid(stim, stim["times"], stim["out_traces"].T).T
        c_roi = regrid(comp, comp["times"], comp["roi_traces"].T).T
        times = t_all * dt
    else:
        s_roi, s_out, c_roi = stim["roi_traces"], stim["out_traces"], None

    in_win = (times >= AMP_WINDOW_S[0]) & (times <= AMP_WINDOW_S[1])
    win_map = np.nanmean(disp[in_win], axis=0) if in_win.any() else np.nanmean(disp, axis=0)
    # auto: the session figures' own rule (0.5/99.5 percentile, rounded up the same
    # NICE ladder, in % dR/R) applied to the amplitude-window average
    vmax = args.fixed_vmax or spf.auto_vmax(win_map[np.isfinite(win_map)])
    print(f"Colour scale: +/-{vmax:g} % dR/R, fixed for every frame ({'given' if args.fixed_vmax else 'auto from the amplitude window'}).")

    present = np.array([np.isfinite(disp[k]).any() for k in range(len(times))])
    spans = missing_spans(times, present, dt)
    rep = stim["report"]
    stim_dur = (rep.get("stimulus_duration_s") or {}).get("median")
    green = green_display(session_dir, region, b)
    shape = green.shape
    um_b = um * b if um else None

    series = [(times, s_roi, C_STIM, f"{args.stim_label}, ROI (n={s_roi.shape[0]})", False)]
    if c_roi is not None:
        series.append((times, c_roi, C_COMPARE, f"{args.compare_label}, same ROI (n={c_roi.shape[0]})", False))
    series.append((times, s_out, C_OUT, "out-region", True))
    xlim = (times[0] - dt, times[-1] + dt)
    note = (f"Trial-averaged ΔR/R, {b}×{b} binned, σ = {PIPELINE_SIGMA * PIPELINE_BINNING / b:.1f} px, "
            f"{args.smooth_frames * dt:.1f} s moving average"
            + ("" if args.no_demean else ", field median removed per frame")
            + f".  Fixed scale ±{vmax:g} %.  Hatched: frames not saved.")

    params = dict(
        session=session_dir.name, stats_dir=str(stats_dir), compare_stats_dir=args.compare_stats_dir,
        trials=[int(t) for t in stim["trials"]], compare_trials=[int(t) for t in comp["trials"]] if comp else None,
        timing=rep, compare_timing=comp["report"] if comp else None, grid_s=dt,
        region_fullres=region, bin=b, smooth_frames=args.smooth_frames, demeaned=not args.no_demean,
        vmax_percent=vmax, um_per_px=um, calibration_source=um_src, orientation_source=orient_src,
        not_saved_spans_s=[[round(a, 3), round(c, 3)] for a, c in spans],
        roi_px=int(roi.sum()), out_region_px=int(out.sum()), created=datetime.now().isoformat(timespec="seconds"),
    )

    # ── movie ─────────────────────────────────────────────────────────────
    ncol = 2 if disp_c is not None else 1
    fig = plt.figure(figsize=(40 / 3, 7.5), dpi=120, facecolor="white")  # 1600 x 900
    img_top, img_h = 0.30, 0.62
    cbw = 0.018
    pw = (0.83 - 0.04 * (ncol - 1)) / ncol
    axes_img, overlays = [], []
    for j in range(ncol):
        axi = fig.add_axes([0.04 + j * (pw + 0.04), img_top, pw, img_h])
        axi.set_axis_off()
        axi.imshow(green, cmap="gray", interpolation="bilinear", vmin=0, vmax=1)
        ov = axi.imshow(np.zeros(shape + (4,)), interpolation="bilinear")
        axi.set_xlim(0, shape[1]); axi.set_ylim(shape[0], 0)
        n_here = s_roi.shape[0] if j == 0 else c_roi.shape[0]
        axi.set_title(f"{args.stim_label if j == 0 else args.compare_label} (n = {n_here})", fontsize=14, pad=6)
        if j == 0:
            if um_b:
                draw_scalebar(axi, shape, um_b, 11)
            draw_compass(axi, shape, orient, 11)
        draw_landmarks(axi, lm, region, b, 11, um)
        axes_img.append(axi); overlays.append(ov)
    cax = fig.add_axes([0.885, img_top + 0.05, cbw, img_h - 0.10])
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(-vmax, vmax), cmap="RdBu"), cax=cax)
    cb.set_ticks(np.linspace(-vmax, vmax, 5))
    cb.set_label("ΔR/R (%)\nnegative = activation", fontsize=11)
    tlabel = fig.text(0.04, 0.955, "", fontsize=22, fontweight="bold", va="center")
    badge = fig.text(0.24, 0.955, "", fontsize=13, fontweight="bold", color="white", va="center",
                     bbox=dict(boxstyle="round,pad=0.35", facecolor="#B03A2E", edgecolor="none"))
    gapnote = fig.text(0.04 + pw / 2, img_top + img_h / 2, "", ha="center", va="center", fontsize=15,
                       color="white", bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.6))
    fig.text(0.965, 0.955, label, ha="right", va="center", fontsize=11, color="0.3")
    axt = fig.add_axes([0.07, 0.105, 0.86, 0.15])
    plot_traces(axt, series, stim_dur, spans, xlim, 11)
    cursor = axt.axvline(times[0], color="black", lw=1.4, zorder=5)
    fig.text(0.5, 0.006, note, ha="center", va="bottom", fontsize=8.5, color="0.35")

    def update(k: int) -> None:
        t = times[k]
        tlabel.set_text(f"t = {t:+.1f} s")
        on = stim_dur is not None and 0 <= t <= stim_dur
        badge.set_text("STIMULUS ON" if on else "")
        badge.set_visible(on)
        frames = [disp[k]] + ([disp_c[k]] if disp_c is not None else [])
        for ov, fr in zip(overlays, frames):
            ov.set_data(overlay_rgba(fr, vmax) if np.isfinite(fr).any() else np.zeros(shape + (4,)))
        missing = not np.isfinite(disp[k]).any()
        gapnote.set_text("frames not saved at this time" if missing else "")
        gapnote.set_visible(missing)
        cursor.set_xdata([t, t])

    fmt = args.format
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg  # optional: pip install imageio-ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if fmt == "auto":
        fmt = "mp4" if ffmpeg else "gif"
    if fmt == "mp4" and not ffmpeg:
        raise SystemExit("MP4 needs ffmpeg (on PATH, or `py -3.10 -m pip install imageio-ffmpeg`). Use --format gif.")
    movie = out_dir / f"{label}_timelapse.{fmt}"
    print(f"Timelapse: rendering {len(times)} frames -> {movie.name}")
    if fmt == "mp4":
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg
        from matplotlib.animation import FFMpegWriter
        # H.264/yuv420p needs even frame dimensions; round down rather than trust figure maths
        writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=4000,
                              extra_args=["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p"])
        dpi = 120
    else:
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=args.fps)
        dpi = 72
    hold = int(round(args.fps))  # hold the last frame ~1 s
    with writer.saving(fig, str(movie), dpi):
        for k in range(len(times)):
            update(k)
            writer.grab_frame()
        for _ in range(hold):
            writer.grab_frame()
    plt.close(fig)

    # ── filmstrip ─────────────────────────────────────────────────────────
    ft = [float(v) for v in args.filmstrip_times.split(",") if v.strip()]
    nrow = 2 if disp_c is not None else 1
    fig = plt.figure(figsize=(2.6 * len(ft) + 1.2, 2.0 * nrow + 2.4), dpi=200, facecolor="white")
    left, right, top_ = 0.05, 0.90, 0.93
    tile_w = (right - left) / len(ft)
    tile_h = tile_w * fig.get_figwidth() / fig.get_figheight() * shape[0] / shape[1]
    strip_rows = [disp] + ([disp_c] if disp_c is not None else [])
    row_names = [args.stim_label] + ([args.compare_label] if disp_c is not None else [])
    for r_i, (stack, rname) in enumerate(zip(strip_rows, row_names)):
        y = top_ - (r_i + 1) * tile_h - r_i * 0.02
        for c_i, t in enumerate(ft):
            axi = fig.add_axes([left + c_i * tile_w + 0.003, y, tile_w - 0.006, tile_h])
            axi.set_axis_off()
            axi.imshow(green, cmap="gray", interpolation="bilinear", vmin=0, vmax=1)
            sel = np.abs(times - t) <= dt * 1.01
            fr = np.nanmean(stack[sel], axis=0) if sel.any() and np.isfinite(stack[sel]).any() else None
            if fr is not None:
                axi.imshow(overlay_rgba(fr, vmax), interpolation="bilinear")
            else:
                axi.text(shape[1] / 2, shape[0] / 2, "not saved", ha="center", va="center", color="white",
                         fontsize=9, bbox=dict(boxstyle="round", facecolor="black", alpha=0.6))
            axi.set_xlim(0, shape[1]); axi.set_ylim(shape[0], 0)
            if r_i == 0:
                on = stim_dur is not None and 0 <= t <= stim_dur
                axi.set_title(f"{t:+.1f} s", fontsize=11, fontweight="bold", color="#922B21" if on else "black")
            if c_i == 0:
                axi.text(-0.04 * shape[1], shape[0] / 2, "\n".join(textwrap.wrap(rname, 16)), rotation=90,
                         ha="right", va="center", fontsize=9, multialignment="center")
                if r_i == 0:
                    if um_b:
                        draw_scalebar(axi, shape, um_b, 6)
                    draw_compass(axi, shape, orient, 6)
            draw_landmarks(axi, lm, region, b, 6, um)
    y_last = top_ - nrow * tile_h - (nrow - 1) * 0.02
    cax = fig.add_axes([right + 0.012, y_last + 0.1 * tile_h, 0.012, nrow * tile_h - 0.2 * tile_h])
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(-vmax, vmax), cmap="RdBu"), cax=cax)
    cb.set_ticks([-vmax, 0, vmax]); cb.ax.tick_params(labelsize=7)
    cb.set_label("ΔR/R (%)", fontsize=8)
    axt = fig.add_axes([left + 0.03, 0.16, right - left - 0.03, y_last - 0.26])
    plot_traces(axt, series, stim_dur, spans, xlim, 9)
    for t in ft:
        axt.axvline(t, color="black", lw=0.7, ls=":", zorder=4)
    fig.text(0.5, 0.004, note, ha="center", va="bottom", fontsize=6.5, color="0.35")
    strip = out_dir / f"{label}_timelapse_filmstrip.png"
    fig.savefig(strip, dpi=200, facecolor="white")
    plt.close(fig)

    params["outputs"] = [movie.name, strip.name]
    (out_dir / f"{label}_timelapse.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"Done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
