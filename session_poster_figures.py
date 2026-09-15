#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
session_poster_figures.py

Poster-style per-session figures, adapted from the lab's earlier
per-session figure scripts into a single, general-purpose,
per-session-invokable tool, so a session run through the GUI's Statistics
tab produces figures in the same visual language: colors, scale bar,
compass, colorbar, autoscale ladder.

Consumes a session's own statistical_analyses.py output (analysis_summary.json,
loo_roi_amplitude_summary.csv, roi_timecourse_raw.npz) plus the session's raw
per-trial arrays and frames. Writes, into --out-dir (default: the stats
folder itself):

    <label>_drr_panel.png       vasculature + trial-mean dR/R + t-map cluster
                                 + LOO ROI outline + out-region circle,
                                 1 mm scale bar, A/P/M/L compass, colorbar.
                                 Autoscaled to the session's own signal.
    <label>_heat.png            same panel, heatmap only (no outlines/legend/
                                 out-region circle).
    <label>_drr_panel_v05.png   \\
    <label>_heat_v05.png        / same two, fixed +/-0.05% scale (--fixed-vmax)
                                   for side-by-side comparison across sessions.
    <label>_green_reference.png full-frame vasculature reference, the
                                 analysis region marked with a dashed
                                 rectangle drawn entirely inside it.
    <label>_targeting_reference.png  same as _green_reference.png plus the
                                 LOO ROI as a solid outline -- no legend,
                                 scale bar, compass, or colorbar. A clean
                                 deliverable for targeting implants/
                                 injections relative to vasculature/
                                 anatomical landmarks.
    <label>_timecourse.png      ROI dR/R(t), mean +/- SEM, gap frames plotted
                                 rather than blanked (the "acquisition gap"
                                 annotation only appears when at least one
                                 plotted series still has a real break there),
                                 the out-of-activation trace (this session,
                                 same trials, a 180-degree-reflected mask --
                                 no second session needed), and, if
                                 --compare-session is given, that session's
                                 own trials measured in THIS session's ROI.
                                 --no-title / --show-amplitude-labels tweak
                                 the header and per-series value labels.
    <label>_extraction_cache.npz  out-region (and compare-session, if used)
                                 per-trial timecourses, for --reuse-
                                 extraction-cache on a re-run.

Two rig-dependent defaults, editable per run: spatial calibration
(--um-per-px, default 3.682, i.e. a 7.07 mm field across 1920 px, measured
against a caliper on this lab's rig) and imaging orientation
(--anterior-side/--medial-side, default left/bottom, a carried-forward
assumption rather than a per-session measurement). Confirm both against
your own rig before quoting a physical distance or a compass direction
from these figures.

Never touches statistical_analyses.py's own output files; only reads them.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label as cc_label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Rectangle

W, H = 1920, 1200
SMOOTH_SIGMA = 5.0
CLUSTER_T_THRESH = 3.0
TOP_FRAC = 0.05

C_CLUSTER, C_ROI = "#FF00C8", "#B2FF00"
C_STIM, C_COMPARE, C_OUT = "#B03A2E", "#7B8A8B", "#2F6DA4"
CB_STRIP = 2.95
FS_LEG, FS_TICK, FS_CBLAB = 27, 23, 22
SCALEBAR_MM = 1.0
# Measured 2026-09-03 against a caliper (1 mm graduations) imaged through the
# current aperture/focus: 271.6 px/mm over the full 1920 px frame -> a 7.07 mm
# field, i.e. 3.682 um/px. Supersedes the old 6.5 mm / 1920 px carried-forward
# assumption. Re-measure if the objective, aperture, or working distance change.
DEFAULT_UM_PER_PX = 7070.0 / 1920.0
DEFAULT_FIXED_VMAX = 0.05

# Fixed ladder for the per-session autoscale: the scale is chosen by the
# data and not by hand, so no session can be quietly flattered by a tighter
# scale than its neighbour.
NICE_VMAX = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06,
             0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]

_SIDES = ("left", "right", "top", "bottom")
_OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}


# ───────────────────────────── orientation / calibration ──────────────────

def orientation_map(anterior_side: str, medial_side: str) -> dict:
    """Full 4-edge compass mapping from just the anterior and medial edges
    (the other two follow: posterior is anterior's opposite, lateral is
    medial's opposite)."""
    if anterior_side not in _SIDES or medial_side not in _SIDES:
        raise ValueError(f"anterior/medial side must be one of {_SIDES}")
    if medial_side in (anterior_side, _OPPOSITE[anterior_side]):
        raise ValueError("Medial side must be perpendicular to the anterior side, not the same edge or its opposite.")
    return {
        anterior_side: "A",
        _OPPOSITE[anterior_side]: "P",
        medial_side: "M",
        _OPPOSITE[medial_side]: "L",
    }


def midline_centered_orientation_map(anterior_side: str) -> dict:
    """Compass for a bilateral image with the cortical midline running down
    the centre of the frame: the A-P axis is the anterior edge and its
    opposite, and BOTH perpendicular edges are lateral (medial is the central
    midline, not an edge -- render_panel draws it as a centre 'M'). Unlike
    orientation_map() there is no medial edge to name."""
    if anterior_side not in _SIDES:
        raise ValueError(f"anterior side must be one of {_SIDES}")
    lateral = [s for s in _SIDES if s not in (anterior_side, _OPPOSITE[anterior_side])]
    return {
        anterior_side: "A",
        _OPPOSITE[anterior_side]: "P",
        lateral[0]: "L",
        lateral[1]: "L",
        "center": "M",
    }


# ───────────────────────────── per-trial data ──────────────────────────────

def load_trial_metadata(session_dir: Path, trial: int) -> dict:
    path = session_dir / f"trial_{trial:03d}" / "meta" / "trial_metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def crop_bounds(session_dir: Path, trial: int) -> tuple[int, int, int, int]:
    active_roi = load_trial_metadata(session_dir, trial).get("active_roi")
    if not active_roi:
        return 0, 0, W, H
    crop = active_roi.get("analysis_crop") or active_roi.get("requested")
    if not crop:
        return 0, 0, W, H
    return int(crop["x"]), int(crop["y"]), int(crop["width"]), int(crop["height"])


def trial_drr_map(session_dir: Path, trial: int) -> np.ndarray:
    a = session_dir / f"trial_{trial:03d}" / "analysis"
    b = np.load(a / "baseline_reference.npy")
    p = np.load(a / "post_mean_analysis_window.npy")
    drr = (p - b) / np.where(b > 0, b, np.nan)
    return gaussian_filter(np.nan_to_num(drr, nan=0.0), sigma=SMOOTH_SIGMA)


def place_binned_mask(binned: np.ndarray, byf: int, bxf: int, crop: tuple[int, int, int, int]) -> np.ndarray:
    cx0, cy0, cw, ch = crop
    full = np.zeros((H, W), dtype=binned.dtype)
    up = np.repeat(np.repeat(binned, byf, axis=0), bxf, axis=1)
    hh, ww = min(up.shape[0], H - cy0), min(up.shape[1], W - cx0)
    full[cy0:cy0 + hh, cx0:cx0 + ww] = up[:hh, :ww]
    return full


# ───────────────────────────────────────── masks ──────────────────────────

def build_masks(session_dir: Path, summary: dict) -> dict:
    """Full-session (non-LOO) ROI, its 180-degree out-region reflection
    (an area-matched patch outside the activation, entirely derivable from
    this one session), and the cluster mask.

    Distinct from loo_core_mask() below: this ROI is the top-5% of the
    FULL-session mean map (no leave-one-out), used only for the out-region
    reflection and any cross-session ROI transfer -- not the panel's own
    displayed outline, which uses the more conservative LOO-fold
    intersection instead.
    """
    trial_ids = summary["usable_trial_ids"]
    rx0, ry0, rw, rh = summary["analysis_region_fullres"]
    crop = crop_bounds(session_dir, trial_ids[0])
    cx0, cy0, cw, ch = crop

    maps = [trial_drr_map(session_dir, t) for t in trial_ids]
    mean_map = np.mean(maps, axis=0)
    sh = mean_map.shape
    byf, bxf = ch // sh[0], cw // sh[1]
    by0 = max(ry0 - cy0, 0) // byf
    by1 = min((ry0 + rh - cy0) // byf, sh[0])
    bx0 = max(rx0 - cx0, 0) // bxf
    bx1 = min((rx0 + rw - cx0) // bxf, sh[1])
    binned_slice = (by0, by1, bx0, bx1)

    sub = mean_map[by0:by1, bx0:bx1]
    thr = np.percentile(sub, TOP_FRAC * 100.0)
    roi_b = np.zeros(sh, dtype=bool)
    roi_b[by0:by1, bx0:bx1] = sub <= thr
    # 180-degree point reflection about the region's own centre -- flip
    # both axes of the SAME sub-rectangle the ROI was selected from, then
    # place it back at that identical location.
    out_b = np.zeros(sh, dtype=bool)
    out_b[by0:by1, bx0:bx1] = roi_b[by0:by1, bx0:bx1][::-1, ::-1]

    stack = np.stack(maps, axis=0)[:, by0:by1, bx0:bx1]
    stack = stack - stack.mean(axis=(1, 2), keepdims=True)
    n = stack.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_map = np.nan_to_num(stack.mean(0) / (stack.std(0, ddof=1) / np.sqrt(n)))
    lbl, ncomp = cc_label(t_map < -CLUSTER_T_THRESH)
    cl_b = np.zeros(sh, dtype=bool)
    if ncomp:
        biggest = int(np.argmax(np.bincount(lbl.ravel())[1:])) + 1
        cl_b[by0:by1, bx0:bx1] = (lbl == biggest)

    return dict(
        roi=place_binned_mask(roi_b, byf, bxf, crop),
        out=place_binned_mask(out_b, byf, bxf, crop),
        cluster=place_binned_mask(cl_b, byf, bxf, crop),
        mean_map=mean_map, maps=maps, crop=crop,
        region=(rx0, ry0, rw, rh), binned_slice=binned_slice,
        byf=byf, bxf=bxf, map_shape=sh,
    )


def loo_core_mask(maps: list, binned_slice: tuple, restrict_to_region: bool) -> np.ndarray:
    """Intersection of the top-5% mask across every leave-one-out fold --
    the more conservative ROI drawn as the panel's outline. Recomputed here
    (not read from loo_masks/) so this script runs against any session's
    per-trial arrays with no dependency on --save-loo-masks. Reproduces
    statistical_analyses.py's make_roi_from_maps()."""
    by0, by1, bx0, bx1 = binned_slice
    sh = maps[0].shape
    core = None
    for i in range(len(maps)):
        fold_mean = np.mean([m for j, m in enumerate(maps) if j != i], axis=0)
        if restrict_to_region:
            sub = fold_mean[by0:by1, bx0:bx1]
            k = np.zeros(sh, dtype=bool)
            k[by0:by1, bx0:bx1] = sub <= np.percentile(sub, TOP_FRAC * 100.0)
        else:
            k = fold_mean <= np.percentile(fold_mean, TOP_FRAC * 100.0)
        core = k if core is None else (core & k)
    return core if core is not None else np.zeros(sh, dtype=bool)


# ───────────────────────────── green reference ─────────────────────────────

def green_reference_mean(session_dir: Path, trim_default: int = 5) -> np.ndarray:
    """Mean green reference frame, full sensor resolution."""
    files = sorted(glob.glob(str(session_dir / "session_green_reference" / "*.raw")))
    if not files:
        files = sorted(glob.glob(str(session_dir / "trial_001" / "green" / "*.raw")))
    if not files:
        raise FileNotFoundError(f"no green reference frames under {session_dir}")
    trim = trim_default
    meta_files = sorted(glob.glob(str(session_dir / "trial_*/meta/trial_metadata.json")))
    if meta_files:
        cfg = json.loads(Path(meta_files[0]).read_text())["trial_config"]
        trim = int(cfg.get("green_reference_trim_frames", trim_default))
    use = files[trim:] if len(files) > trim + 1 else files
    acc = np.zeros(W * H, dtype=np.float64)
    for f in use:
        acc += np.fromfile(f, dtype=np.uint16).astype(np.float64)
    return (acc / len(use)).reshape(H, W)


def _render_green_reference_figure(session_dir: Path, region_fullres: tuple, roi_full: np.ndarray | None):
    """Shared core for render_green_reference()/render_targeting_reference():
    full-frame green reference + the analysis region as a dashed rectangle,
    optionally with an ROI mask contoured on top. No legend, scale bar,
    compass, or colorbar -- callers add nothing further."""
    x0, y0, w, h = region_fullres
    g = green_reference_mean(session_dir)
    lo, hi = np.percentile(g, [0.2, 99.8])
    gn = np.clip((g - lo) / (hi - lo), 0, 1) ** 0.85

    dpi = 100.0
    fig = plt.figure(figsize=(W / dpi, H / dpi))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(gn, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    full_frame = (x0, y0, w, h) == (0, 0, W, H)
    if not full_frame:
        halo_px, dash_px, aa_guard = 6.0, 3.0, 1.5
        px2pt = 72.0 / dpi
        pad = halo_px / 2.0 + aa_guard
        rx, ry, rw, rh = x0 + pad, y0 + pad, w - 2 * pad, h - 2 * pad
        ax.add_patch(Rectangle((rx, ry), rw, rh, fill=False, edgecolor="black",
                                linewidth=halo_px * px2pt, alpha=0.45, zorder=5))
        ax.add_patch(Rectangle((rx, ry), rw, rh, fill=False, edgecolor="white",
                                linewidth=dash_px * px2pt, linestyle=(0, (9, 7)), zorder=6))

    if roi_full is not None and roi_full.any():
        ax.contour(roi_full.astype(float), levels=[0.5], colors=C_ROI, linewidths=3.0, zorder=7)

    return fig, dpi


def render_green_reference(session_dir: Path, region_fullres: tuple, out_path: Path) -> None:
    fig, dpi = _render_green_reference_figure(session_dir, region_fullres, roi_full=None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, pad_inches=0)
    plt.close(fig)


def render_targeting_reference(session_dir: Path, region_fullres: tuple, roi_full: np.ndarray, out_path: Path) -> None:
    """Clean deliverable for targeting implants/injections relative to
    vasculature/anatomical landmarks: full-frame green reference, the
    analysis crop as a dashed rectangle, and the LOO ROI as a solid outline
    -- no legend, scale bar, compass, or colorbar (see _render_green_
    reference_figure)."""
    fig, dpi = _render_green_reference_figure(session_dir, region_fullres, roi_full=roi_full)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, pad_inches=0)
    plt.close(fig)


# ───────────────────────────── cortical panel figure ───────────────────────

def auto_vmax(demeaned_map: np.ndarray) -> float:
    """Symmetric display limit: the larger tail of the 0.5/99.5 percentile
    range, rounded UP to the next NICE_VMAX step. Percentiles rather than
    min/max so a handful of hot pixels or a vessel edge can't set the scale."""
    finite = demeaned_map[np.isfinite(demeaned_map)]
    lo, hi = np.percentile(finite, [0.5, 99.5])
    r = max(abs(float(lo)), abs(float(hi)))
    for n in NICE_VMAX:
        if r <= n * 1.0001:
            return n
    return NICE_VMAX[-1]


def render_panel(
    dmap_full: np.ndarray, roi_full: np.ndarray, cluster_full: np.ndarray,
    green_full: np.ndarray, region_fullres: tuple, out_path: Path, *,
    vmax: float | None, outlines: bool, out_region_mask: np.ndarray | None,
    um_per_px: float | None, orient: dict, colorbar: bool = True,
) -> float:
    """One annotated cortical panel. Returns the vmax actually used.
    um_per_px=None draws no scale bar (spatial calibration unknown).
    """
    x0, y0, w, h = region_fullres
    gn = green_full[y0:y0 + h, x0:x0 + w]
    dmap = dmap_full[y0:y0 + h, x0:x0 + w]
    cluster = cluster_full[y0:y0 + h, x0:x0 + w]
    roi = roi_full[y0:y0 + h, x0:x0 + w]
    lo_g, hi_g = np.percentile(gn, [0.2, 99.8])
    gn = np.clip((gn - lo_g) / (hi_g - lo_g + 1e-9), 0, 1) ** 0.85

    Hh, Ww = gn.shape
    iw, ih = Ww / 100.0, Hh / 100.0
    fw = iw + (CB_STRIP if colorbar else 0.0)
    if vmax is None:
        vmax = auto_vmax(dmap)

    norm = Normalize(-vmax, vmax)
    rgba = plt.get_cmap("RdBu")(norm(np.nan_to_num(dmap)))
    alpha = np.clip(np.abs(dmap) / vmax, 0, 1) ** 0.7
    alpha[np.isnan(dmap)] = 0.0
    alpha[alpha < 0.18] = 0.0
    rgba[..., 3] = np.clip(alpha, 0, 0.88)

    fig = plt.figure(figsize=(fw, ih))
    ax = fig.add_axes([0, 0, iw / fw, 1])
    ax.set_axis_off()
    ax.imshow(gn, cmap="gray", interpolation="bilinear")
    ax.imshow(rgba, interpolation="bilinear")

    if outlines:
        if cluster.any():
            ax.contour(cluster.astype(float), levels=[0.5], colors=C_CLUSTER, linewidths=2.8)
        if roi.any():
            ax.contour(roi.astype(float), levels=[0.5], colors=C_ROI, linewidths=3.0)
    if out_region_mask is not None:
        m = out_region_mask[y0:y0 + h, x0:x0 + w]
        if m.any():
            ys, xs = np.nonzero(m)
            ax.add_patch(Circle(
                (xs.mean(), ys.mean()), np.sqrt(m.sum() / np.pi), fill=False, ec="white",
                lw=4.2, linestyle=(0, (2.2, 2.6)), zorder=7,
                path_effects=[pe.Stroke(linewidth=7.9, foreground="black", alpha=0.55), pe.Normal()],
            ))

    ax.set_xlim(0, Ww)
    ax.set_ylim(Hh, 0)

    M = 0.028 * Ww
    if um_per_px is not None:
        bar = SCALEBAR_MM * 1000.0 / um_per_px
        bh = 0.011 * Ww
        pad = 0.013 * Ww
        lab_h = FS_LEG / 0.72
        bw = bar + 2 * pad
        bhh = pad + lab_h + 0.006 * Hh + bh + pad
        ax.add_patch(FancyBboxPatch((Ww - M - bw, Hh - M - bhh), bw, bhh,
                                     boxstyle=f"round,pad=0,rounding_size={0.012 * Ww}",
                                     facecolor="black", alpha=0.5, edgecolor="none", zorder=8))
        ax.add_patch(Rectangle((Ww - M - pad - bar, Hh - M - pad - bh), bar, bh,
                                facecolor="white", edgecolor="black", lw=1.2, zorder=9))
        ax.text(Ww - M - pad - bar / 2, Hh - M - pad - bh - 0.006 * Hh, f"{SCALEBAR_MM:g} mm",
                ha="center", va="bottom", fontsize=FS_LEG, fontweight="bold", color="white", zorder=9)

    leg = None
    if outlines:
        leg = ax.legend(
            handles=[
                Patch(facecolor="none", edgecolor=C_CLUSTER, linewidth=3.4, label="t-map cluster"),
                Patch(facecolor="none", edgecolor=C_ROI, linewidth=3.4, label="LOO ROI core"),
            ],
            loc="lower left", bbox_to_anchor=(M / Ww, M / Hh), fontsize=FS_LEG, frameon=True,
            facecolor="white", framealpha=0.9, edgecolor="0.35", handlelength=1.5,
            handleheight=1.0, borderpad=0.55, labelspacing=0.45,
        )
    fig.canvas.draw()
    if leg is not None:
        bb = leg.get_window_extent().transformed(ax.transData.inverted())
        lx0 = min(bb.x0, bb.x1)
        cbot = min(bb.y0, bb.y1) - 0.014 * Ww
    else:
        lx0, cbot = M, Hh - M

    arm = 0.030 * Ww
    gap = 0.015 * Ww
    fs_c = FS_LEG * 0.72
    lh = fs_c / 0.72 * 0.5
    cpad = 0.011 * Ww
    ext = arm + gap + lh + cpad
    ccx, ccy = lx0 + ext, cbot - ext
    ax.add_patch(FancyBboxPatch((ccx - ext, ccy - ext), 2 * ext, 2 * ext,
                                 boxstyle=f"round,pad=0,rounding_size={0.014 * Ww}",
                                 facecolor="black", alpha=0.5, edgecolor="none", zorder=8))
    ax.plot([ccx - arm, ccx + arm], [ccy, ccy], color="white", lw=2.2, zorder=9, solid_capstyle="round")
    ax.plot([ccx, ccx], [ccy - arm, ccy + arm], color="white", lw=2.2, zorder=9, solid_capstyle="round")
    d = (arm + ext) / 2
    for lx, ly, side in ((ccx - d, ccy, "left"), (ccx + d, ccy, "right"),
                          (ccx, ccy - d, "top"), (ccx, ccy + d, "bottom")):
        ax.text(lx, ly, orient[side], ha="center", va="center", fontsize=fs_c,
                fontweight="bold", color="white", zorder=9)
    # Midline-centered layout: medial is the central midline, not an edge --
    # mark it at the compass centre, haloed so it reads over the cross lines.
    if orient.get("center"):
        ax.text(ccx, ccy, orient["center"], ha="center", va="center", fontsize=fs_c,
                fontweight="bold", color="white", zorder=10,
                path_effects=[pe.withStroke(linewidth=3.2, foreground="black")])

    if colorbar:
        cax = fig.add_axes([(iw + 0.30) / fw, 0.055, 0.46 / fw, 0.89])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm, cmap="RdBu"), cax=cax)
        cb.set_ticks(np.linspace(-vmax, vmax, 5))
        dp = 4 if vmax < 0.02 else 3
        cb.ax.set_yticklabels([(f"{0:.{dp}f}" if abs(t) < 1e-12 else f"{t:+.{dp}f}")
                               for t in np.linspace(-vmax, vmax, 5)])
        cb.ax.tick_params(labelsize=FS_TICK, length=6, width=1.4)
        cb.outline.set_linewidth(1.2)
        cb.set_label("trial-mean $\\Delta$R/R (%)\nnegative = activation",
                      fontsize=FS_CBLAB, labelpad=12, linespacing=1.45)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, transparent=True, pad_inches=0)
    plt.close(fig)
    return vmax


# ───────────────────────────── timecourse extraction ───────────────────────

def parse_ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def camera_frame_interval(session_dir: Path) -> float:
    counts = []
    with open(session_dir / "session_log.csv") as fp:
        for row in csv.DictReader(fp):
            if row["phase"] == "baseline_collect":
                counts.append(int(row["camera_timestamp"]))
            elif counts:
                break
    return float(np.median(np.diff(counts)) / 1e9)


def phase_offsets(session_dir: Path) -> tuple[float, float]:
    per_trial: dict = {}
    with open(session_dir / "marker_log.csv") as fp:
        for row in csv.DictReader(fp):
            if row["trial_index"] == "0":
                continue
            millis = row["arduino_millis"] or row["raw_text"].split(",")[1].strip()
            per_trial.setdefault(row["trial_index"], {})[row["marker"]] = int(millis)
    need = {"STIM_START", "BASELINE_LAST_FRAME_TRIGGER", "POST_FIRST_FRAME_TRIGGER"}
    bl = [(m["BASELINE_LAST_FRAME_TRIGGER"] - m["STIM_START"]) / 1000.0 for m in per_trial.values() if need <= set(m)]
    pf = [(m["POST_FIRST_FRAME_TRIGGER"] - m["STIM_START"]) / 1000.0 for m in per_trial.values() if need <= set(m)]
    return float(np.median(bl)), float(np.median(pf))


def session_axes(session_dir: Path, summary: dict) -> dict:
    trial_ids = summary["usable_trial_ids"]
    cfg = load_trial_metadata(session_dir, trial_ids[0])["trial_config"]
    nb, npost = cfg["baseline_frames"], cfg["post_frames"]
    f0, f1 = cfg["analysis_start_frame"], cfg["analysis_end_frame"]
    dt = camera_frame_interval(session_dir)
    b_last, p_first = phase_offsets(session_dir)
    t_axis = np.concatenate([
        b_last - (nb - 1 - np.arange(nb)) * dt,
        p_first + np.arange(npost) * dt,
    ])
    return dict(nb=nb, npost=npost, f0=f0, f1=f1, dt=dt, b_last=b_last, p_first=p_first,
                t_axis=t_axis, w0=float(t_axis[nb + f0]), w1=float(t_axis[nb + f1 - 1]))


def extract_mask_timecourse(session_dir: Path, trial_ids: list, mask: np.ndarray) -> dict:
    """Per-trial (baseline+post) and gap dR/R(%) timecourses for `mask`
    applied to session_dir's raw frames, plus the per-trial amplitude
    (mean over the amplitude window). Same math as statistical_analyses.py's
    roi_mean_timecourse(), generalized to any session/trial-list/mask
    combination -- used both for "this session's own out-region" and "a
    different session's trials measured in this session's ROI"."""
    axes = session_axes(session_dir, {"usable_trial_ids": trial_ids})
    values, gap_values = [], []
    for t in trial_ids:
        trial_dir = session_dir / f"trial_{t:03d}"
        b_files = sorted(glob.glob(str(trial_dir / "baseline" / "*.raw")))
        p_files = sorted(glob.glob(str(trial_dir / "post" / "*.raw")))
        g_files = sorted(glob.glob(str(trial_dir / "gap" / "*.raw")))

        def mean_in_mask(path: str) -> float:
            return float(np.fromfile(path, dtype=np.uint16).reshape(H, W)[mask].mean())

        b_vals = np.array([mean_in_mask(f) for f in b_files], dtype=float)
        r0 = float(b_vals.mean())
        p_vals = np.array([mean_in_mask(f) for f in p_files], dtype=float)
        v = 100.0 * (np.concatenate([b_vals, p_vals]) - r0) / r0
        values.append(v)
        if g_files:
            gv = np.array([mean_in_mask(f) for f in g_files], dtype=float)
            gap_values.append(100.0 * (gv - r0) / r0)
        else:
            gap_values.append(np.zeros(0))

    values = np.stack(values)
    nb, f0, f1 = axes["nb"], axes["f0"], axes["f1"]
    amps = values[:, nb:][:, f0:f1].mean(axis=1)
    return dict(trial_ids=np.array(trial_ids), values=values, gap_values=gap_values, amps=amps)


def _ragged_object_array(arrays: list) -> np.ndarray:
    """Build a genuine 1D object array of per-trial arrays for npz storage.

    np.array(list_of_arrays, dtype=object) is a numpy footgun: when every
    sub-array happens to have the SAME length (a session where every trial
    has the same gap-frame count -- the common case), numpy silently stacks
    them into a 2D array of boxed-object scalars instead of a 1D array of
    ndarray objects. Reloaded, each "row" is then a 1D dtype=object array
    of individually-boxed floats, and np.interp refuses to cast that to
    float64. Assigning into a pre-sized empty object array bypasses numpy's
    shape inference entirely, so this is safe regardless of whether the
    per-trial lengths are uniform or ragged."""
    out = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        out[i] = np.asarray(a, dtype=np.float64)
    return out


def gap_fills_span(gap_t: np.ndarray, b_last: float, p_first: float, tol: float) -> bool:
    if len(gap_t) == 0:
        return False
    boundary = np.concatenate([[b_last], np.sort(gap_t), [p_first]])
    diffs = np.diff(boundary)
    return bool(np.all(diffs >= -1e-9) and np.all(diffs <= tol))


def gridded_mean_sem(session_dir: Path, axes: dict, values: np.ndarray, gap_values: list, gap_times: list, grid: np.ndarray):
    """Mean +/- SEM on `grid`, splicing each trial's real gap frames into
    the acquisition-pause break where coverage is complete (matches
    statistical_analyses.py's own gap-fill logic). gap_times[i] are the
    per-trial gap-frame timestamps (relative to that trial's STIM_START);
    pass empty arrays for a session with no gap frames."""
    nb = axes["nb"]
    tol = max(2.5 * axes["dt"], 0.05)
    curves = []
    for i in range(values.shape[0]):
        # np.asarray(..., dtype=float64): gap_times/gap_values coming back
        # from an --reuse-extraction-cache .npz can carry dtype=object even
        # when each element is a plain number -- np.savez'ing a list of
        # same-length per-trial arrays collapses it into a 2D object array
        # of boxed scalars instead of a 1D array of float arrays, which
        # np.interp then refuses to cast. Force it back to float64 here so
        # a cache written by an older/buggy version of this script still
        # renders instead of crashing.
        gt = np.asarray(gap_times[i] if i < len(gap_times) else np.zeros(0), dtype=np.float64)
        gv = np.asarray(gap_values[i] if i < len(gap_values) else np.zeros(0), dtype=np.float64)
        n = min(len(gt), len(gv))
        if gap_fills_span(gt[:n], axes["b_last"], axes["p_first"], tol):
            order = np.argsort(gt[:n])
            tt = np.concatenate([axes["t_axis"][:nb], gt[:n][order], axes["t_axis"][nb:]])
            vv = np.concatenate([values[i][:nb], gv[:n][order], values[i][nb:]])
            curve = np.interp(grid, tt, vv, left=np.nan, right=np.nan)
        else:
            curve = np.interp(grid, axes["t_axis"], values[i], left=np.nan, right=np.nan)
            curve[(grid > axes["b_last"]) & (grid < axes["p_first"])] = np.nan
        curves.append(curve)
    curves = np.array(curves)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(curves, axis=0)
        n_valid = np.sum(~np.isnan(curves), axis=0)
        sem = np.nanstd(curves, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))
    mean = np.where(n_valid >= 3, mean, np.nan)
    sem = np.where(n_valid >= 3, sem, np.nan)
    return mean, sem, curves.shape[0]


def _declutter(positions: list, min_gap: float) -> list:
    """Nudge a set of 1D positions apart (symmetric pairwise relaxation) so no
    two end up closer than min_gap, while keeping their rank order and
    disturbing well-separated positions as little as possible. Used to keep
    the per-series amplitude-window value labels from overlapping when two
    series' means land close together."""
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    adjusted = [positions[i] for i in order]
    for _ in range(100):
        moved = False
        for i in range(1, len(adjusted)):
            gap = adjusted[i] - adjusted[i - 1]
            if gap < min_gap:
                shift = (min_gap - gap) / 2
                adjusted[i - 1] -= shift
                adjusted[i] += shift
                moved = True
        if not moved:
            break
    result = [0.0] * len(positions)
    for rank, i in enumerate(order):
        result[i] = adjusted[rank]
    return result


def _draw_amplitude_labels(fig, ax, series: list, w1: float) -> None:
    """Numeric dR/R value (e.g. "-0.068%") next to each series' mean-over-
    amplitude-window line, colored to match, on a soft borderless white wash
    for legibility over the traces/shading. Vertically decluttered in pixel
    space so two series with close means don't overlap -- a nudged label
    gets a thin leader line back to its actual line height so the value is
    still unambiguous.

    Deliberately NOT a patheffects.withStroke halo (glyph-outline stroking)
    or a manual multi-copy offset halo (many overlapping semi-transparent
    text copies): both were tried and both visibly corrupted specific
    glyphs at this font size under Agg -- "-0.019%" rendering with a
    spurious extra dot that reads as "-0:019%". Confirmed via connected-
    component analysis of the actual output pixels (not just eyeballing):
    an isolated, fully-opaque extra blob appeared near glyph boundaries that
    doesn't exist in the underlying SVG path geometry, so the corruption is
    specific to compositing many overlapping anti-aliased text draws under
    Agg. A single opaque bbox behind a single plain text draw has no
    overlapping-copy compositing to go wrong.

    Must run after fig.tight_layout() (needs the axes' final on-figure
    position for the pixel-space decluttering to be accurate) and before
    fig.savefig()."""
    fig.canvas.draw()
    inv = ax.transData.inverted()
    x0, x1 = ax.get_xlim()
    dx_leader, dx_text = 0.018 * (x1 - x0), 0.022 * (x1 - x0)
    m0_values = [m0 for *_, (m0, _s0) in series]
    raw_px = [ax.transData.transform((w1, m0))[1] for m0 in m0_values]
    adj_px = _declutter(raw_px, min_gap=15.0)
    for (*_, c, _lbl, _lw, (m0, _s0)), y_raw_px, y_adj_px in zip(series, raw_px, adj_px):
        y_label = inv.transform((0, y_adj_px))[1]
        if abs(y_adj_px - y_raw_px) > 1.5:
            ax.plot([w1, w1 + dx_leader], [m0, y_label], color=c, lw=0.8, alpha=0.6, zorder=7,
                     clip_on=False)
        ax.text(w1 + dx_text, y_label, f"{m0:+.3f}%", color=c, fontsize=8.2,
                ha="left", va="center", fontweight="bold", zorder=8, clip_on=False,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.82))


def render_timecourse(
    session_dir: Path, summary: dict, timecourse_raw: dict, out_extraction: dict,
    out_path: Path, *, title: str, compare_series: dict | None, compare_label: str,
    suppress_title: bool = False, show_amplitude_labels: bool = False,
) -> None:
    """ROI dR/R(t) mean +/- SEM, the out-region trace, and (optionally) a
    compare-session trace measured in this session's ROI."""
    axes = session_axes(session_dir, summary)
    grid = timecourse_raw["common_grid"]
    trial_ids = list(timecourse_raw["trial_ids"])

    roi_values = np.stack([timecourse_raw[f"trial_{t:02d}_v"] for t in trial_ids])
    roi_gap_t = [timecourse_raw.get(f"trial_{t:02d}_gap_t", np.zeros(0)) for t in trial_ids]
    roi_gap_v = [timecourse_raw.get(f"trial_{t:02d}_gap_v", np.zeros(0)) for t in trial_ids]
    roi_mean, roi_sem, roi_n = gridded_mean_sem(session_dir, axes, roi_values, roi_gap_v, roi_gap_t, grid)
    roi_amp_mean = float(np.mean(summary["_amps"]))
    roi_amp_sem = float(np.std(summary["_amps"], ddof=1) / np.sqrt(len(summary["_amps"]))) if len(summary["_amps"]) > 1 else 0.0

    series = []
    if compare_series is not None:
        cm, cs, cn = gridded_mean_sem(
            compare_series["session_dir"], axes, compare_series["values"],
            compare_series["gap_values"], compare_series["gap_times"], grid,
        )
        a = compare_series["amps"]
        series.append((cm, cs, C_COMPARE, f"{compare_label}, stim ROI (n={cn})", 1.7,
                       (float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0)))

    om, os_, on = gridded_mean_sem(
        session_dir, axes, out_extraction["values"], out_extraction["gap_values"],
        out_extraction["gap_times"], grid,
    )
    oa = out_extraction["amps"]
    series.append((om, os_, C_OUT, f"Stimulation, outside activation (n={on})", 1.7,
                   (float(oa.mean()), float(oa.std(ddof=1) / np.sqrt(len(oa))) if len(oa) > 1 else 0.0)))
    series.append((roi_mean, roi_sem, C_STIM, f"Stimulation, activation ROI (n={roi_n})", 2.0,
                   (roi_amp_mean, roi_amp_sem)))

    # Whether the acquisition-gap annotation is still warranted: gridded_mean_sem()
    # already decided, per series, whether that trial's real gap frames fully
    # spliced the break (gap_fills_span()) or had to be left NaN there. If every
    # series actually being plotted here came out fully covered, the curves
    # don't show a break at all, and the box would be describing something the
    # reader can't see -- so only draw it when at least one shown series still
    # has a real gap (matches today's behavior for a session with no/partial
    # gap-frame coverage; the box only newly disappears when coverage is
    # complete for every series drawn).
    gap_mask = (grid > axes["b_last"]) & (grid < axes["p_first"])
    gap_visible = any(np.isnan(m[gap_mask]).any() for m, *_ in series)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.axvspan(axes["w0"], axes["w1"], color="#FFD400", alpha=0.22,
               label=f"Amplitude window (stimulus frames {axes['f0'] + 1}-{axes['f1']})")
    if gap_visible:
        ax.axvspan(axes["b_last"], axes["p_first"], color="0.90", alpha=0.85, zorder=0)
    for m, s, c, lbl, lw, _ in series:
        ax.fill_between(grid, m - s, m + s, color=c, alpha=0.26, lw=0)
        ax.plot(grid, m, color=c, lw=lw, label=lbl)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="crimson", lw=1.1, ls="--", label="Stimulus onset")
    halo = [pe.Stroke(linewidth=4.4, foreground="white"), pe.Normal()]
    for *_, c, _l, _w, (m0, _s0) in series:
        ax.plot([axes["w0"], axes["w1"]], [m0, m0], color=c, lw=2.6, solid_capstyle="butt",
                path_effects=halo, zorder=6)
    ax.plot([], [], color="0.35", lw=2.6, label="Mean over amplitude window")
    ax.set_xlabel("Time relative to stimulus onset (s)")
    ax.set_ylabel("ROI $\\Delta$R/R (%)")
    ax.set_xlim(axes["t_axis"].min() - 0.4, axes["t_axis"].max() + 0.95)
    all_vals = np.concatenate([
        np.concatenate([(m - s)[np.isfinite(m)], (m + s)[np.isfinite(m)]]) for m, s, *_ in series
    ])
    lo, hi = np.nanpercentile(all_vals, 1.0), np.nanpercentile(all_vals, 99.0)
    pad = 0.32 * (hi - lo) + 1e-3
    ax.set_ylim(lo - pad, hi + pad)
    y0, y1 = ax.get_ylim()
    if gap_visible:
        ax.text((axes["b_last"] + axes["p_first"]) / 2, y1 - 0.06 * (y1 - y0),
                "acquisition gap\n(gap frames plotted)", ha="center", va="top", fontsize=7,
                color="0.30", linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.75", lw=0.6, alpha=0.9))
    if not suppress_title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.6, loc="lower left", framealpha=0.92)
    ax.grid(alpha=0.15, lw=0.6)
    fig.tight_layout()
    if show_amplitude_labels:
        _draw_amplitude_labels(fig, ax, series, axes["w1"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ───────────────────────────── orchestration ────────────────────────────────

def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir")
    ap.add_argument("--stats-dir", type=str, default=None,
                     help="Folder with analysis_summary.json / roi_timecourse_raw.npz "
                          "(default: SESSION_DIR/session_stats).")
    ap.add_argument("--out-dir", type=str, default=None, help="Default: same as --stats-dir.")
    ap.add_argument("--label", type=str, default=None, help="Default: SESSION_DIR's own folder name.")
    ap.add_argument("--compare-session", type=str, default=None,
                     help="A second session's folder, measured in THIS session's ROI "
                          "(e.g. a no-stim control or interleaved catch arm).")
    ap.add_argument("--compare-stats-dir", type=str, default=None,
                     help="Default: COMPARE_SESSION/session_stats.")
    ap.add_argument("--compare-label", type=str, default="Comparison session")
    ap.add_argument("--um-per-px", type=float, default=DEFAULT_UM_PER_PX,
                     help=f"Spatial calibration for the scale bar, um per full-frame (1920 px) pixel. "
                          f"Default {DEFAULT_UM_PER_PX:.4f} (7.07 mm field / 1920 px) was measured "
                          "2026-09-03 against a caliper; re-measure if the objective, aperture, or "
                          "working distance change.")
    ap.add_argument("--no-scale-bar", action="store_true",
                     help="Omit the scale bar (e.g. when this rig's spatial calibration is unknown). "
                          "--um-per-px is ignored.")
    ap.add_argument("--anterior-side", type=str, default="left", choices=_SIDES)
    ap.add_argument("--medial-side", type=str, default="bottom", choices=_SIDES)
    ap.add_argument("--midline-centered", action="store_true",
                     help="Both hemispheres in frame with the midline down the centre: "
                          "label both edges perpendicular to --anterior-side as lateral "
                          "and mark medial at the compass centre. --medial-side is ignored.")
    ap.add_argument("--fixed-vmax", type=float, default=DEFAULT_FIXED_VMAX,
                     help="Shared scale for the _v05-style panel variants, for comparing sessions side by side.")
    ap.add_argument("--reuse-extraction-cache", action="store_true",
                     help="Skip out-region/compare-session raw-frame streaming if "
                          "<label>_extraction_cache.npz already exists in --out-dir.")
    ap.add_argument("--no-title", action="store_true",
                     help="Don't auto-generate the '<label>: leave-one-out ROI dR/R(t)' header "
                          "on <label>_timecourse.png.")
    ap.add_argument("--show-amplitude-labels", action="store_true",
                     help="Draw each series' numerical dR/R value (e.g. '-0.068%%') next to its "
                          "mean-over-amplitude-window line on <label>_timecourse.png.")
    args = ap.parse_args(argv)

    session_dir = Path(args.session_dir)
    stats_dir = Path(args.stats_dir) if args.stats_dir else session_dir / "session_stats"
    out_dir = Path(args.out_dir) if args.out_dir else stats_dir
    label = args.label or session_dir.name
    if args.midline_centered:
        orient = midline_centered_orientation_map(args.anterior_side)
    else:
        orient = orientation_map(args.anterior_side, args.medial_side)

    summary_path = stats_dir / "analysis_summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run Statistics on this session first.", file=sys.stderr)
        return 1
    summary = json.loads(summary_path.read_text())

    print(f"Session: {session_dir.name}  stats: {stats_dir}  label: {label}")
    print("Building masks (full-session ROI, 180-degree out-region, cluster)...")
    masks = build_masks(session_dir, summary)
    trial_ids = summary["usable_trial_ids"]

    print("Computing LOO ROI core (for the panel outline)...")
    restrict = bool(summary.get("roi_within_region", False))
    core_binned = loo_core_mask(masks["maps"], masks["binned_slice"], restrict)
    roi_core_full = place_binned_mask(core_binned, masks["byf"], masks["bxf"], masks["crop"])

    # The panel display uses a DIFFERENT mean map than masks["mean_map"]:
    # build_masks()'s mean_map is the raw, non-demeaned average, used only
    # for ROI-threshold selection and the out-region reflection -- never
    # meant for display. The panel display instead demeans each trial's full map by its own full-frame spatial
    # mean (removing thermal-drift/vasomotion DC offset -- the same
    # correction statistical_analyses.py's t-map applies) before averaging
    # across trials. Reproduce that here rather than reusing masks["mean_map"].
    stack_full = np.stack(masks["maps"], axis=0)
    stack_full = stack_full - stack_full.mean(axis=(1, 2), keepdims=True)
    demeaned_mean_map = stack_full.mean(axis=0)

    dmap_full = np.full((H, W), np.nan)
    up = np.repeat(np.repeat(100.0 * demeaned_mean_map, masks["byf"], axis=0), masks["bxf"], axis=1)
    cx0, cy0, cw, ch = masks["crop"]
    hh, ww = min(up.shape[0], H - cy0), min(up.shape[1], W - cx0)
    dmap_full[cy0:cy0 + hh, cx0:cx0 + ww] = up[:hh, :ww]

    print("Rendering green reference...")
    render_green_reference(session_dir, masks["region"], out_dir / f"{label}_green_reference.png")

    print("Rendering targeting reference (green reference + LOO ROI outline, for implant/injection targeting)...")
    render_targeting_reference(session_dir, masks["region"], roi_core_full,
                                out_dir / f"{label}_targeting_reference.png")

    scale_um_per_px = None if args.no_scale_bar else args.um_per_px
    print("Rendering cortical panels..." + (" (no scale bar)" if scale_um_per_px is None else ""))
    v1 = render_panel(dmap_full, roi_core_full, masks["cluster"], _green_display(session_dir),
                       masks["region"], out_dir / f"{label}_drr_panel.png",
                       vmax=None, outlines=True, out_region_mask=masks["out"],
                       um_per_px=scale_um_per_px, orient=orient)
    print(f"  {label}_drr_panel.png  scale +/-{v1:.4g}%")
    v2 = render_panel(dmap_full, roi_core_full, masks["cluster"], _green_display(session_dir),
                       masks["region"], out_dir / f"{label}_heat.png",
                       vmax=None, outlines=False, out_region_mask=None,
                       um_per_px=scale_um_per_px, orient=orient)
    print(f"  {label}_heat.png  scale +/-{v2:.4g}%")
    render_panel(dmap_full, roi_core_full, masks["cluster"], _green_display(session_dir),
                 masks["region"], out_dir / f"{label}_drr_panel_v05.png",
                 vmax=args.fixed_vmax, outlines=True, out_region_mask=masks["out"],
                 um_per_px=scale_um_per_px, orient=orient)
    render_panel(dmap_full, roi_core_full, masks["cluster"], _green_display(session_dir),
                 masks["region"], out_dir / f"{label}_heat_v05.png",
                 vmax=args.fixed_vmax, outlines=False, out_region_mask=None,
                 um_per_px=scale_um_per_px, orient=orient)
    print(f"  {label}_drr_panel_v05.png / {label}_heat_v05.png  scale +/-{args.fixed_vmax:g}%")

    cache_path = out_dir / f"{label}_extraction_cache.npz"
    out_extraction = None
    compare_extraction = None
    if args.reuse_extraction_cache and cache_path.exists():
        print(f"Reusing cached extraction from {cache_path} (skipping raw-frame streaming).")
        cached = np.load(cache_path, allow_pickle=True)
        out_extraction = dict(
            values=cached["out_values"],
            gap_values=list(cached["out_gap_values"]),
            gap_times=list(cached["out_gap_times"]),
            amps=cached["out_amps"],
        )
        if "compare_values" in cached:
            compare_extraction = dict(
                values=cached["compare_values"],
                gap_values=list(cached["compare_gap_values"]),
                gap_times=list(cached["compare_gap_times"]),
                amps=cached["compare_amps"],
                session_dir=Path(str(cached["compare_session_dir"])),
            )

    if out_extraction is None:
        print(f"Extracting out-region timecourse ({len(trial_ids)} trials, streaming raw frames)...")
        raw = extract_mask_timecourse(session_dir, trial_ids, masks["out"])
        axes = session_axes(session_dir, summary)
        gap_times = []
        for t in trial_ids:
            g_files = sorted(glob.glob(str(session_dir / f"trial_{t:03d}" / "gap" / "*.raw")))
            gap_times.append(np.array([_gap_frame_time(f, session_dir, t) for f in g_files]))
        out_extraction = dict(values=raw["values"], gap_values=raw["gap_values"], gap_times=gap_times, amps=raw["amps"])
        print(f"  out-region amp {raw['amps'].mean():+.4f}% +/- {raw['amps'].std(ddof=1) / np.sqrt(len(raw['amps'])):.4f}%")

    compare_session_dir = None
    if args.compare_session:
        compare_session_dir = Path(args.compare_session)
        compare_stats_dir = Path(args.compare_stats_dir) if args.compare_stats_dir else compare_session_dir / "session_stats"
        compare_summary_path = compare_stats_dir / "analysis_summary.json"
        if not compare_summary_path.exists():
            print(f"ERROR: {compare_summary_path} not found. Run Statistics on the compare session first.", file=sys.stderr)
            return 1
        compare_summary = json.loads(compare_summary_path.read_text())
        compare_trial_ids = compare_summary["usable_trial_ids"]
        if compare_extraction is None:
            print(f"Extracting compare-session ROI timecourse ({len(compare_trial_ids)} trials, streaming raw frames)...")
            raw = extract_mask_timecourse(compare_session_dir, compare_trial_ids, masks["roi"])
            gap_times = []
            for t in compare_trial_ids:
                g_files = sorted(glob.glob(str(compare_session_dir / f"trial_{t:03d}" / "gap" / "*.raw")))
                gap_times.append(np.array([_gap_frame_time(f, compare_session_dir, t) for f in g_files]))
            compare_extraction = dict(values=raw["values"], gap_values=raw["gap_values"], gap_times=gap_times,
                                       amps=raw["amps"], session_dir=compare_session_dir)
            print(f"  compare-session ROI amp {raw['amps'].mean():+.4f}% +/- {raw['amps'].std(ddof=1) / np.sqrt(len(raw['amps'])):.4f}%")

    save_kwargs = dict(
        out_values=out_extraction["values"], out_amps=out_extraction["amps"],
        out_gap_values=_ragged_object_array(out_extraction["gap_values"]),
        out_gap_times=_ragged_object_array(out_extraction["gap_times"]),
    )
    if compare_extraction is not None:
        save_kwargs.update(
            compare_values=compare_extraction["values"], compare_amps=compare_extraction["amps"],
            compare_gap_values=_ragged_object_array(compare_extraction["gap_values"]),
            compare_gap_times=_ragged_object_array(compare_extraction["gap_times"]),
            compare_session_dir=str(compare_extraction["session_dir"]),
        )
    np.savez(cache_path, **save_kwargs)

    print("Rendering timecourse...")
    timecourse_raw = dict(np.load(stats_dir / "roi_timecourse_raw.npz"))
    amps_csv = list(csv.DictReader(open(stats_dir / "loo_roi_amplitude_summary.csv")))
    amp_key = "peak_drr_percent" if "peak_drr_percent" in amps_csv[0] else "mean_drr_percent"
    summary["_amps"] = np.array([float(r[amp_key]) for r in amps_csv])
    render_timecourse(
        session_dir, summary, timecourse_raw, out_extraction, out_dir / f"{label}_timecourse.png",
        title=f"{label}: leave-one-out ROI dR/R(t)",
        compare_series=compare_extraction, compare_label=args.compare_label,
        suppress_title=args.no_title, show_amplitude_labels=args.show_amplitude_labels,
    )

    print("Done ->", out_dir)
    return 0


def _green_display(session_dir: Path) -> np.ndarray:
    """Cached per-process; the same green mean is used for both the
    standalone green-reference figure (full contrast stretch) and the
    faint underlay in the cortical panels."""
    if not hasattr(_green_display, "_cache"):
        _green_display._cache = {}
    key = str(session_dir)
    if key not in _green_display._cache:
        g = green_reference_mean(session_dir)
        lo, hi = np.percentile(g, [1, 99])
        _green_display._cache[key] = np.clip((g - lo) / (hi - lo + 1e-9), 0, 1)
    return _green_display._cache[key]


def _gap_frame_time(raw_path: str, session_dir: Path, trial: int) -> float:
    """Look up one gap frame's timestamp (relative to that trial's
    STIM_START) from session_log.csv / marker_log.csv, by filename."""
    fname = Path(raw_path).name
    if not hasattr(_gap_frame_time, "_cache"):
        _gap_frame_time._cache = {}
    cache_key = str(session_dir)
    if cache_key not in _gap_frame_time._cache:
        by_trial: dict = {}
        with open(session_dir / "session_log.csv", newline="", encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                if row.get("phase") != "gap":
                    continue
                try:
                    t = int(row["trial_index"])
                except (KeyError, ValueError):
                    continue
                fn = row["filename"].strip()
                if fn:
                    by_trial.setdefault(t, {})[fn] = row["host_timestamp_iso"]
        stim_start: dict = {}
        with open(session_dir / "marker_log.csv", newline="", encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                if row["marker"] == "STIM_START":
                    stim_start[int(row["trial_index"])] = row["host_timestamp_iso"]
        _gap_frame_time._cache[cache_key] = (by_trial, stim_start)
    by_trial, stim_start = _gap_frame_time._cache[cache_key]
    ts = by_trial.get(trial, {}).get(fname)
    t0 = stim_start.get(trial)
    if ts is None or t0 is None:
        return float("nan")
    return parse_ts(ts) - parse_ts(t0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
