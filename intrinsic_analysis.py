#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Offline intrinsic imaging processor for saved trial folders.

This script applies the offline version of the intrinsic_imaging analysis logic:
    optional session-folder picker
    green reference from the middle/inner green frames
    interactive software ROI selection from the averaged green reference
    optional dark-frame subtraction before ROI/binning/analysis
    optional software analysis binning
    raw pixel-count differencing or fractional-reflectance ΔR/R with denominator floor/masking
    optional median filtering for ΔR/R, median centering, Gaussian filtering
    TIFF scaling = selectable: signed symmetric or min-max 0 to 65535

It can process either:
    1) A session folder containing trial_001/, trial_002/, ... plus optional session_green_reference/
    2) A single trial folder containing baseline/ and post/ directories
    3) A parent folder containing multiple trial-like subfolders

Example with RAW files:
    py -3.10 intrinsic_analysis.py --folder .\\captures\\session_YYYYMMDD_HHMMSS --width 1440 --height 1080 --pixel-format Mono16

Optional ROI:
    py -3.10 intrinsic_analysis.py --roi-config selected_roi.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def _try_import_tkinter():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        return tk, filedialog, messagebox
    except Exception:
        return None, None, None


# Analysis-method identifiers. "fractional" is accepted as a shorthand for
# "fractional_reflectance". Used as argparse `type=`, which runs before
# `choices=`, so --help advertises only the canonical names.
_ANALYSIS_METHOD_ALIASES = {
    "fractional": "fractional_reflectance",
}


def normalize_analysis_method(value: str) -> str:
    key = str(value).strip().lower()
    return _ANALYSIS_METHOD_ALIASES.get(key, key)


@dataclass
class ProcessingConfig:
    folder: Optional[Path] = None
    output_name: str = "offline_analysis"
    green_dir: Optional[Path] = None
    roi_config: Optional[Path] = None
    # Fallback RAW dimensions. Used only when auto_frame_dims is False or when no
    # trial_metadata.json is found. The live pipeline derives these from the camera
    # at capture time, so prefer auto-detection over these defaults.
    width: Optional[int] = 1920
    height: Optional[int] = 1200
    pixel_format: str = "Mono16"
    auto_frame_dims: bool = True
    # Defaults aligned to intrinsic_imaging.TrialConfig (start=5, end=35) so an
    # offline re-run reproduces the live post-analysis window by default.
    analysis_start_frame: int = 5
    analysis_end_frame: Optional[int] = 35  # exclusive, <=0 means all remaining frames
    smoothing_sigma: float = 5.0
    analysis_binning: int = 1
    analysis_method: str = "raw_counts"
    dark_reference_path: Optional[Path] = None
    denominator_floor_percentile: float = 5.0
    denominator_floor_counts: float = 100.0
    median_filter_size: int = 3
    rescale_bit_depth: int = 16
    rescale_mode: str = "signed_symmetric"
    # Mirror live display inversion: for raw_counts, optionally plot reflectance
    # decreases as positive/hot. Affects display PNGs only, not quantitative arrays.
    invert_display_signal: bool = False
    save_arrays: bool = True
    make_overlay: bool = True
    overlay_cmap: str = "seismic"
    overlay_alpha_mode: str = "magnitude"
    overlay_alpha: float = 0.55
    overlay_threshold_frac: float = 0.0
    recursive: bool = True
    interactive_roi: bool = True
    # Analysis green reference now mirrors intrinsic_imaging: trim this many frames
    # from BOTH ends of the green stack and average the rest.
    green_reference_trim_frames: int = 5
    # ROI-picker-only: number of middle green frames averaged for the interactive
    # ROI reference image. Does not affect the quantitative analysis reference.
    green_inner_frames: int = 20


class OfflineIntrinsicProcessor:
    def __init__(self, cfg: ProcessingConfig):
        self.cfg = cfg
        self.active_roi = self._load_roi_config(cfg.roi_config)

    @staticmethod
    def _load_roi_config(path: Optional[Path]) -> Optional[dict]:
        if path is None:
            return None
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ROI config not found: {path}")
        with open(path, "r", encoding="utf-8") as fp:
            roi = json.load(fp)
        if not bool(roi.get("enabled", True)):
            return None
        for key in ("x", "y", "width", "height"):
            if key not in roi:
                raise RuntimeError(f"ROI config missing required key: {key}")
        return {
            "x": int(roi["x"]),
            "y": int(roi["y"]),
            "width": int(roi["width"]),
            "height": int(roi["height"]),
            "source_path": str(path),
        }

    def _analysis_roi_bounds(self, image_shape: tuple[int, int]) -> Optional[tuple[int, int, int, int]]:
        if self.active_roi is None:
            return None
        height, width = int(image_shape[0]), int(image_shape[1])
        x0 = max(0, min(int(self.active_roi["x"]), width - 1))
        y0 = max(0, min(int(self.active_roi["y"]), height - 1))
        x1 = max(x0 + 1, min(x0 + int(self.active_roi["width"]), width))
        y1 = max(y0 + 1, min(y0 + int(self.active_roi["height"]), height))
        return y0, y1, x0, x1

    def _crop_to_roi(self, arr: np.ndarray) -> np.ndarray:
        bounds = self._analysis_roi_bounds(arr.shape[-2:])
        if bounds is None:
            return arr
        y0, y1, x0, x1 = bounds
        if arr.ndim == 2:
            return arr[y0:y1, x0:x1]
        return arr[..., y0:y1, x0:x1]

    def _load_image(self, path: Path) -> np.ndarray:
        ext = path.suffix.lower()
        if ext == ".raw":
            if self.cfg.width is None or self.cfg.height is None:
                raise RuntimeError(
                    "RAW files require --width and --height because RAW has no header."
                )
            dtype_by_format = {
                "Mono8": np.uint8,
                "Mono12": np.uint16,
                "Mono16": np.uint16,
            }
            dtype = dtype_by_format.get(self.cfg.pixel_format)
            if dtype is None:
                raise RuntimeError(f"Unsupported pixel format for RAW: {self.cfg.pixel_format}")
            data = np.fromfile(path, dtype=dtype)
            expected = int(self.cfg.width) * int(self.cfg.height)
            if data.size != expected:
                raise RuntimeError(
                    f"RAW frame has {data.size} pixels, expected {expected}. File: {path}"
                )
            return data.reshape((int(self.cfg.height), int(self.cfg.width))).astype(np.float64)

        from PIL import Image
        return np.array(Image.open(path)).astype(np.float64)

    def _load_stack(self, folder: Path) -> np.ndarray:
        patterns = ["*.tiff", "*.tif", "*.png", "*.raw"]
        files: list[Path] = []
        for pattern in patterns:
            files.extend(sorted(folder.glob(pattern)))
        files = sorted(files)
        if not files:
            raise RuntimeError(f"No image files found in {folder}")
        return np.stack([self._load_image(f) for f in files], axis=0)

    @staticmethod
    def _display_scale(img: np.ndarray) -> np.ndarray:
        finite = np.isfinite(img)
        if not np.any(finite):
            return np.zeros_like(img, dtype=np.float32)
        lo, hi = np.nanpercentile(img[finite], [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(img[finite])), float(np.nanmax(img[finite]))
        if hi <= lo:
            return np.zeros_like(img, dtype=np.float32)
        return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)

    @staticmethod
    def _signed_overlay_rgba(
        signal: np.ndarray,
        vmax: float,
        cmap_name: str,
        alpha_mode: str = "magnitude",
        flat_alpha: float = 0.55,
        threshold_frac: float = 0.0,
    ) -> np.ndarray:
        """Map a signed, zero-centered signal to an RGBA overlay.

        Color encodes sign/magnitude through a diverging colormap (zero -> mid).
        Alpha controls how much anatomy shows through:
            "flat":      constant flat_alpha on every finite pixel.
            "magnitude": alpha scales with |signal|/vmax, so near-zero pixels are
                         transparent (grey anatomy shows) and strong pixels are
                         opaque (color pops). This is what keeps both layers legible.
        threshold_frac: pixels with |signal| < threshold_frac*vmax are forced fully
                        transparent, giving clean activation islands instead of haze.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
        finite = np.isfinite(signal)
        filled = np.where(finite, signal, 0.0)

        cmap = plt.get_cmap(cmap_name)
        norm = Normalize(vmin=-vmax, vmax=vmax)
        rgba = cmap(norm(filled))  # H x W x 4, float in [0, 1]

        if str(alpha_mode).lower() == "flat":
            alpha = np.full(signal.shape, float(flat_alpha), dtype=np.float64)
        else:
            alpha = np.clip(np.abs(filled) / vmax, 0.0, 1.0)

        if threshold_frac and threshold_frac > 0:
            alpha = np.where(np.abs(filled) < float(threshold_frac) * vmax, 0.0, alpha)

        rgba[..., 3] = np.where(finite, alpha, 0.0)
        return rgba

    def _save_signal_overlay_png(
        self,
        reference: np.ndarray,
        signal: np.ndarray,
        vmax: float,
        out_path: Path,
        title: str,
        colorbar_label: str,
    ) -> None:
        """Grey anatomy underneath, recolored signed signal on top, with a
        colorbar driven by the same diverging scale."""
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
        rgba = self._signed_overlay_rgba(
            signal,
            vmax,
            self.cfg.overlay_cmap,
            alpha_mode=self.cfg.overlay_alpha_mode,
            flat_alpha=self.cfg.overlay_alpha,
            threshold_frac=self.cfg.overlay_threshold_frac,
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self._display_scale(reference), cmap="gray")
        ax.imshow(rgba)  # RGBA already carries per-pixel alpha
        ax.set_title(title)
        ax.axis("off")

        sm = ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax), cmap=self.cfg.overlay_cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    @staticmethod
    def _rescale_signed_symmetric_to_bit_depth(img: np.ndarray, bit_depth: int = 16) -> np.ndarray:
        """Scale a signed map to unsigned integer range using zero as mid-gray.

        Mapping:
            -max_abs -> 0
             0       -> midpoint
            +max_abs -> 2^bit_depth - 1
        """
        finite = np.isfinite(img)
        max_value = int((2 ** int(bit_depth)) - 1)
        dtype = np.uint16 if int(bit_depth) > 8 else np.uint8
        if not np.any(finite):
            return np.zeros_like(img, dtype=dtype)
        max_abs = float(np.nanmax(np.abs(img[finite])))
        if not np.isfinite(max_abs) or max_abs <= 0:
            return np.full_like(img, int(round(max_value / 2.0)), dtype=dtype)
        scaled = np.clip(img / max_abs, -1.0, 1.0)
        unsigned = (scaled + 1.0) * (max_value / 2.0)
        return np.clip(unsigned, 0, max_value).astype(dtype)

    @staticmethod
    def _rescale_minmax_to_bit_depth(img: np.ndarray, bit_depth: int = 16) -> np.ndarray:
        """Scale a map to unsigned integer range using min-max normalization.

        Mapping:
            finite minimum -> 0
            finite maximum -> 2^bit_depth - 1

        Note: zero does not necessarily map to mid-gray with this mode.
        """
        finite = np.isfinite(img)
        max_value = int((2 ** int(bit_depth)) - 1)
        dtype = np.uint16 if int(bit_depth) > 8 else np.uint8
        if not np.any(finite):
            return np.zeros_like(img, dtype=dtype)
        lo = float(np.nanmin(img[finite]))
        hi = float(np.nanmax(img[finite]))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(img, dtype=dtype)
        scaled = (img - lo) / (hi - lo)
        unsigned = scaled * max_value
        return np.clip(unsigned, 0, max_value).astype(dtype)

    @classmethod
    def _rescale_to_bit_depth(cls, img: np.ndarray, bit_depth: int = 16, mode: str = "signed_symmetric") -> np.ndarray:
        """Scale a floating-point analysis map to an unsigned integer image.

        Args:
            img: Floating-point map to export.
            bit_depth: Output bit depth, usually 16.
            mode:
                "signed_symmetric" keeps zero at mid-gray.
                "minmax" maps the finite minimum to 0 and finite maximum to max.
        """
        mode = str(mode).lower().strip().replace("-", "_")
        if mode in {"signed", "symmetric", "signed_symmetric", "symmetric_signed"}:
            return cls._rescale_signed_symmetric_to_bit_depth(img, bit_depth)
        if mode in {"minmax", "min_max", "min-max"}:
            return cls._rescale_minmax_to_bit_depth(img, bit_depth)
        raise ValueError(f"Unsupported rescale mode: {mode!r}. Use 'signed_symmetric' or 'minmax'.")

    # IMX249: 12-bit ADC left-shifted into a 16-bit Mono16 container. Clipping
    # happens at 65520 (4095 << 4), NOT 65535. Comparing against the dtype max
    # never fires, so saturation must be checked against this value instead.
    SENSOR_SATURATION_COUNT = 65520.0

    @staticmethod
    def _saturated_fraction(stack: np.ndarray, saturation_count: float = SENSOR_SATURATION_COUNT) -> float:
        arr = np.asarray(stack)
        return float(np.mean(arr >= float(saturation_count))) if arr.size else 0.0

    @staticmethod
    def _stack_qc_stats(
        stack: np.ndarray,
        phase: str,
        bit_depth: int = 16,
        saturation_count: float = SENSOR_SATURATION_COUNT,
    ) -> dict:
        """Return basic image-quality statistics for a frame stack.

        saturation_count is the true sensor clip level in raw ADC counts (65520
        for the IMX249 in Mono16), not the container dtype max. The saturated
        fraction reported here is computed on whatever stack is passed in; the
        caller should override it with a fraction measured on the RAW frames
        before dark subtraction / ROI crop / binning to match the live pipeline.
        """
        finite = np.isfinite(stack)
        values = stack[finite]
        dtype_max = float((2 ** bit_depth) - 1)
        if np.issubdtype(np.asarray(stack).dtype, np.integer):
            try:
                dtype_max = float(np.iinfo(np.asarray(stack).dtype).max)
            except ValueError:
                pass
        sat_threshold = float(saturation_count)
        if values.size == 0:
            return {"phase": phase, "frames": int(stack.shape[0]), "min": None, "p01": None, "p50": None, "p95": None, "p99": None, "max": None, "mean": None, "dtype_max": dtype_max, "saturation_threshold": sat_threshold, "saturated_fraction": None}
        return {
            "phase": phase,
            "frames": int(stack.shape[0]),
            "min": float(np.nanmin(values)),
            "p01": float(np.nanpercentile(values, 1)),
            "p50": float(np.nanpercentile(values, 50)),
            "p95": float(np.nanpercentile(values, 95)),
            "p99": float(np.nanpercentile(values, 99)),
            "max": float(np.nanmax(values)),
            "mean": float(np.nanmean(values)),
            "dtype_max": dtype_max,
            "saturation_threshold": sat_threshold,
            "saturated_fraction": float(np.mean(stack >= sat_threshold)),
        }

    def _find_trial_folders(self, selected_folder: Path) -> list[Path]:
        selected_folder = Path(selected_folder)
        if (selected_folder / "baseline").is_dir() and (selected_folder / "post").is_dir():
            return [selected_folder]

        direct_trials = [
            p for p in sorted(selected_folder.iterdir())
            if p.is_dir() and (p / "baseline").is_dir() and (p / "post").is_dir()
        ]
        if direct_trials:
            return direct_trials

        if self.cfg.recursive:
            recursive_trials = [
                p for p in sorted(selected_folder.rglob("*"))
                if p.is_dir() and (p / "baseline").is_dir() and (p / "post").is_dir()
            ]
            # Avoid duplicates from nested weirdness.
            unique: list[Path] = []
            seen = set()
            for p in recursive_trials:
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    unique.append(p)
            return unique

        return []

    def _find_green_dir_for_trial(self, selected_folder: Path, trial_folder: Path) -> Optional[Path]:
        if self.cfg.green_dir is not None:
            return Path(self.cfg.green_dir)
        candidates = [
            selected_folder / "session_green_reference",
            trial_folder / "green",
            trial_folder / ".." / "session_green_reference",
            trial_folder.parent / "session_green_reference",
        ]
        for c in candidates:
            c = c.resolve()
            if c.is_dir() and any(c.glob("*.*")):
                return c
        return None

    def _choose_middle_frame_window(self, n_frames: int, desired_count: int) -> tuple[int, int]:
        """Return 0-indexed [start, end) bounds for the middle desired_count frames."""
        n_frames = int(n_frames)
        desired_count = int(desired_count)
        if n_frames <= 0:
            return 0, 0
        if desired_count <= 0 or desired_count >= n_frames:
            return 0, n_frames
        start = max(0, (n_frames - desired_count) // 2)
        end = min(n_frames, start + desired_count)
        return start, end

    def _green_reference_from_stack(self, green_stack: np.ndarray) -> np.ndarray:
        """Build the analysis green reference the same way intrinsic_imaging does:
        trim green_reference_trim_frames from BOTH ends, then average the rest.
        Falls back to averaging the whole stack if it is too short to trim."""
        trim = max(0, int(self.cfg.green_reference_trim_frames))
        if trim > 0 and green_stack.shape[0] > 2 * trim:
            green_for_reference = green_stack[trim:-trim]
        else:
            green_for_reference = green_stack
        return np.nanmean(green_for_reference, axis=0)

    @staticmethod
    def _frame_dims_from_metadata(trial_folder: Path) -> Optional[dict]:
        """Read frame width/height/pixel_format from the live pipeline's
        trial_<NNN>/meta/trial_metadata.json, if present."""
        meta_path = Path(trial_folder) / "meta" / "trial_metadata.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
        except Exception:
            return None
        width = meta.get("image_width_px")
        height = meta.get("image_height_px")
        pixel_format = meta.get("pixel_format")
        if width is None or height is None:
            return None
        return {"width": int(width), "height": int(height), "pixel_format": pixel_format}

    def _find_session_green_dir(self, selected_folder: Path) -> Optional[Path]:
        """Find a green reference directory to use for interactive ROI selection."""
        if self.cfg.green_dir is not None:
            green_dir = Path(self.cfg.green_dir)
            return green_dir if green_dir.is_dir() else None
        candidates = [
            selected_folder / "session_green_reference",
            selected_folder / "green",
        ]
        for c in candidates:
            c = c.resolve()
            if c.is_dir() and any(c.glob("*.*")):
                return c
        return None

    def _make_green_roi_reference(self, green_dir: Path) -> tuple[np.ndarray, dict]:
        """Load green frames and average the middle N frames for ROI selection."""
        green_stack = self._load_stack(green_dir)
        start, end = self._choose_middle_frame_window(green_stack.shape[0], self.cfg.green_inner_frames)
        if end <= start:
            raise RuntimeError(f"No green frames available for ROI selection in {green_dir}")
        green_reference = np.nanmean(green_stack[start:end], axis=0)
        info = {
            "green_dir": str(green_dir),
            "green_frames_available": int(green_stack.shape[0]),
            "green_inner_frames_requested": int(self.cfg.green_inner_frames),
            "green_frames_used_start_1_indexed": int(start + 1),
            "green_frames_used_end_1_indexed_inclusive": int(end),
            "reference_shape_yx": [int(green_reference.shape[0]), int(green_reference.shape[1])],
        }
        return green_reference, info

    def _select_roi_interactively(self, reference_image: np.ndarray) -> Optional[dict]:
        """Open a matplotlib window and let the user draw a rectangular ROI."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import RectangleSelector
        except Exception as exc:
            print(f"Warning: could not open ROI selector because matplotlib is unavailable: {exc}")
            return None

        roi: dict[str, int] = {}
        display = self._display_scale(reference_image)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(display, cmap="gray")
        ax.set_title("Draw ROI on averaged inner green reference, then close this window")
        ax.set_xlabel("Click-drag to select; resize if needed; close window to continue")

        def onselect(eclick, erelease):
            x0, y0 = int(round(eclick.xdata)), int(round(eclick.ydata))
            x1, y1 = int(round(erelease.xdata)), int(round(erelease.ydata))
            x = max(0, min(x0, x1))
            y = max(0, min(y0, y1))
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            if w > 0 and h > 0:
                roi.clear()
                roi.update({"enabled": True, "x": x, "y": y, "width": w, "height": h})

        selector = RectangleSelector(
            ax,
            onselect,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        plt.show()

        # Keep a reference alive until after plt.show() returns.
        _ = selector

        if not roi:
            print("No ROI selected; proceeding with full-frame analysis.")
            return None
        return roi

    def process_folder(self, selected_folder: Path) -> Path:
        selected_folder = Path(selected_folder)
        if not selected_folder.exists():
            raise FileNotFoundError(f"Folder not found: {selected_folder}")

        trial_folders = self._find_trial_folders(selected_folder)
        if not trial_folders:
            raise RuntimeError(
                "No trial folders found. Expected a folder containing baseline/ and post/, "
                "or a session folder containing trial_###/baseline and trial_###/post."
            )

        session_out = selected_folder / self.cfg.output_name
        session_out.mkdir(parents=True, exist_ok=True)

        if self.active_roi is None and self.cfg.interactive_roi:
            green_dir = self._find_session_green_dir(selected_folder)
            if green_dir is not None:
                try:
                    green_reference, green_info = self._make_green_roi_reference(green_dir)
                    selected_roi = self._select_roi_interactively(green_reference)
                    if selected_roi is not None:
                        selected_roi.update({
                            "source_path": str(session_out / "selected_roi_from_green.json"),
                            "selection_reference": green_info,
                        })
                        self.active_roi = selected_roi
                        with open(session_out / "selected_roi_from_green.json", "w", encoding="utf-8") as fp:
                            json.dump(selected_roi, fp, indent=2)
                        print(f"Using selected ROI: x={selected_roi['x']}, y={selected_roi['y']}, width={selected_roi['width']}, height={selected_roi['height']}")
                except Exception as exc:
                    print(f"Warning: interactive green ROI selection failed; proceeding full-frame. Reason: {exc}")
            else:
                print("No green reference directory found for ROI selection; proceeding full-frame.")

        trial_results: list[dict] = []
        trial_maps: list[np.ndarray] = []
        display_maps: list[np.ndarray] = []
        green_refs: list[np.ndarray] = []
        reference_images: list[np.ndarray] = []

        for i, trial_folder in enumerate(trial_folders, start=1):
            print(f"Processing trial {i}/{len(trial_folders)}: {trial_folder}")
            result = self.process_trial(selected_folder, trial_folder, session_out, i)
            trial_results.append(result["summary"])
            trial_maps.append(result["smoothed_map"])
            display_maps.append(result["display_signal"])
            if result["green_reference"] is not None:
                green_refs.append(result["green_reference"])
            reference_images.append(result["reference_image"])

            # Save the same style of post-trial review panel used by the live
            # acquisition script: current trial, running average through this
            # trial, and anatomical/reference image.
            self._save_trial_running_average_panel(
                session_out=session_out,
                trial_summaries=trial_results,
                display_maps=display_maps,
                reference_images=reference_images,
                current_trial=i,
            )

        self._save_session_average(session_out, trial_results, trial_maps, display_maps, green_refs)
        return session_out


    def _load_dark_reference_for_shape(self, image_shape: tuple[int, int]) -> Optional[np.ndarray]:
        if self.cfg.dark_reference_path is None:
            return None
        dark_path = Path(self.cfg.dark_reference_path)
        if not dark_path.exists():
            raise FileNotFoundError(f"Dark reference file not found: {dark_path}")
        if dark_path.suffix.lower() == ".npy":
            dark = np.load(dark_path).astype(np.float64)
        else:
            from PIL import Image
            dark = np.array(Image.open(dark_path)).astype(np.float64)
        if dark.ndim != 2:
            raise RuntimeError(f"Dark reference must be a 2D image; got shape {dark.shape}")
        if tuple(dark.shape) != tuple(image_shape):
            raise RuntimeError(
                f"Dark reference shape {dark.shape} does not match frame shape {image_shape}. "
                "Use a dark reference acquired with the same resolution/pixel format/exposure."
            )
        return dark

    @staticmethod
    def _software_bin_spatial(arr: np.ndarray, factor: int) -> np.ndarray:
        factor = max(1, int(factor))
        if factor <= 1:
            return arr
        if arr.ndim < 2:
            return arr
        height = int(arr.shape[-2])
        width = int(arr.shape[-1])
        binned_height = height // factor
        binned_width = width // factor
        if binned_height < 1 or binned_width < 1:
            raise RuntimeError(f"Analysis binning factor {factor} is too large for image shape {arr.shape}.")
        cropped = arr[..., :binned_height * factor, :binned_width * factor]
        if arr.ndim == 2:
            return cropped.reshape(binned_height, factor, binned_width, factor).mean(axis=(1, 3))
        leading = cropped.shape[:-2]
        reshaped = cropped.reshape(*leading, binned_height, factor, binned_width, factor)
        return reshaped.mean(axis=(-3, -1))

    def process_trial(self, selected_folder: Path, trial_folder: Path, session_out: Path, trial_number: int) -> dict:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        try:
            from scipy.ndimage import gaussian_filter, median_filter
        except ImportError as exc:
            raise RuntimeError("scipy is required for Gaussian/median filtering. Install with: py -3.10 -m pip install scipy") from exc

        baseline_dir = trial_folder / "baseline"
        post_dir = trial_folder / "post"
        green_dir = self._find_green_dir_for_trial(selected_folder, trial_folder)

        # Prefer frame dimensions recorded by the live pipeline over CLI defaults.
        # This is the single biggest RAW footgun: camera binning shrinks the saved
        # frame, and a stale 1920x1200 default would mis-reshape or error.
        if self.cfg.auto_frame_dims:
            dims = self._frame_dims_from_metadata(trial_folder)
            if dims is not None:
                changed = (self.cfg.width, self.cfg.height) != (dims["width"], dims["height"])
                self.cfg.width = dims["width"]
                self.cfg.height = dims["height"]
                if dims.get("pixel_format"):
                    if self.cfg.pixel_format != dims["pixel_format"]:
                        changed = True
                    self.cfg.pixel_format = dims["pixel_format"]
                if changed:
                    print(
                        f"Auto-detected frame dims from trial_metadata.json: "
                        f"{self.cfg.width}x{self.cfg.height} {self.cfg.pixel_format}"
                    )

        green_stack_full = self._load_stack(green_dir) if green_dir is not None else None
        baseline_stack_full = self._load_stack(baseline_dir)
        post_stack_full = self._load_stack(post_dir)
        full_frame_shape = tuple(int(v) for v in baseline_stack_full.shape[-2:])

        # Measure saturation on the RAW frames BEFORE dark subtraction, ROI crop, and
        # binning, exactly like intrinsic_imaging. Doing this after processing (as the
        # old code did) plus the wrong 65535 threshold meant saturation never registered.
        raw_saturation = {
            "green": self._saturated_fraction(green_stack_full) if green_stack_full is not None else None,
            "baseline": self._saturated_fraction(baseline_stack_full),
            "post_all": self._saturated_fraction(post_stack_full),
        }

        dark_reference = self._load_dark_reference_for_shape(full_frame_shape)
        dark_reference_applied = dark_reference is not None
        if dark_reference_applied:
            if green_stack_full is not None:
                green_stack_full = green_stack_full - dark_reference
            baseline_stack_full = baseline_stack_full - dark_reference
            post_stack_full = post_stack_full - dark_reference
            print(f"Applied dark-frame subtraction using: {self.cfg.dark_reference_path}")

        # Same order as intrinsic_imaging: dark subtraction -> ROI crop -> software binning -> analysis.
        baseline_stack = self._crop_to_roi(baseline_stack_full)
        post_stack = self._crop_to_roi(post_stack_full)
        green_stack = self._crop_to_roi(green_stack_full) if green_stack_full is not None else None
        analysis_frame_shape_pre_binning = tuple(int(v) for v in baseline_stack.shape[-2:])

        analysis_binning = max(1, int(self.cfg.analysis_binning))
        if analysis_binning > 1:
            baseline_stack = self._software_bin_spatial(baseline_stack, analysis_binning)
            post_stack = self._software_bin_spatial(post_stack, analysis_binning)
            if green_stack is not None:
                green_stack = self._software_bin_spatial(green_stack, analysis_binning)
            print(
                f"Applied {analysis_binning}x{analysis_binning} software binning for analysis only; "
                "saved camera frames are unchanged."
            )
        analysis_frame_shape = tuple(int(v) for v in baseline_stack.shape[-2:])

        green_reference = None
        if green_stack is not None:
            try:
                green_reference = self._green_reference_from_stack(green_stack)
            except Exception as exc:
                print(f"Warning: could not build green reference for {trial_folder}: {exc}")

        baseline_mean = np.nanmean(baseline_stack, axis=0)
        start = max(0, int(self.cfg.analysis_start_frame))
        end = self.cfg.analysis_end_frame
        if end is None or end <= 0 or end > post_stack.shape[0]:
            end = post_stack.shape[0]
        end = int(end)
        if start >= end:
            start = 0
            end = post_stack.shape[0]
        post_window = post_stack[start:end]
        post_mean = np.nanmean(post_window, axis=0)

        bit_depth = int(self.cfg.rescale_bit_depth)
        qc_rows = []
        if green_stack is not None:
            qc_rows.append(self._stack_qc_stats(green_stack, "green", bit_depth))
        qc_rows.extend([
            self._stack_qc_stats(baseline_stack, "baseline", bit_depth),
            self._stack_qc_stats(post_stack, "post_all", bit_depth),
            self._stack_qc_stats(post_window, "post_analysis_window", bit_depth),
        ])

        # Replace the (processed-stack) saturated fraction with the RAW-frame value
        # measured before dark/ROI/binning. The analysis window is a subset of post,
        # so it inherits the post_all raw fraction, matching intrinsic_imaging.
        raw_saturation_by_phase = {
            "green": raw_saturation["green"],
            "baseline": raw_saturation["baseline"],
            "post_all": raw_saturation["post_all"],
            "post_analysis_window": raw_saturation["post_all"],
        }
        for row in qc_rows:
            if row["phase"] in raw_saturation_by_phase:
                row["saturated_fraction"] = raw_saturation_by_phase[row["phase"]]

        analysis_method = self.cfg.analysis_method.lower().strip()
        analysis_mask = np.isfinite(baseline_mean)
        denominator_floor = None
        denominator_floor_percentile = None
        denominator_floor_counts = None
        denominator_mask_fraction_kept = None
        median_filter_size = 0
        median_filter_applied = False

        if analysis_method == "raw_counts":
            raw_diff = post_mean - baseline_mean
            median_reference = float(np.nanmedian(raw_diff[analysis_mask])) if np.any(analysis_mask) else float(np.nanmedian(raw_diff))
            centered_map = raw_diff - median_reference
            quantitative_label = "stim_minus_baseline_median_centered"
            colorbar_label = "post - baseline, median-centered"
            display_label = colorbar_label
        elif analysis_method == "fractional_reflectance":
            finite_baseline = np.isfinite(baseline_mean)
            positive_baseline = baseline_mean[finite_baseline & (baseline_mean > 0)]
            denominator_floor_percentile = float(self.cfg.denominator_floor_percentile)
            denominator_floor_counts = float(self.cfg.denominator_floor_counts)
            if positive_baseline.size > 0:
                percentile_floor = float(np.nanpercentile(positive_baseline, denominator_floor_percentile))
            else:
                percentile_floor = float("nan")
            denominator_floor = denominator_floor_counts if not np.isfinite(percentile_floor) else max(percentile_floor, denominator_floor_counts)
            safe_baseline = baseline_mean.copy()
            denominator_mask = finite_baseline & (safe_baseline >= denominator_floor)
            denominator_mask_fraction_kept = float(np.mean(denominator_mask)) if denominator_mask.size else None
            safe_baseline[~denominator_mask] = np.nan

            raw_diff = (post_mean - baseline_mean) / safe_baseline

            median_filter_size = int(self.cfg.median_filter_size)
            if median_filter_size > 1:
                if median_filter_size % 2 == 0:
                    median_filter_size += 1
                finite_raw = np.isfinite(raw_diff)
                if np.any(finite_raw):
                    raw_fill_value = float(np.nanmedian(raw_diff[finite_raw]))
                    raw_for_filter = np.where(finite_raw, raw_diff, raw_fill_value)
                    raw_filtered = median_filter(raw_for_filter, size=median_filter_size)
                    raw_diff = np.where(finite_raw, raw_filtered, np.nan)
                    median_filter_applied = True
            else:
                median_filter_size = 0

            drr_analysis_mask = analysis_mask & np.isfinite(raw_diff)
            median_reference = float(np.nanmedian(raw_diff[drr_analysis_mask])) if np.any(drr_analysis_mask) else float(np.nanmedian(raw_diff))
            centered_map = raw_diff - median_reference
            quantitative_label = "delta_r_over_r_median_centered"
            colorbar_label = "ΔR/R, median-centered"
            display_label = colorbar_label
        else:
            raise RuntimeError(f"Unsupported analysis_method: {self.cfg.analysis_method}")

        smoothed_map = gaussian_filter(centered_map, sigma=float(self.cfg.smoothing_sigma)) if self.cfg.smoothing_sigma > 0 else centered_map
        masked_map = smoothed_map

        # Mirror intrinsic_imaging: for raw_counts, optionally invert the DISPLAY signal
        # so reflectance decreases plot as positive/hot. Quantitative arrays are unchanged.
        if self.cfg.invert_display_signal and analysis_method == "raw_counts":
            display_signal = -masked_map
            display_label = "-(post - baseline), median-centered"
        else:
            display_signal = masked_map

        trial_name = trial_folder.name if trial_folder.name else f"trial_{trial_number:03d}"
        out_dir = session_out / trial_name
        out_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(self._rescale_to_bit_depth(smoothed_map, bit_depth, self.cfg.rescale_mode)).save(out_dir / f"activation_map_rescaled_{bit_depth}bit.tiff")

        if self.cfg.save_arrays:
            if green_reference is not None:
                np.save(out_dir / "green_reference.npy", green_reference)
            np.save(out_dir / "baseline_reference.npy", baseline_mean)
            np.save(out_dir / "post_mean_analysis_window.npy", post_mean)
            np.save(out_dir / f"activation_map_raw_{quantitative_label}.npy", raw_diff)
            np.save(out_dir / f"activation_map_centered_{quantitative_label}.npy", centered_map)
            np.save(out_dir / f"activation_map_smoothed_{quantitative_label}.npy", smoothed_map)
            np.save(out_dir / f"activation_map_masked_{quantitative_label}.npy", masked_map)
            np.save(out_dir / "activation_map_display_signal.npy", display_signal)
            if analysis_method == "raw_counts":
                np.save(out_dir / "activation_map_raw_counts.npy", centered_map)
                np.save(out_dir / "activation_map_raw_counts_smoothed.npy", smoothed_map)
                np.save(out_dir / "activation_map_raw_counts_masked.npy", masked_map)
            else:
                np.save(out_dir / "activation_map_fractional_reflectance.npy", centered_map)
                np.save(out_dir / "activation_map_fractional_reflectance_smoothed.npy", smoothed_map)
                np.save(out_dir / "activation_map_masked_fractional_reflectance.npy", masked_map)

        with open(out_dir / "frame_qc_stats.csv", "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(qc_rows[0].keys()))
            writer.writeheader()
            writer.writerows(qc_rows)

        for row in qc_rows:
            sat = row.get("saturated_fraction")
            if sat is not None and sat > 0:
                print(f"Warning: {trial_name} {row['phase']} has {100.0 * float(sat):.5f}% saturated pixels.")

        if green_reference is not None:
            plt.figure(figsize=(8, 8))
            plt.imshow(self._display_scale(green_reference), cmap="gray")
            plt.title("Green anatomical reference")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(out_dir / "green_reference.png", dpi=200)
            plt.close()

        plt.figure(figsize=(8, 8))
        plt.imshow(self._display_scale(baseline_mean), cmap="gray")
        plt.title("Red baseline reference")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / "baseline_reference.png", dpi=200)
        plt.close()

        finite_display = np.isfinite(display_signal)
        abs_lim = float(np.nanpercentile(np.abs(display_signal[finite_display]), 99)) if np.any(finite_display) else 1.0
        if not np.isfinite(abs_lim) or abs_lim <= 0:
            abs_lim = 1.0

        plt.figure(figsize=(8, 8))
        im = plt.imshow(display_signal, cmap="gray", vmin=-abs_lim, vmax=abs_lim)
        plt.colorbar(im, fraction=0.046, pad=0.04, label=display_label)
        plt.title(f"Activation map: {trial_name}, post frames {start + 1}-{end}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / "activation_map.png", dpi=200)
        plt.close()

        if green_reference is not None and self.cfg.make_overlay:
            self._save_signal_overlay_png(
                reference=green_reference,
                signal=display_signal,
                vmax=abs_lim,
                out_path=out_dir / "activation_overlay.png",
                title=f"Overlay: {trial_name}",
                colorbar_label=display_label,
            )

        finite_masked = np.isfinite(masked_map)
        summary = {
            "trial_name": trial_name,
            "trial_folder": str(trial_folder),
            "baseline_dir": str(baseline_dir),
            "post_dir": str(post_dir),
            "green_dir": str(green_dir) if green_dir is not None else None,
            "analysis_method": analysis_method,
            "quantitative_label": quantitative_label,
            "full_frame_shape_yx": list(full_frame_shape),
            "analysis_frame_shape_yx_before_binning": list(analysis_frame_shape_pre_binning),
            "analysis_frame_shape_yx": list(analysis_frame_shape),
            "analysis_binning": int(analysis_binning),
            "roi": self.active_roi,
            "dark_reference_path": str(self.cfg.dark_reference_path) if self.cfg.dark_reference_path is not None else None,
            "dark_reference_applied": bool(dark_reference_applied),
            "denominator_floor": float(denominator_floor) if denominator_floor is not None and np.isfinite(denominator_floor) else None,
            "denominator_floor_percentile": denominator_floor_percentile,
            "denominator_floor_counts": denominator_floor_counts,
            "denominator_mask_fraction_kept": denominator_mask_fraction_kept,
            "median_filter_size": int(median_filter_size),
            "median_filter_applied": bool(median_filter_applied),
            "analysis_window_post_frame_start_1_indexed": start + 1,
            "analysis_window_post_frame_end_1_indexed_inclusive": end,
            "smoothing_sigma_px": float(self.cfg.smoothing_sigma),
            "green_reference_trim_frames": int(self.cfg.green_reference_trim_frames),
            "invert_display_signal": bool(self.cfg.invert_display_signal and analysis_method == "raw_counts"),
            "median_subtracted_value": median_reference,
            "peak_quantitative_signal": float(np.nanmax(masked_map)) if np.any(finite_masked) else None,
            "min_quantitative_signal": float(np.nanmin(masked_map)) if np.any(finite_masked) else None,
            "peak_display_signal": float(np.nanmax(display_signal)) if np.any(finite_display) else None,
            "green_frames_used": int(green_stack.shape[0]) if green_stack is not None else 0,
            "baseline_frames_used": int(baseline_stack.shape[0]),
            "post_frames_available": int(post_stack.shape[0]),
            "post_frames_used_for_analysis": int(post_window.shape[0]),
            "outputs_dir": str(out_dir),
            "rescale_mode": self.cfg.rescale_mode,
            "notes": "Offline mirror of intrinsic_imaging analysis: dark subtraction before ROI/binning, raw counts or ΔR/R, median centering, optional median filter for ΔR/R, Gaussian smoothing, gray PNG displays.",
        }
        with open(out_dir / "analysis_summary.json", "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

        return {
            "summary": summary,
            "smoothed_map": smoothed_map,
            "display_signal": display_signal,
            "green_reference": green_reference,
            "reference_image": green_reference if green_reference is not None else baseline_mean,
        }

    def _save_trial_running_average_panel(
        self,
        session_out: Path,
        trial_summaries: list[dict],
        display_maps: list[np.ndarray],
        reference_images: list[np.ndarray],
        current_trial: int,
    ) -> None:
        """Save current-trial / running-average / reference PNG after each trial.

        Output is written to offline_analysis/session_average so the offline
        processor mirrors the live acquisition script's review images.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not display_maps:
            return

        session_dir = session_out / "session_average"
        session_dir.mkdir(parents=True, exist_ok=True)

        trial_display = np.asarray(display_maps[-1], dtype=np.float64)
        running_display = np.nanmean(np.stack(display_maps, axis=0), axis=0)
        finite_running = np.isfinite(running_display)
        if np.any(finite_running):
            running_display = running_display - float(np.nanmedian(running_display[finite_running]))

        reference = np.asarray(reference_images[-1], dtype=np.float64)

        combined_values = np.concatenate([
            np.ravel(trial_display[np.isfinite(trial_display)]),
            np.ravel(running_display[np.isfinite(running_display)]),
        ])
        if combined_values.size:
            vmax = float(np.nanpercentile(np.abs(combined_values), 99))
        else:
            vmax = 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        trial_name = trial_summaries[-1].get("trial_name", f"trial_{current_trial:03d}")
        n_included = len(display_maps)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

        im_trial = axes[0].imshow(trial_display, cmap="gray", vmin=-vmax, vmax=vmax)
        axes[0].set_title(f"{trial_name} red subtraction")
        axes[0].axis("off")

        im_running = axes[1].imshow(running_display, cmap="gray", vmin=-vmax, vmax=vmax)
        axes[1].set_title(f"Running average red subtraction\nthrough {trial_name} (n={n_included})")
        axes[1].axis("off")

        axes[2].imshow(self._display_scale(reference), cmap="gray")
        ref_label = "Green reference" if trial_summaries[-1].get("green_dir") else "Baseline reference"
        axes[2].set_title(ref_label)
        axes[2].axis("off")

        fig.colorbar(
            im_running if im_running is not None else im_trial,
            ax=axes[:2].ravel().tolist(),
            fraction=0.035,
            pad=0.02,
            label="red subtraction display signal",
        )
        fig.suptitle("Offline intrinsic imaging post-trial summary", fontsize=14)

        panel_path = session_dir / f"trial_{current_trial:03d}_running_average_reference_panel.png"
        fig.savefig(panel_path, dpi=200)
        plt.close(fig)

        if self.cfg.make_overlay:
            self._save_signal_overlay_png(
                reference=reference,
                signal=running_display,
                vmax=vmax,
                out_path=session_dir / f"trial_{current_trial:03d}_running_average_overlay.png",
                title=f"Running-average overlay through {trial_name} (n={n_included})",
                colorbar_label="running-average red subtraction",
            )

        panel_summary = {
            "trial_number": int(current_trial),
            "trial_name": trial_name,
            "num_trials_included": int(n_included),
            "panel_path": str(panel_path),
            "shared_display_vmax_p99_abs": float(vmax),
            "reference_panel": ref_label,
            "panels": [
                "current trial red subtraction",
                "running average red subtraction",
                ref_label,
            ],
            "notes": "Three-panel PNG uses one shared signed display scale for the trial and running-average maps.",
        }
        with open(session_dir / f"trial_{current_trial:03d}_running_average_reference_panel_summary.json", "w", encoding="utf-8") as fp:
            json.dump(panel_summary, fp, indent=2)

        print(f"Saved three-panel running-average image: {panel_path}")

    def _save_session_average(
        self,
        session_out: Path,
        trial_summaries: list[dict],
        trial_maps: list[np.ndarray],
        display_maps: list[np.ndarray],
        green_refs: list[np.ndarray],
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        if not trial_maps:
            return

        session_dir = session_out / "session_average"
        session_dir.mkdir(parents=True, exist_ok=True)

        mean_map = np.nanmean(np.stack(trial_maps, axis=0), axis=0)
        mean_display = np.nanmean(np.stack(display_maps, axis=0), axis=0)
        finite = np.isfinite(mean_map)
        if np.any(finite):
            mean_map = mean_map - float(np.nanmedian(mean_map[finite]))
        finite_display = np.isfinite(mean_display)
        if np.any(finite_display):
            mean_display = mean_display - float(np.nanmedian(mean_display[finite_display]))

        mean_green = None
        if green_refs:
            mean_green = np.nanmean(np.stack(green_refs, axis=0), axis=0)

        if self.cfg.save_arrays:
            np.save(session_dir / "session_mean_activation_map.npy", mean_map)
            np.save(session_dir / "session_mean_display_signal.npy", mean_display)
            if mean_green is not None:
                np.save(session_dir / "session_mean_green_reference.npy", mean_green)

        bit_depth = int(self.cfg.rescale_bit_depth)
        Image.fromarray(self._rescale_to_bit_depth(mean_map, bit_depth, self.cfg.rescale_mode)).save(session_dir / f"session_mean_activation_rescaled_{bit_depth}bit.tiff")

        abs_lim = float(np.nanpercentile(np.abs(mean_display[np.isfinite(mean_display)]), 99)) if np.any(np.isfinite(mean_display)) else 1.0
        if not np.isfinite(abs_lim) or abs_lim <= 0:
            abs_lim = 1.0

        plt.figure(figsize=(8, 8))
        im = plt.imshow(mean_display, cmap="gray", vmin=-abs_lim, vmax=abs_lim)
        plt.colorbar(im, fraction=0.046, pad=0.04, label="session mean post - baseline")
        plt.title(f"Session mean intrinsic signal, n={len(trial_maps)} trials")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(session_dir / "session_mean_activation_map.png", dpi=200)
        plt.close()

        if mean_green is not None and self.cfg.make_overlay:
            self._save_signal_overlay_png(
                reference=mean_green,
                signal=mean_display,
                vmax=abs_lim,
                out_path=session_dir / "session_mean_activation_overlay.png",
                title="Session mean overlay",
                colorbar_label="session mean post - baseline",
            )

        summary = {
            "num_trials_included": len(trial_maps),
            "trial_names": [s["trial_name"] for s in trial_summaries],
            "smoothing_sigma_px": float(self.cfg.smoothing_sigma),
            "roi": self.active_roi,
            "session_average_dir": str(session_dir),
            "rescale_mode": self.cfg.rescale_mode,
            "notes": "Mean of offline processed signed trial maps, re-centered by median after averaging.",
        }
        with open(session_dir / "session_analysis_summary.json", "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

        # Also save a compact CSV of trial-level summaries.
        if trial_summaries:
            keys = [
                "trial_name", "baseline_frames_used", "post_frames_available",
                "post_frames_used_for_analysis", "median_subtracted_value",
                "peak_signal", "min_signal", "outputs_dir"
            ]
            with open(session_dir / "trial_analysis_summary.csv", "w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=keys)
                writer.writeheader()
                for s in trial_summaries:
                    writer.writerow({k: s.get(k, "") for k in keys})


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline intrinsic imaging processor")
    parser.add_argument("--folder", type=Path, default=None, help="Folder to process. If omitted, a folder picker opens.")
    parser.add_argument("--output-name", type=str, default="offline_analysis", help="Output folder name created inside the selected folder.")
    parser.add_argument("--green-dir", type=Path, default=None, help="Optional green reference folder. If omitted, script searches for session_green_reference/ or trial green/.")
    parser.add_argument("--roi-config", type=Path, default=None, help="Optional selected_roi.json. Used only as software analysis crop.")
    parser.add_argument("--width", type=int, default=1920, help="RAW input full-frame image width. Default: 1920.")
    parser.add_argument("--height", type=int, default=1200, help="RAW input full-frame image height. Default: 1200.")
    parser.add_argument("--pixel-format", type=str, default="Mono16", choices=["Mono8", "Mono12", "Mono16"], help="RAW pixel format.")
    parser.add_argument("--analysis-start-frame", type=int, default=5, help="0-indexed first post frame to average. Default 5 matches intrinsic_imaging.")
    parser.add_argument("--analysis-end-frame", type=int, default=35, help="0-indexed exclusive end post frame. Default 35 matches intrinsic_imaging. Use <=0 for all remaining frames.")
    parser.add_argument("--smoothing-sigma", type=float, default=5.0, help="Gaussian filter sigma in pixels.")
    parser.add_argument("--analysis-method", type=normalize_analysis_method, default="raw_counts", choices=["raw_counts", "fractional_reflectance"], help="Analysis method. Default 'raw_counts' is post mean - baseline mean. 'fractional_reflectance' uses fractional reflectance ΔR/R with a denominator floor. Method names used by earlier versions are still accepted.")
    parser.add_argument("--analysis-binning", type=int, default=1, help="Software analysis binning factor. Saved RAW files are unchanged.")
    parser.add_argument("--dark-reference", type=Path, default=None, help="Optional master dark reference .npy/.tiff subtracted before ROI/binning/analysis.")
    parser.add_argument("--analysis-denominator-floor-percentile", type=float, default=5.0, help="For ΔR/R, mask baseline pixels below this positive-baseline percentile.")
    parser.add_argument("--analysis-denominator-floor-counts", type=float, default=100.0, help="For ΔR/R, also require at least this many dark-corrected counts before division.")
    parser.add_argument("--analysis-median-filter-size", type=int, default=3, help="Odd median-filter size applied to ΔR/R before centering/smoothing. Use 1 to disable.")
    parser.add_argument("--rescale-bit-depth", type=int, default=16, help="Bit depth for rescaled TIFF output.")
    parser.add_argument("--rescale-mode", type=str, default="signed_symmetric", choices=["signed_symmetric", "minmax"], help="TIFF scaling mode. signed_symmetric maps zero to mid-gray; minmax maps finite min to 0 and max to 65535.")
    parser.add_argument("--invert-display-signal", action="store_true", help="For raw_counts only: plot reflectance decreases as positive/hot in display PNGs. Quantitative arrays are unchanged. Mirrors intrinsic_imaging.")
    parser.add_argument("--green-reference-trim-frames", type=int, default=5, help="Frames trimmed from BOTH ends of the green stack before averaging the analysis green reference. Mirrors intrinsic_imaging (default 5).")
    parser.add_argument("--no-auto-frame-dims", action="store_true", help="Do not read frame width/height/pixel_format from trial meta/trial_metadata.json; use --width/--height/--pixel-format instead.")
    parser.add_argument("--no-save-arrays", action="store_true", help="Skip .npy output arrays.")
    parser.add_argument("--no-overlay", action="store_true", help="Skip green-reference overlay PNGs.")
    parser.add_argument("--overlay-cmap", type=str, default="seismic", help="Diverging colormap for the recolored signal overlay (e.g. seismic, bwr, RdBu_r, coolwarm).")
    parser.add_argument("--overlay-alpha-mode", type=str, default="magnitude", choices=["magnitude", "flat"], help="magnitude: transparency scales with |signal| so anatomy shows through weak regions. flat: constant alpha everywhere.")
    parser.add_argument("--overlay-alpha", type=float, default=0.55, help="Flat overlay opacity, used only when --overlay-alpha-mode flat.")
    parser.add_argument("--overlay-threshold-frac", type=float, default=0.0, help="Hide pixels with |signal| below this fraction of the display scale. 0 shows everything; try 0.2-0.3 for clean activation islands.")
    parser.add_argument("--no-recursive", action="store_true", help="Do not search recursively for trial folders.")
    parser.add_argument("--no-interactive-roi", action="store_true", help="Skip drawing an ROI on the averaged green reference.")
    parser.add_argument("--green-inner-frames", type=int, default=20, help="ROI-picker only: number of middle green frames averaged for the interactive ROI reference image. Does NOT affect the quantitative analysis green reference (see --green-reference-trim-frames). Default: 20.")
    return parser.parse_args(argv)


def choose_folder_with_gui() -> Optional[Path]:
    tk, filedialog, messagebox = _try_import_tkinter()
    if tk is None:
        return None
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select intrinsic imaging SESSION folder (session_YYYYMMDD_HHMMSS)")
    root.destroy()
    return Path(folder) if folder else None


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    folder = args.folder or choose_folder_with_gui()
    if folder is None:
        print("No folder selected. Use --folder or run in an environment with tkinter.", file=sys.stderr)
        return 1

    cfg = ProcessingConfig(
        folder=folder,
        output_name=args.output_name,
        green_dir=args.green_dir,
        roi_config=args.roi_config,
        width=args.width,
        height=args.height,
        pixel_format=args.pixel_format,
        analysis_start_frame=args.analysis_start_frame,
        analysis_end_frame=args.analysis_end_frame if args.analysis_end_frame > 0 else None,
        smoothing_sigma=args.smoothing_sigma,
        analysis_binning=args.analysis_binning,
        analysis_method=args.analysis_method,
        dark_reference_path=args.dark_reference,
        denominator_floor_percentile=args.analysis_denominator_floor_percentile,
        denominator_floor_counts=args.analysis_denominator_floor_counts,
        median_filter_size=args.analysis_median_filter_size,
        rescale_bit_depth=args.rescale_bit_depth,
        rescale_mode=args.rescale_mode,
        invert_display_signal=args.invert_display_signal,
        auto_frame_dims=not args.no_auto_frame_dims,
        green_reference_trim_frames=args.green_reference_trim_frames,
        save_arrays=not args.no_save_arrays,
        make_overlay=not args.no_overlay,
        overlay_cmap=args.overlay_cmap,
        overlay_alpha_mode=args.overlay_alpha_mode,
        overlay_alpha=args.overlay_alpha,
        overlay_threshold_frac=args.overlay_threshold_frac,
        recursive=not args.no_recursive,
        interactive_roi=not args.no_interactive_roi,
        green_inner_frames=args.green_inner_frames,
    )

    try:
        processor = OfflineIntrinsicProcessor(cfg)
        out = processor.process_folder(folder)
        print(f"Done. Offline analysis saved to: {out}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
