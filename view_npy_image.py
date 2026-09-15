#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
view_npy_image.py

Open a file picker, select a NumPy .npy array, and view it as an image.

Usage:
    py -3.10 view_npy_image.py

Optional:
    py -3.10 view_npy_image.py --mode signed
    py -3.10 view_npy_image.py --mode gray
    py -3.10 view_npy_image.py --mode hot
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt


def choose_file(initial_dir: Path | None = None) -> Path | None:
    """Open a Windows/File Explorer-style picker and return the selected .npy file."""
    root = tk.Tk()
    root.withdraw()
    root.update()

    filename = filedialog.askopenfilename(
        title="Select a NumPy array to view",
        initialdir=str(initial_dir) if initial_dir else None,
        filetypes=[
            ("NumPy arrays", "*.npy"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    if not filename:
        return None

    return Path(filename)


def array_to_2d_image(arr: np.ndarray) -> np.ndarray:
    """Convert common array shapes into a 2D image for display."""
    arr = np.asarray(arr)

    if arr.ndim == 2:
        return arr.astype(np.float64)

    if arr.ndim == 3:
        # If this is an image stack shaped frames x height x width, average frames.
        if arr.shape[0] > 1 and arr.shape[-1] not in (3, 4):
            print(f"Array looks like a stack: {arr.shape}. Displaying mean across axis 0.")
            return np.nanmean(arr.astype(np.float64), axis=0)

        # If this is RGB/RGBA, convert to grayscale for simple scientific display.
        if arr.shape[-1] in (3, 4):
            rgb = arr[..., :3].astype(np.float64)
            print(f"Array looks like RGB/RGBA: {arr.shape}. Displaying grayscale luminance.")
            return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    raise ValueError(
        f"Cannot display array with shape {arr.shape}. "
        "Expected a 2D image, a frame stack, or an RGB/RGBA image."
    )


def robust_limits(img: np.ndarray, mode: str, percentile: float) -> tuple[float, float]:
    """Choose display limits for gray, hot, or signed visualization."""
    finite = np.isfinite(img)
    if not np.any(finite):
        return 0.0, 1.0

    values = img[finite]

    if mode == "signed":
        abs_lim = float(np.nanpercentile(np.abs(values), percentile))
        if not np.isfinite(abs_lim) or abs_lim <= 0:
            abs_lim = 1.0
        return -abs_lim, abs_lim

    if mode == "hot":
        vmax = float(np.nanpercentile(np.abs(values), percentile))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        return 0.0, vmax

    # gray
    lo = float(np.nanpercentile(values, 1))
    hi = float(np.nanpercentile(values, percentile))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    return lo, hi


def infer_mode(path: Path, img: np.ndarray, requested_mode: str) -> str:
    """Pick a useful default mode when --mode auto is used."""
    if requested_mode != "auto":
        return requested_mode

    name = path.name.lower()

    if any(key in name for key in ["green", "baseline", "reference", "post_mean"]):
        return "gray"

    if any(key in name for key in ["activation", "display_signal", "raw_counts", "fractional", "neg_drr", "subtraction", "diff"]):
        return "signed"

    finite = np.isfinite(img)
    if np.any(finite) and np.nanmin(img[finite]) < 0:
        return "signed"

    return "gray"


def print_stats(path: Path, arr: np.ndarray, img: np.ndarray) -> None:
    """Print useful diagnostics to the command line."""
    finite = np.isfinite(img)

    print("\nSelected file:")
    print(f"  {path}")
    print("\nArray:")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")

    print("\nDisplayed image:")
    print(f"  shape: {img.shape}")
    print(f"  dtype: {img.dtype}")

    if np.any(finite):
        values = img[finite]
        print(f"  min:    {np.nanmin(values):.6g}")
        print(f"  p1:     {np.nanpercentile(values, 1):.6g}")
        print(f"  median: {np.nanmedian(values):.6g}")
        print(f"  p99:    {np.nanpercentile(values, 99):.6g}")
        print(f"  max:    {np.nanmax(values):.6g}")
    else:
        print("  No finite values found.")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Pick and view a NumPy .npy array as an image.")
    parser.add_argument(
        "--mode",
        choices=["auto", "gray", "signed", "hot"],
        default="auto",
        help=(
            "Display mode. auto uses file name/content to choose. "
            "gray is best for reference images, signed is best for subtraction maps, "
            "hot is one-sided positive display."
        ),
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Upper percentile used for display scaling. Default: 99.",
    )
    parser.add_argument(
        "--initial-dir",
        type=Path,
        default=None,
        help="Optional folder where the file picker should start.",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also save a PNG next to the selected .npy file using the current display settings.",
    )

    args = parser.parse_args(argv)

    path = choose_file(args.initial_dir)
    if path is None:
        print("No file selected.")
        return 0

    try:
        arr = np.load(path)
        img = array_to_2d_image(arr)
    except Exception as exc:
        message = f"Could not load/display this file:\n\n{path}\n\nError:\n{exc}"
        print(message, file=sys.stderr)
        try:
            messagebox.showerror("NumPy viewer error", message)
        except Exception:
            pass
        return 1

    mode = infer_mode(path, img, args.mode)
    vmin, vmax = robust_limits(img, mode, args.percentile)

    if mode == "signed":
        cmap = "seismic"
        label = "signed value"
    elif mode == "hot":
        cmap = "hot"
        label = "positive display value"
    else:
        cmap = "gray"
        label = "pixel intensity"

    print_stats(path, arr, img)
    print("\nDisplay:")
    print(f"  mode: {mode}")
    print(f"  cmap: {cmap}")
    print(f"  vmin: {vmin:.6g}")
    print(f"  vmax: {vmax:.6g}")

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    ax.set_title(path.name)
    ax.axis("off")
    fig.tight_layout()

    if args.save_png:
        out_path = path.with_suffix("")
        out_path = out_path.parent / f"{out_path.name}_{mode}_view.png"
        fig.savefig(out_path, dpi=200)
        print(f"\nSaved PNG: {out_path}")

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
