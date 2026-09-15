#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
r"""
Convert a NumPy .npy array into a TIFF image.

Built for the Blackfly / IOI analysis pipeline, where .npy files are a mix of
quantitative float maps (activation maps, centered/ΔR/R maps, green/baseline
references) and integer raw stacks.

Two output philosophies, kept deliberately separate:

  --scale none  (DEFAULT, quantitative)
      Float arrays are written as float32 TIFF with values untouched: negatives,
      NaN-masked pixels, and sub-count precision all survive. Integer arrays are
      written as-is. This is the file you analyze, not the file you eyeball.

  --scale signed_symmetric | minmax  (display)
      Maps the array to an 8- or 16-bit image for viewing in ImageJ/Fiji.
        signed_symmetric: -max_abs -> 0, 0 -> mid-gray, +max_abs -> full scale.
                          Use this for signed maps so dips and bumps are both
                          visible and zero is neutral gray.
        minmax:           finite min -> 0, finite max -> full scale. Zero is NOT
                          necessarily mid-gray.
      The scale mode is stamped into the output filename so a display TIFF is
      never mistaken for a quantitative one.

2D arrays become a single-page TIFF; 3D arrays (frame, H, W) become a multipage
TIFF. Scaling for a 3D stack is computed globally so frames stay comparable.

Run with no arguments and it pops a file picker for the .npy and a folder picker
for the output. CLI flags override the pickers for headless / batch use.

Examples:
  py -3.10 npy_to_tiff.py
  py -3.10 npy_to_tiff.py --input activation_map_centered.npy --output .\tiffs
  py -3.10 npy_to_tiff.py -i activation_map_centered.npy -o .\tiffs --scale signed_symmetric
  py -3.10 npy_to_tiff.py -i green_reference.npy -o .\tiffs --scale minmax --bit-depth 8
  py -3.10 npy_to_tiff.py -i green_reference.npy -o .\tiffs --scale percentile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError as exc:
    raise SystemExit(
        "tifffile is required. Install it with: py -3.10 -m pip install tifffile"
    ) from exc


def rescale_signed_symmetric(img: np.ndarray, bit_depth: int) -> np.ndarray:
    """-max_abs -> 0, 0 -> mid-gray, +max_abs -> 2^bit_depth - 1. Matches pipeline."""
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
    return np.clip(np.nan_to_num(unsigned, nan=0.0), 0, max_value).astype(dtype)


def rescale_minmax(img: np.ndarray, bit_depth: int) -> np.ndarray:
    """finite min -> 0, finite max -> 2^bit_depth - 1. Matches pipeline."""
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
    return np.clip(np.nan_to_num(unsigned, nan=0.0), 0, max_value).astype(dtype)


def rescale_percentile(img: np.ndarray, bit_depth: int,
                       lo_pct: float, hi_pct: float) -> np.ndarray:
    """Percentile clip-and-stretch, mirroring intrinsic_imaging._display_scale_for_analysis.

    lo_pct percentile -> 0, hi_pct percentile -> 2^bit_depth - 1, clipped.
    This is the scaling the three-panel figure uses for the green reference
    (default 1st/99th percentile). Falls back to full min/max if the percentile
    window is degenerate.
    """
    finite = np.isfinite(img)
    max_value = int((2 ** int(bit_depth)) - 1)
    dtype = np.uint16 if int(bit_depth) > 8 else np.uint8
    if not np.any(finite):
        return np.zeros_like(img, dtype=dtype)
    lo, hi = np.nanpercentile(img, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(img)); hi = float(np.nanmax(img))
    if hi <= lo:
        return np.zeros_like(img, dtype=dtype)
    norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0)  # the [0,1] array the panel feeds to imshow
    return np.clip(np.nan_to_num(norm * max_value, nan=0.0), 0, max_value).astype(dtype)


def pick_open_file(title: str, filetypes) -> Path | None:
    """Native file-open dialog. Returns None on cancel, raises if no GUI."""
    try:
        import tkinter
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("tkinter is not available in this Python environment.") from exc
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        raise RuntimeError(f"No GUI display available ({exc}).") from exc
    root.withdraw()
    root.attributes("-topmost", True)
    chosen = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return Path(chosen) if chosen else None


def pick_directory(title: str) -> Path | None:
    """Native folder-select dialog. Returns None on cancel, raises if no GUI."""
    try:
        import tkinter
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("tkinter is not available in this Python environment.") from exc
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        raise RuntimeError(f"No GUI display available ({exc}).") from exc
    root.withdraw()
    root.attributes("-topmost", True)
    chosen = filedialog.askdirectory(title=title, mustexist=False)
    root.destroy()
    return Path(chosen) if chosen else None


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Convert a .npy array to a TIFF image.")
    p.add_argument("-i", "--input", type=Path, default=None,
                   help="Source .npy file. If omitted, a file picker opens.")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output FOLDER for the TIFF. If omitted, a folder picker opens.")
    p.add_argument("--scale", choices=["none", "signed_symmetric", "minmax", "percentile"], default="none",
                   help="none: preserve quantitative values (float32/integer as-is). "
                        "signed_symmetric / minmax / percentile: produce a display-scaled image. "
                        "percentile matches the three-panel figure's green reference scaling. Default none.")
    p.add_argument("--clip-percentiles", type=float, nargs=2, metavar=("LO", "HI"), default=[1.0, 99.0],
                   help="Low/high percentiles for --scale percentile. Default 1 99, matching "
                        "intrinsic_imaging._display_scale_for_analysis.")
    p.add_argument("--bit-depth", type=int, choices=[8, 16], default=16,
                   help="Bit depth for scaled display output. Ignored when --scale none. Default 16.")
    args = p.parse_args(argv)

    # Resolve source .npy.
    in_path = args.input
    if in_path is None:
        print("Select the SOURCE .npy file...")
        try:
            in_path = pick_open_file("Select .npy file",
                                     [("NumPy array", "*.npy"), ("All files", "*.*")])
        except RuntimeError as exc:
            print(f"ERROR: {exc} Use --input to specify the file instead.", file=sys.stderr)
            return 1
        if in_path is None:
            print("Cancelled: no input file selected.")
            return 1

    if not in_path.is_file():
        print(f"ERROR: not a file: {in_path}", file=sys.stderr)
        return 1

    # Resolve output folder.
    out_folder = args.output
    if out_folder is None:
        print("Select the OUTPUT folder for the TIFF...")
        try:
            out_folder = pick_directory("Select OUTPUT folder (TIFF)")
        except RuntimeError as exc:
            print(f"ERROR: {exc} Use --output to specify the folder instead.", file=sys.stderr)
            return 1
        if out_folder is None:
            print("Cancelled: no output folder selected.")
            return 1

    # Load the array. allow_pickle=False: plain numeric arrays only, no code execution.
    try:
        arr = np.load(in_path, allow_pickle=False)
    except ValueError as exc:
        print(f"ERROR: could not load {in_path.name} without pickle ({exc}). "
              f"This file is not a plain numeric array.", file=sys.stderr)
        return 1

    if arr.ndim not in (2, 3):
        print(f"ERROR: array is {arr.ndim}-D with shape {arr.shape}. "
              f"Only 2D images and 3D (frame, H, W) stacks are supported.", file=sys.stderr)
        return 1
    if not np.issubdtype(arr.dtype, np.number):
        print(f"ERROR: array dtype {arr.dtype} is not numeric.", file=sys.stderr)
        return 1

    n_pages = 1 if arr.ndim == 2 else arr.shape[0]
    finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.floating) else np.ones_like(arr, dtype=bool)
    has_nan = arr.size and not np.all(finite)
    if np.any(finite):
        vmin = float(np.min(arr[finite])); vmax = float(np.max(arr[finite]))
    else:
        vmin = vmax = float("nan")

    print(f"Loaded {in_path.name}: shape={arr.shape}, dtype={arr.dtype}, "
          f"{'1 page' if n_pages == 1 else f'{n_pages} pages'}")
    print(f"Value range (finite): [{vmin:.4g}, {vmax:.4g}]"
          + (", contains NaN/inf (masked pixels)" if has_nan else ""))

    # Convert.
    if args.scale == "none":
        if np.issubdtype(arr.dtype, np.floating):
            out = arr.astype(np.float32)
            kind = "float32 (quantitative)"
        else:
            out = arr  # integer counts written as-is
            kind = f"{arr.dtype} (quantitative)"
        suffix = ""
    else:
        af = arr.astype(np.float64, copy=False)
        if args.scale == "signed_symmetric":
            out = rescale_signed_symmetric(af, args.bit_depth)
            suffix = f"_{args.scale}{args.bit_depth}"
        elif args.scale == "minmax":
            out = rescale_minmax(af, args.bit_depth)
            suffix = f"_{args.scale}{args.bit_depth}"
        else:  # percentile
            lo_pct, hi_pct = args.clip_percentiles
            out = rescale_percentile(af, args.bit_depth, lo_pct, hi_pct)
            suffix = f"_pct{lo_pct:g}-{hi_pct:g}_{args.bit_depth}"
        kind = f"uint{args.bit_depth} ({args.scale} display)"
        if has_nan:
            print("Note: NaN/inf pixels map to 0 (black) in the scaled output.")

    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / f"{in_path.stem}{suffix}.tiff"
    tifffile.imwrite(str(out_path), out)

    print(f"Wrote {kind} TIFF"
          + (f" ({n_pages} pages)" if n_pages > 1 else "")
          + f": {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
