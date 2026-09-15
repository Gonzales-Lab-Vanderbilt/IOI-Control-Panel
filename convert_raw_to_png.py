#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
convert_raw_to_png.py

Recursively convert folders of headerless Blackfly .raw images to .png.

Typical usage:
    py -3.10 convert_raw_to_png.py --width 1920 --height 1200 --dtype uint16

This opens two folder pickers:
    1. Choose the input/session folder or any parent folder that contains .raw images.
    2. Choose the output folder where PNGs should be saved.

The script finds all .raw files underneath the input folder and preserves the
folder structure inside the selected output folder.

Important:
    Headerless .raw files do not store width, height, or dtype.
    You must supply the correct frame width/height and pixel format.

Example with explicit settings:
    py -3.10 convert_raw_folders_to_png.py --width 1920 --height 1200 --dtype uint16

For hardware ROI captures, use the ROI width/height printed by your acquisition script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image


def choose_folder(title: str) -> Path | None:
    """Open a folder picker and return the selected folder."""
    root = tk.Tk()
    root.withdraw()
    root.update()

    folder = filedialog.askdirectory(title=title)

    root.destroy()

    if not folder:
        return None

    return Path(folder)


def scale_to_uint8(img: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Robustly scale a 2D image to uint8 for PNG viewing."""
    img = img.astype(np.float64, copy=False)
    finite = np.isfinite(img)

    if not np.any(finite):
        return np.zeros(img.shape, dtype=np.uint8)

    values = img[finite]
    lo, hi = np.nanpercentile(values, [low_pct, high_pct])

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(values)
        hi = np.nanmax(values)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)

    scaled = (img - lo) / (hi - lo)
    scaled = np.clip(scaled, 0, 1) * 255.0
    return scaled.astype(np.uint8)


def read_raw_image(path: Path, width: int, height: int, dtype: np.dtype) -> np.ndarray:
    data = np.fromfile(path, dtype=dtype)
    expected = int(width) * int(height)

    if data.size != expected:
        raise ValueError(
            f"{path.name}: found {data.size} pixels, expected {expected}. "
            f"Check width={width}, height={height}, dtype={dtype}."
        )

    return data.reshape((int(height), int(width)))


def output_path_for(raw_path: Path, input_root: Path, output_root: Path | None, suffix: str) -> Path:
    if output_root is None:
        return raw_path.with_name(raw_path.stem + suffix + ".png")

    relative = raw_path.relative_to(input_root)
    out_path = output_root / relative.with_suffix("")
    return out_path.with_name(out_path.name + suffix + ".png")


def convert_one(
    raw_path: Path,
    input_root: Path,
    output_root: Path | None,
    width: int,
    height: int,
    dtype: np.dtype,
    overwrite: bool,
    suffix: str,
    low_pct: float,
    high_pct: float,
) -> tuple[bool, str]:
    out_path = output_path_for(raw_path, input_root, output_root, suffix)

    if out_path.exists() and not overwrite:
        return False, f"Skipped existing: {out_path}"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = read_raw_image(raw_path, width=width, height=height, dtype=dtype)
    png = scale_to_uint8(img, low_pct=low_pct, high_pct=high_pct)

    Image.fromarray(png, mode="L").save(out_path)
    return True, f"Saved: {out_path}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Recursively convert headerless Blackfly .raw images to PNG."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input folder. If omitted, a folder picker opens.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output root folder. If omitted, a folder picker opens. "
            "The folder structure under the input folder is preserved."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Frame width in pixels. Must match the RAW capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Frame height in pixels. Must match the RAW capture height.",
    )
    parser.add_argument(
        "--dtype",
        choices=["uint8", "uint16"],
        default="uint16",
        help="RAW pixel dtype. Mono16/Mono12 saved as raw usually use uint16. Mono8 uses uint8.",
    )
    parser.add_argument(
        "--pattern",
        default="*.raw",
        help="File pattern to search recursively. Default: *.raw",
    )
    parser.add_argument(
        "--suffix",
        default="_view",
        help="Suffix added to output PNG filenames. Default: _view",
    )
    parser.add_argument(
        "--low-pct",
        type=float,
        default=1.0,
        help="Lower percentile for display scaling. Default: 1.",
    )
    parser.add_argument(
        "--high-pct",
        type=float,
        default=99.0,
        help="Upper percentile for display scaling. Default: 99.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNGs.",
    )

    args = parser.parse_args(argv)

    input_root = args.input
    if input_root is None:
        input_root = choose_folder("Select input folder containing .raw images")

    if input_root is None:
        print("No input folder selected.")
        return 0

    input_root = input_root.resolve()
    if not input_root.exists() or not input_root.is_dir():
        print(f"ERROR: input folder does not exist or is not a directory: {input_root}", file=sys.stderr)
        return 1

    output_root = args.output
    if output_root is None:
        output_root = choose_folder("Select output folder for converted PNG images")

    if output_root is None:
        print("No output folder selected.")
        return 0

    output_root = output_root.resolve()
    dtype = np.dtype(args.dtype)

    raw_files = sorted(input_root.rglob(args.pattern))
    if not raw_files:
        print(f"No files matching {args.pattern!r} found under: {input_root}")
        return 0

    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root if output_root else 'next to each RAW file'}")
    print(f"Found {len(raw_files)} RAW files.")
    print(f"Width x height: {args.width} x {args.height}")
    print(f"Dtype: {dtype}")
    print()

    converted = 0
    skipped = 0
    failed = 0

    for idx, raw_path in enumerate(raw_files, start=1):
        try:
            did_convert, message = convert_one(
                raw_path=raw_path,
                input_root=input_root,
                output_root=output_root,
                width=args.width,
                height=args.height,
                dtype=dtype,
                overwrite=args.overwrite,
                suffix=args.suffix,
                low_pct=args.low_pct,
                high_pct=args.high_pct,
            )

            if did_convert:
                converted += 1
            else:
                skipped += 1

            print(f"[{idx}/{len(raw_files)}] {message}")

        except Exception as exc:
            failed += 1
            print(f"[{idx}/{len(raw_files)}] FAILED: {raw_path} -> {exc}", file=sys.stderr)

    print()
    print("Done.")
    print(f"  Converted: {converted}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
