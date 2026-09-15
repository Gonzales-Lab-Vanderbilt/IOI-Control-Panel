#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
render_crop_reference.py

Renders a full-sensor (1920x1200), uncropped green-reference PNG for the
Analysis tab's interactive crop selector (gui/crop_selector.py) to display
as a drag-a-rectangle background, so a crop can be picked before running
statistical_analyses.py rather than only after a stats run has already
produced session_poster_figures.py's own green-reference figure.

Averages session_green_reference/*.raw (falling back to trial_001/green/
*.raw), the same source and trim-frame handling as session_poster_figures.py's
green_reference_mean() -- duplicated here rather than imported so this stays
a fast, light, headless render: numpy + Pillow only, no matplotlib/scipy.

Always headless -- both arguments are required, no folder-picker dialogs.
This is a GUI-support script, not part of the validated analysis pipeline;
it only ever reads raw frames already on disk and never touches session data.

Usage:
    py -3.10 render_crop_reference.py --session-dir <session folder> --out <png path>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

WIDTH, HEIGHT = 1920, 1200
DEFAULT_TRIM_FRAMES = 5


def green_reference_mean(session_dir: Path) -> np.ndarray:
    """Mean green reference frame, full sensor resolution. Same source and
    trim-frame logic as session_poster_figures.py's green_mean()-derived
    green_reference_mean(), duplicated (not imported) to avoid pulling in
    that module's matplotlib/scipy dependency for what needs to be a fast,
    light render."""
    files = sorted(glob.glob(str(session_dir / "session_green_reference" / "*.raw")))
    if not files:
        files = sorted(glob.glob(str(session_dir / "trial_001" / "green" / "*.raw")))
    if not files:
        raise FileNotFoundError(f"no green reference frames found under {session_dir}")

    trim = DEFAULT_TRIM_FRAMES
    meta_files = sorted(glob.glob(str(session_dir / "trial_*/meta/trial_metadata.json")))
    if meta_files:
        cfg = json.loads(Path(meta_files[0]).read_text())["trial_config"]
        trim = int(cfg.get("green_reference_trim_frames", DEFAULT_TRIM_FRAMES))
    use = files[trim:] if len(files) > trim + 1 else files

    acc = np.zeros(WIDTH * HEIGHT, dtype=np.float64)
    for f in use:
        frame = np.fromfile(f, dtype=np.uint16)
        if frame.size != WIDTH * HEIGHT:
            raise ValueError(
                f"{Path(f).name}: found {frame.size} pixels, expected "
                f"{WIDTH * HEIGHT} (not a {WIDTH}x{HEIGHT} Mono16 frame)"
            )
        acc += frame.astype(np.float64)
    return (acc / len(use)).reshape(HEIGHT, WIDTH)


def scale_to_uint8(
    img: np.ndarray, low_pct: float = 0.2, high_pct: float = 99.8, gamma: float = 0.85
) -> np.ndarray:
    """Percentile stretch + gamma, matching session_poster_figures.py's own
    green-reference display scaling (0.2/99.8 percentiles, **0.85), so this
    preview looks the same as the figure the user will see later."""
    img = img.astype(np.float64, copy=False)
    lo, hi = np.percentile(img, [low_pct, high_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(img)), float(np.max(img))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    normalized = np.clip((img - lo) / (hi - lo), 0.0, 1.0) ** gamma
    return (normalized * 255.0).astype(np.uint8)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Render a full-sensor, uncropped green-reference PNG preview "
                     "for the Analysis tab's interactive crop selector."
    )
    parser.add_argument("--session-dir", type=Path, required=True, help="Session folder to read from.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    args = parser.parse_args(argv)

    session_dir = args.session_dir.resolve()
    if not session_dir.is_dir():
        print(f"ERROR: session folder not found: {session_dir}", file=sys.stderr)
        return 1

    try:
        mean_frame = green_reference_mean(session_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: could not build green reference from {session_dir}: {exc}", file=sys.stderr)
        return 1

    png = scale_to_uint8(mean_frame)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(png, mode="L").save(args.out)
    print(f"Done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
