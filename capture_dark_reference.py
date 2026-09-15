#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Capture a capped-lens dark reference stack from a FLIR/Blackfly camera and average it.

Use case:
    1) Put the lens cap on.
    2) Turn ALL LEDs/room lights off.
    3) Run this script with the same exposure/gain/pixel format used for ISI.
    4) Use dark_reference_mean.npy or dark_reference_mean.tiff for later correction.

Example:
    py -3.10 capture_dark_reference.py --output .\dark_refs --frames 1000 --exposure-us 23000 --pixel-format Mono16 --save-frames

If you only want the averaged dark reference and not 1000 individual frames:
    py -3.10 capture_dark_reference.py --output .\dark_refs --frames 1000 --exposure-us 23000 --pixel-format Mono16
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import PySpin  # type: ignore
except ImportError as exc:
    raise SystemExit("PySpin is not installed or not visible to this Python environment.") from exc


def get_node(nodemap, name: str):
    node = nodemap.GetNode(name)
    if node is None or not PySpin.IsAvailable(node):
        raise RuntimeError(f"Camera node not available: {name}")
    return node


def set_enum(nodemap, name: str, entry_name: str) -> None:
    node = PySpin.CEnumerationPtr(get_node(nodemap, name))
    if not PySpin.IsWritable(node):
        raise RuntimeError(f"Enum node not writable: {name}")
    entry = node.GetEntryByName(entry_name)
    if entry is None or not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
        raise RuntimeError(f"Enum entry {entry_name!r} not available for {name}")
    node.SetIntValue(entry.GetValue())


def set_float(nodemap, name: str, value: float) -> float:
    node = PySpin.CFloatPtr(get_node(nodemap, name))
    if not PySpin.IsWritable(node):
        raise RuntimeError(f"Float node not writable: {name}")
    clipped = max(float(node.GetMin()), min(float(node.GetMax()), float(value)))
    node.SetValue(clipped)
    return float(node.GetValue())


def get_float(nodemap, name: str) -> float | None:
    try:
        node = PySpin.CFloatPtr(get_node(nodemap, name))
        if PySpin.IsReadable(node):
            return float(node.GetValue())
    except Exception:
        return None
    return None


def get_int(nodemap, name: str) -> int | None:
    try:
        node = PySpin.CIntegerPtr(get_node(nodemap, name))
        if PySpin.IsReadable(node):
            return int(node.GetValue())
    except Exception:
        return None
    return None


def get_enum(nodemap, name: str) -> str | None:
    try:
        node = PySpin.CEnumerationPtr(get_node(nodemap, name))
        entry = node.GetCurrentEntry()
        if entry is not None and PySpin.IsReadable(entry):
            return str(entry.GetSymbolic())
    except Exception:
        return None
    return None


def configure_camera(cam, args) -> dict:
    nodemap = cam.GetNodeMap()

    # Free-run acquisition. This is intentionally independent of the Arduino.
    set_enum(nodemap, "TriggerMode", "Off")
    set_enum(nodemap, "AcquisitionMode", "Continuous")
    set_enum(nodemap, "PixelFormat", args.pixel_format)

    try:
        set_enum(nodemap, "ExposureAuto", "Off")
        set_enum(nodemap, "ExposureMode", "Timed")
        exposure_actual = set_float(nodemap, "ExposureTime", args.exposure_us)
    except Exception as exc:
        raise RuntimeError(f"Could not configure exposure: {exc}") from exc

    gain_actual = None
    try:
        set_enum(nodemap, "GainAuto", "Off")
        if args.gain_db is not None:
            gain_actual = set_float(nodemap, "Gain", args.gain_db)
        else:
            gain_actual = get_float(nodemap, "Gain")
    except Exception:
        gain_actual = get_float(nodemap, "Gain")

    width = get_int(nodemap, "Width")
    height = get_int(nodemap, "Height")

    if width is None or height is None:
        raise RuntimeError("Could not read camera Width/Height.")

    return {
        "pixel_format": get_enum(nodemap, "PixelFormat"),
        "width": width,
        "height": height,
        "exposure_us_requested": float(args.exposure_us),
        "exposure_us_actual": exposure_actual,
        "gain_db_requested": args.gain_db,
        "gain_db_actual": gain_actual,
        "trigger_mode": get_enum(nodemap, "TriggerMode"),
        "acquisition_mode": get_enum(nodemap, "AcquisitionMode"),
    }


def image_to_array(image, pixel_format: str) -> np.ndarray:
    arr = image.GetNDArray()
    # PySpin usually returns uint16 for Mono16/Mono12 and uint8 for Mono8.
    if pixel_format in {"Mono16", "Mono12"} and arr.dtype != np.uint16:
        arr = arr.astype(np.uint16, copy=False)
    elif pixel_format == "Mono8" and arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return arr


def save_tiff(path: Path, arr: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(arr).save(path)


def save_preview(path: Path, arr: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = np.isfinite(arr)
    if np.any(finite):
        lo, hi = np.nanpercentile(arr[finite], [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(arr[finite])), float(np.nanmax(arr[finite]))
    else:
        lo, hi = 0.0, 1.0

    plt.figure(figsize=(8, 6))
    plt.imshow(arr, cmap="gray", vmin=lo, vmax=hi)
    plt.title("Mean dark reference, percentile display")
    plt.colorbar(fraction=0.046, pad=0.04, label="camera counts")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture and average capped-lens dark reference frames from a Blackfly camera.")
    parser.add_argument("--output", type=Path, required=True, help="Output base folder.")
    parser.add_argument("--frames", type=int, default=1000, help="Number of capped frames to acquire.")
    parser.add_argument("--exposure-us", type=float, default=23000.0, help="Exposure time in microseconds. Match your ISI script.")
    parser.add_argument("--gain-db", type=float, default=None, help="Optional gain in dB. Omit to leave at current/default after GainAuto off.")
    parser.add_argument("--pixel-format", type=str, default="Mono16", choices=["Mono8", "Mono12", "Mono16"])
    parser.add_argument("--warmup-frames", type=int, default=20, help="Frames to discard before recording the stack.")
    parser.add_argument("--timeout-ms", type=int, default=2000, help="Timeout per frame in milliseconds.")
    parser.add_argument("--save-frames", action="store_true", help="Save all individual frames as TIFFs. Otherwise only average/reference outputs are saved.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing timestamp folder if it somehow exists.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    session_dir = args.output / time.strftime("dark_reference_%Y%m%d_%H%M%S")
    if session_dir.exists() and not args.overwrite:
        raise RuntimeError(f"Output folder already exists: {session_dir}")
    session_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = session_dir / "frames"
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    print("Dark-reference acquisition")
    print("  Put the lens cap on.")
    print("  Turn all LEDs and room lights OFF.")
    print(f"  Frames: {args.frames}")
    print(f"  Output: {session_dir}")

    system = None
    cam_list = None
    cam = None

    try:
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        if cam_list.GetSize() < 1:
            raise RuntimeError("No Blackfly/Spinnaker camera detected.")

        cam = cam_list.GetByIndex(0)
        cam.Init()
        camera_settings = configure_camera(cam, args)

        print("Camera settings:")
        for key, value in camera_settings.items():
            print(f"  {key}: {value}")

        metadata = {
            "created_iso": datetime.now().isoformat(timespec="seconds"),
            "purpose": "capped-lens dark reference for camera offset/fixed-pattern correction",
            "frames_requested": int(args.frames),
            "warmup_frames": int(args.warmup_frames),
            "save_individual_frames": bool(args.save_frames),
            "camera_settings": camera_settings,
            "notes": "Use with the same exposure, gain, pixel format, and temperature conditions as ISI acquisition.",
        }

        height = int(camera_settings["height"])
        width = int(camera_settings["width"])
        running_sum = np.zeros((height, width), dtype=np.float64)
        running_sum_sq = np.zeros((height, width), dtype=np.float64)
        frame_stats_path = session_dir / "frame_stats.csv"

        cam.BeginAcquisition()
        try:
            # Discard warmup frames so exposure/readout state settles.
            for i in range(max(0, int(args.warmup_frames))):
                image = cam.GetNextImage(args.timeout_ms)
                try:
                    if not image.IsIncomplete():
                        pass
                finally:
                    image.Release()
            if args.warmup_frames > 0:
                print(f"Discarded {args.warmup_frames} warmup frames.")

            with open(frame_stats_path, "w", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                writer.writerow(["frame_index", "timestamp_iso", "camera_timestamp", "min", "mean", "median", "p99", "max"])

                for idx in range(1, int(args.frames) + 1):
                    image = cam.GetNextImage(args.timeout_ms)
                    try:
                        if image.IsIncomplete():
                            print(f"Warning: incomplete image at frame {idx}; skipping.")
                            continue

                        arr = image_to_array(image, args.pixel_format)
                        arr_f = arr.astype(np.float64)
                        running_sum += arr_f
                        running_sum_sq += arr_f * arr_f

                        cam_ts = int(image.GetTimeStamp())
                        writer.writerow([
                            idx,
                            datetime.now().isoformat(timespec="milliseconds"),
                            cam_ts,
                            float(np.min(arr_f)),
                            float(np.mean(arr_f)),
                            float(np.median(arr_f)),
                            float(np.percentile(arr_f, 99)),
                            float(np.max(arr_f)),
                        ])

                        if args.save_frames:
                            save_tiff(frames_dir / f"dark_{idx:05d}.tiff", arr)

                        if idx % 50 == 0 or idx == 1 or idx == args.frames:
                            print(f"Captured {idx}/{args.frames} frames")
                    finally:
                        image.Release()
        finally:
            try:
                cam.EndAcquisition()
            except Exception:
                pass

        n = int(args.frames)
        mean = running_sum / max(1, n)
        variance = (running_sum_sq / max(1, n)) - (mean * mean)
        variance = np.maximum(variance, 0)
        std = np.sqrt(variance)

        np.save(session_dir / "dark_reference_mean.npy", mean)
        np.save(session_dir / "dark_reference_std.npy", std)

        # Save a uint16 TIFF version of the mean for easy viewing/use in Fiji.
        if args.pixel_format == "Mono8":
            mean_tiff = np.clip(np.rint(mean), 0, 255).astype(np.uint8)
        else:
            mean_tiff = np.clip(np.rint(mean), 0, 65535).astype(np.uint16)
        save_tiff(session_dir / "dark_reference_mean.tiff", mean_tiff)

        # Save std as float-compatible display product scaled into uint16 for convenience.
        std_scaled = std.copy()
        if np.nanmax(std_scaled) > 0:
            std_scaled = std_scaled / np.nanmax(std_scaled) * 65535.0
        save_tiff(session_dir / "dark_reference_std_scaled_16bit.tiff", np.clip(std_scaled, 0, 65535).astype(np.uint16))

        save_preview(session_dir / "dark_reference_mean_preview.png", mean)
        save_preview(session_dir / "dark_reference_std_preview.png", std)

        summary = {
            **metadata,
            "frames_used_for_average": n,
            "mean_counts_global": float(np.mean(mean)),
            "median_counts_global": float(np.median(mean)),
            "min_counts_global": float(np.min(mean)),
            "max_counts_global": float(np.max(mean)),
            "std_counts_global_mean": float(np.mean(std)),
            "std_counts_global_median": float(np.median(std)),
            "outputs": {
                "mean_npy": str(session_dir / "dark_reference_mean.npy"),
                "std_npy": str(session_dir / "dark_reference_std.npy"),
                "mean_tiff": str(session_dir / "dark_reference_mean.tiff"),
                "mean_preview_png": str(session_dir / "dark_reference_mean_preview.png"),
                "frame_stats_csv": str(frame_stats_path),
            },
        }
        with open(session_dir / "dark_reference_summary.json", "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

        print("Done.")
        print(f"Mean dark reference: {session_dir / 'dark_reference_mean.npy'}")
        print(f"Preview: {session_dir / 'dark_reference_mean_preview.png'}")
        return 0

    finally:
        if cam is not None:
            try:
                cam.DeInit()
            except Exception:
                pass
        if cam_list is not None:
            cam_list.Clear()
        if system is not None:
            system.ReleaseInstance()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
