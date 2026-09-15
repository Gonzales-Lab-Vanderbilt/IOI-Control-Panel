#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
live_preview.py

Free-running live camera preview for animal placement and focus checks
(e.g. judging a "clean vs. blurry" vasculature capture under green light
before starting a session). Opens the Blackfly camera, grabs frames
continuously, and displays them in a live-updating matplotlib window.

Does NOT talk to the Arduino — turn LEDs on/off separately (Utilities tab,
or red.py / green.py) before and after running this.

Press 's' in the preview window to save a full-resolution snapshot (a
quantitative .npy array plus a quick-look .png) to the snapshot folder.

Usage:
    py -3.10 live_preview.py
    py -3.10 live_preview.py --camera-index 0 --exposure-us 15000 --fps 10
    py -3.10 live_preview.py --snapshot-dir .\captures_snapshots

Stop with Ctrl+C / Ctrl+Break, or by closing the preview window.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from PIL import Image

try:
    import PySpin  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "PySpin is not installed or not visible to this Python environment."
    ) from exc

_DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parent / "live_preview_snapshots"
_SNAPSHOT_MESSAGE_DURATION_S = 2.5

# IMX249: 12-bit ADC left-shifted into a 16-bit Mono16 container. Clipping
# happens at 65520 (4095 << 4), NOT 65535 (the container dtype max).
SENSOR_SATURATION_COUNT = 65520.0
_SATURATION_WARN_INTERVAL_S = 2.0

# Rendering a full 1920x1200 frame through matplotlib's Agg backend every
# redraw is the dominant cost of the display loop, not frame capture — a
# live focus/placement check doesn't need full sensor resolution on screen,
# so the shown image is downsampled. Stats (percentile scaling, saturation)
# use an even coarser subsample since they don't need visual fidelity at all.
_DISPLAY_STRIDE = 2
_STATS_STRIDE = 4


class CameraConfigError(RuntimeError):
    pass


@dataclass
class PreviewConfig:
    camera_index: int = 0
    exposure_us: float = 15000.0
    gain_db: float | None = 0.0
    pixel_format: str = "Mono16"
    fps: float = 10.0
    display_percentile_low: float = 1.0
    display_percentile_high: float = 99.0
    snapshot_dir: Path = _DEFAULT_SNAPSHOT_DIR


class LivePreview:
    def __init__(self, cfg: PreviewConfig) -> None:
        self.cfg = cfg
        self.system = None
        self.cam_list = None
        self.cam = None

        # Frame capture runs on a background thread so the matplotlib window's
        # own event loop (driven by FuncAnimation's timer) is never blocked by
        # PySpin calls or by per-frame processing — this is what keeps the
        # window responsive even when the camera free-runs faster than the
        # display can keep up (e.g. AcquisitionFrameRateEnable unavailable).
        self._latest_img: np.ndarray | None = None
        self._latest_lock = threading.Lock()
        self._frame_count = 0
        self._capture_error: Exception | None = None
        self._stop_capture = threading.Event()

        # Serializes access to the PySpin camera object between the capture
        # thread (GetNextImage) and the main thread (the exposure slider's
        # SetValue calls) -- changing ExposureTime while BeginAcquisition is
        # active is supported by the SDK (same pattern intrinsic_calibration.py
        # uses for its exposure sweep), but concurrent calls into the same
        # PySpin camera object from two threads at once are not.
        self._cam_lock = threading.Lock()
        self.exposure_limits_us: tuple[float, float] = (1.0, 1_000_000.0)

    # ── Camera / PySpin helpers (same idioms as intrinsic_calibration.py) ──────

    def get_node(self, name: str):
        node = self.cam.GetNodeMap().GetNode(name)
        if node is None or not PySpin.IsAvailable(node):
            raise CameraConfigError(f"Node not available: {name}")
        return node

    def set_enum(self, name: str, entry_name: str) -> None:
        node = PySpin.CEnumerationPtr(self.get_node(name))
        if not PySpin.IsWritable(node):
            raise CameraConfigError(f"Enum node not writable: {name}")
        entry = node.GetEntryByName(entry_name)
        if entry is None or not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
            raise CameraConfigError(f"Enum entry {entry_name} not available for {name}")
        node.SetIntValue(entry.GetValue())

    def get_enum_symbolic(self, name: str) -> str:
        node = PySpin.CEnumerationPtr(self.get_node(name))
        entry = node.GetCurrentEntry()
        if entry is None or not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
            return "UNKNOWN"
        return entry.GetSymbolic()

    def set_float(self, name: str, value: float) -> float:
        node = PySpin.CFloatPtr(self.get_node(name))
        if not PySpin.IsWritable(node):
            raise CameraConfigError(f"Float node not writable: {name}")
        clipped = max(node.GetMin(), min(node.GetMax(), float(value)))
        node.SetValue(clipped)
        return float(clipped)

    def get_float_limits(self, name: str) -> tuple[float, float]:
        node = PySpin.CFloatPtr(self.get_node(name))
        return float(node.GetMin()), float(node.GetMax())

    def set_bool(self, name: str, value: bool) -> None:
        node = PySpin.CBooleanPtr(self.get_node(name))
        if not PySpin.IsWritable(node):
            raise CameraConfigError(f"Boolean node not writable: {name}")
        node.SetValue(value)

    def set_int(self, name: str, value: int) -> int:
        node = PySpin.CIntegerPtr(self.get_node(name))
        if not PySpin.IsWritable(node):
            raise CameraConfigError(f"Integer node not writable: {name}")
        min_value, max_value = int(node.GetMin()), int(node.GetMax())
        inc = max(1, int(node.GetInc()) if hasattr(node, "GetInc") else 1)
        clipped = max(min_value, min(max_value, int(value)))
        aligned = min_value + ((clipped - min_value) // inc) * inc
        node.SetValue(max(min_value, min(max_value, aligned)))
        return int(node.GetValue())

    def reset_roi_to_full_frame(self) -> None:
        try:
            self.set_int("OffsetX", 0)
            self.set_int("OffsetY", 0)
            width_node = PySpin.CIntegerPtr(self.get_node("Width"))
            height_node = PySpin.CIntegerPtr(self.get_node("Height"))
            self.set_int("Width", int(width_node.GetMax()))
            self.set_int("Height", int(height_node.GetMax()))
            self.set_int("OffsetX", 0)
            self.set_int("OffsetY", 0)
        except CameraConfigError as exc:
            print(f"Warning: could not reset camera ROI to full frame: {exc}")

    def setup_camera(self) -> None:
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()

        num_cameras = self.cam_list.GetSize()
        if num_cameras < 1:
            raise RuntimeError("No Blackfly/Spinnaker camera detected.")
        if self.cfg.camera_index < 0 or self.cfg.camera_index >= num_cameras:
            raise RuntimeError(
                f"Camera index {self.cfg.camera_index} is invalid. "
                f"{num_cameras} camera(s) detected."
            )

        self.cam = self.cam_list.GetByIndex(self.cfg.camera_index)
        self.cam.Init()

        self.set_enum("TriggerMode", "Off")
        self.set_enum("AcquisitionMode", "Continuous")
        self.set_enum("PixelFormat", self.cfg.pixel_format)
        self.reset_roi_to_full_frame()

        self.set_enum("ExposureAuto", "Off")
        self.set_enum("ExposureMode", "Timed")
        actual_exposure = self.set_float("ExposureTime", self.cfg.exposure_us)
        self.exposure_limits_us = self.get_float_limits("ExposureTime")

        try:
            self.set_enum("GainAuto", "Off")
            if self.cfg.gain_db is not None:
                self.set_float("Gain", self.cfg.gain_db)
        except CameraConfigError:
            print("Warning: Gain node unavailable or not writable; leaving gain unchanged.")

        if self.cfg.fps > 0:
            try:
                self.set_bool("AcquisitionFrameRateEnable", True)
                self.set_float("AcquisitionFrameRate", self.cfg.fps)
            except CameraConfigError as exc:
                print(f"Warning: could not set AcquisitionFrameRate ({exc}).")

        print(
            f"Camera ready: index={self.cfg.camera_index}, "
            f"pixel_format={self.get_enum_symbolic('PixelFormat')}, "
            f"exposure_us={actual_exposure:.1f}, gain_db={self.cfg.gain_db}"
        )
        self.cam.BeginAcquisition()

    def teardown(self) -> None:
        if self.cam is not None:
            try:
                self.cam.EndAcquisition()
            except Exception:
                pass
            try:
                self.cam.DeInit()
            except Exception:
                pass
            self.cam = None
        if self.cam_list is not None:
            try:
                self.cam_list.Clear()
            except Exception:
                pass
            self.cam_list = None
        if self.system is not None:
            try:
                self.system.ReleaseInstance()
            except Exception:
                pass
            self.system = None

    def capture_one_frame(self, timeout_ms: int = 1000) -> np.ndarray:
        with self._cam_lock:
            image = self.cam.GetNextImage(timeout_ms)
            try:
                if image.IsIncomplete():
                    raise RuntimeError(f"Incomplete image. Status: {image.GetImageStatus()}")
                return image.GetNDArray().copy()
            finally:
                image.Release()

    def _capture_loop(self) -> None:
        """Background thread: pull frames as fast as the camera produces them.

        Only the single latest frame is kept (no queue) — the display loop
        always shows the freshest image and naturally drops backlog if it
        can't keep up with the camera's native rate.
        """
        consecutive_failures = 0
        while not self._stop_capture.is_set():
            try:
                img = self.capture_one_frame(timeout_ms=500)
            except PySpin.SpinnakerException:
                consecutive_failures += 1
                if consecutive_failures >= 20:
                    self._capture_error = RuntimeError(
                        "Camera stopped producing frames (20 consecutive timeouts)."
                    )
                    return
                continue
            except Exception as exc:
                self._capture_error = exc
                return

            consecutive_failures = 0
            with self._latest_lock:
                self._latest_img = img
                self._frame_count += 1

    # ── Display ──────────────────────────────────────────────────────────────

    def _display_limits(self, img: np.ndarray) -> tuple[float, float]:
        lo = float(np.percentile(img, self.cfg.display_percentile_low))
        hi = float(np.percentile(img, self.cfg.display_percentile_high))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(img)), float(np.max(img))
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def save_snapshot(self, img: np.ndarray) -> Path:
        """Save the full-resolution current frame: a quantitative .npy array
        plus a quick-look, percentile-scaled .png. Returns the .npy path."""
        self.cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        npy_path = self.cfg.snapshot_dir / f"snapshot_{stamp}.npy"
        png_path = self.cfg.snapshot_dir / f"snapshot_{stamp}.png"

        np.save(npy_path, img)

        lo, hi = self._display_limits(img)
        scaled = np.clip((img.astype(np.float64) - lo) / (hi - lo), 0, 1)
        Image.fromarray((scaled * 255).astype(np.uint8)).save(png_path)

        return npy_path

    def run(self) -> int:
        self.setup_camera()

        capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        capture_t_start = time.time()
        capture_thread.start()

        # Wait for the first frame before building the plot.
        t_wait_end = time.time() + 5.0
        while self._latest_img is None and time.time() < t_wait_end:
            if self._capture_error is not None:
                raise self._capture_error
            time.sleep(0.05)
        if self._latest_img is None:
            self._stop_capture.set()
            raise RuntimeError("Timed out waiting for the first camera frame.")

        with self._latest_lock:
            first = self._latest_img

        fig, ax = plt.subplots(figsize=(8, 8.9))
        fig.canvas.manager.set_window_title("Live Camera Preview")
        ax.axis("off")
        # Bottom margin leaves room for the exposure slider (and its left-drawn
        # "Exposure (µs)" label, which otherwise clips against the window edge).
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0.11)

        lo, hi = self._display_limits(first[::_STATS_STRIDE, ::_STATS_STRIDE])
        im = ax.imshow(
            first[::_DISPLAY_STRIDE, ::_DISPLAY_STRIDE],
            cmap="gray", vmin=lo, vmax=hi, animated=True,
        )
        # HUD text lives inside the Axes (not a figure title) so it
        # participates in blitting instead of forcing a full-canvas redraw.
        hud = ax.text(
            0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
            color="lime", fontsize=10, family="monospace",
            bbox=dict(facecolor="black", alpha=0.5, pad=3),
        )
        snapshot_hud = ax.text(
            0.02, 0.02, "", transform=ax.transAxes, va="bottom", ha="left",
            color="yellow", fontsize=10, family="monospace",
            bbox=dict(facecolor="black", alpha=0.6, pad=3),
        )

        state = {
            "cached_lo": lo, "cached_hi": hi, "last_scale_t": 0.0, "last_warn_t": 0.0,
            "display_count": 0, "display_t_start": time.time(),
            "snapshot_message": "", "snapshot_message_expires": 0.0,
        }

        def _on_key(event) -> None:
            if event.key != "s":
                return
            with self._latest_lock:
                img = self._latest_img
            if img is None:
                return
            try:
                path = self.save_snapshot(img)
            except OSError as exc:
                print(f"Snapshot failed: {exc}", file=sys.stderr)
                state["snapshot_message"] = f"Snapshot failed: {exc}"
            else:
                print(f"Saved snapshot: {path}")
                state["snapshot_message"] = f"Saved: {path.name}"
            state["snapshot_message_expires"] = time.time() + _SNAPSHOT_MESSAGE_DURATION_S

        fig.canvas.mpl_connect("key_press_event", _on_key)
        print(f"Press 's' in the preview window to save a snapshot to: {self.cfg.snapshot_dir}")

        # Exposure slider -- adjusts ExposureTime live while acquisition
        # keeps running. Range comes from the camera's own reported limits
        # rather than a guessed default, so it always reflects what the
        # hardware will actually accept.
        exp_min, exp_max = self.exposure_limits_us
        # left 0.22 (not 0.14) gives the label a real gutter so "Exposure (µs)"
        # isn't clipped off-canvas; width 0.58 keeps the value text clear of the
        # right edge across the full range; bottom 0.055 seats it in the 0.11 margin.
        ax_slider = fig.add_axes([0.22, 0.055, 0.58, 0.03])
        exposure_slider = Slider(
            ax_slider, "Exposure (µs)",
            valmin=exp_min, valmax=exp_max, valinit=self.cfg.exposure_us,
            valstep=max(1.0, (exp_max - exp_min) / 2000.0),
            color="#4a90d9",
        )
        exposure_slider.label.set_color("white")
        exposure_slider.label.set_fontsize(11)
        exposure_slider.valtext.set_color("white")

        def _on_exposure_changed(val: float) -> None:
            with self._cam_lock:
                try:
                    actual = self.set_float("ExposureTime", val)
                except CameraConfigError as exc:
                    print(f"Warning: could not set exposure: {exc}", file=sys.stderr)
                    return
            self.cfg.exposure_us = actual
            # The hardware may clip/align the requested value (step
            # increment, min/max) -- reflect what was actually applied back
            # onto the slider without re-firing this callback.
            if abs(actual - val) > 0.5:
                exposure_slider.eventson = False
                exposure_slider.set_val(actual)
                exposure_slider.eventson = True
            # FuncAnimation(blit=True) below only repaints the image/HUD
            # artists each frame; nudge a full redraw so the slider itself
            # (a separate, non-blitted Axes) visibly tracks the drag.
            fig.canvas.draw_idle()

        exposure_slider.on_changed(_on_exposure_changed)
        print(f"Drag the Exposure slider to adjust ExposureTime live ({exp_min:.0f}-{exp_max:.0f} us).")

        def _update(_frame):
            if self._capture_error is not None:
                print(f"ERROR during capture: {self._capture_error}", file=sys.stderr)
                plt.close(fig)
                return (im, hud, snapshot_hud)

            with self._latest_lock:
                img = self._latest_img
                capture_count = self._frame_count
            if img is None:
                return (im, hud, snapshot_hud)

            now = time.time()
            # Recompute contrast scaling on a cheap subsample, and only a
            # few times a second — this (plus the background capture
            # thread and blitted rendering below) is what keeps the window
            # responsive even when the camera free-runs much faster than
            # the display can keep up.
            sample = img[::_STATS_STRIDE, ::_STATS_STRIDE]
            if now - state["last_scale_t"] > 0.5:
                state["cached_lo"], state["cached_hi"] = self._display_limits(sample)
                state["last_scale_t"] = now

            im.set_data(img[::_DISPLAY_STRIDE, ::_DISPLAY_STRIDE])
            im.set_clim(state["cached_lo"], state["cached_hi"])
            state["display_count"] += 1

            sat_frac = float(np.mean(sample >= SENSOR_SATURATION_COUNT))
            capture_fps = capture_count / max(1e-6, now - capture_t_start)
            display_fps = state["display_count"] / max(1e-6, now - state["display_t_start"])
            hud.set_text(
                f"frame {capture_count}\n"
                f"capture {capture_fps:.1f} fps  |  display {display_fps:.1f} fps\n"
                f"exposure {self.cfg.exposure_us:.0f} us  |  saturated {sat_frac * 100:.1f}%"
            )

            if sat_frac > 0.02 and (now - state["last_warn_t"]) > _SATURATION_WARN_INTERVAL_S:
                print(f"Warning: {sat_frac * 100:.1f}% of pixels saturated (>= {SENSOR_SATURATION_COUNT:.0f}).")
                state["last_warn_t"] = now

            if state["snapshot_message"] and now < state["snapshot_message_expires"]:
                snapshot_hud.set_text(state["snapshot_message"])
            elif snapshot_hud.get_text():
                snapshot_hud.set_text("")
                state["snapshot_message"] = ""

            return (im, hud, snapshot_hud)

        interval_ms = max(15, int(1000.0 / self.cfg.fps)) if self.cfg.fps > 0 else 66
        # Keep a reference — matplotlib only keeps a weak reference to the
        # animation, and an unreferenced FuncAnimation gets garbage collected
        # (and stops firing) as soon as run() would otherwise return control.
        anim = FuncAnimation(fig, _update, interval=interval_ms, blit=True, cache_frame_data=False)

        try:
            plt.show()
        except KeyboardInterrupt:
            print("Stopping (Ctrl+C / Ctrl+Break received)...")
        finally:
            self._stop_capture.set()
            capture_thread.join(timeout=2.0)
            plt.close(fig)
            self.teardown()

        if self._capture_error is not None:
            print(f"ERROR during capture: {self._capture_error}", file=sys.stderr)

        print("Done.")
        return 0


def install_break_handler() -> None:
    """
    Route Windows CTRL_BREAK_EVENT into the normal KeyboardInterrupt path.

    The GUI stops a subprocess with CTRL_BREAK_EVENT (see gui/script_runner.py).
    Python maps CTRL_C_EVENT to KeyboardInterrupt on its own, but SIGBREAK
    defaults to SIG_DFL, which lets the OS terminate the process outright with
    STATUS_CONTROL_C_EXIT (0xC000013A) - no except, no finally, and therefore no
    lights-off teardown. With this installed, KeyboardInterrupt is raised
    normally, the finally blocks run, and the LEDs are commanded off on the way
    out. Same handler intrinsic_visual_stimulus.py installs, for the same reason.
    """
    handler = getattr(signal, "SIGBREAK", None)
    if handler is not None:
        signal.signal(handler, signal.default_int_handler)


def main(argv: list[str]) -> int:
    install_break_handler()
    parser = argparse.ArgumentParser(
        description="Free-running live camera preview for animal placement / focus checks."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--exposure-us", type=float, default=15000.0)
    parser.add_argument("--gain-db", type=float, default=0.0, help="Use a negative value to leave gain unchanged.")
    parser.add_argument("--pixel-format", type=str, default="Mono16", choices=["Mono8", "Mono12", "Mono16"])
    parser.add_argument("--fps", type=float, default=10.0, help="Preview frame rate cap. Use <=0 to skip AcquisitionFrameRate control.")
    parser.add_argument(
        "--snapshot-dir", type=Path, default=_DEFAULT_SNAPSHOT_DIR,
        help="Folder for snapshots saved by pressing 's' in the preview window.",
    )
    args = parser.parse_args(argv)

    cfg = PreviewConfig(
        camera_index=args.camera_index,
        exposure_us=args.exposure_us,
        gain_db=None if args.gain_db < 0 else args.gain_db,
        pixel_format=args.pixel_format,
        fps=args.fps,
        snapshot_dir=args.snapshot_dir,
    )

    preview = LivePreview(cfg)
    try:
        return preview.run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        preview.teardown()
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
