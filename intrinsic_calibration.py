#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Interactive exposure calibration utility for FLIR/Teledyne Blackfly cameras.

Workflow:
1. Open optional Arduino serial connection.
2. Command Arduino to turn on red illumination.
3. Configure the Blackfly for free-running preview acquisition with TriggerMode Off.
4. Sweep exposure times and capture preview images.
5. Let the user choose an exposure using a matplotlib slider.
6. Let the user draw a rectangular acquisition ROI.
7. Save the chosen exposure, ROI, and image statistics to disk.
8. Command Arduino to turn lights off.

Typical usage:
py -3.10 intrinsic_calibration.py --output .\captures --port COM4

Then use the printed value with your acquisition script:
py -3.10 intrinsic_imaging.py ... --exposure-us <selected value>
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RectangleSelector

try:
    import PySpin  # type: ignore
except ImportError as exc:
    raise SystemExit("PySpin is not installed or not visible to this Python environment.") from exc

try:
    import serial  # type: ignore
except ImportError as exc:
    raise SystemExit("pyserial is not installed. Install it with: py -3.10 -m pip install pyserial") from exc


class CameraConfigError(RuntimeError):
    pass


@dataclass
class CalibratorConfig:
    output: Path
    label: str = "calibration"
    port: Optional[str] = None
    baud: int = 115200
    serial_timeout_s: float = 0.1
    arduino_ready_marker: str = "ARDUINO_READY"
    red_cmd: str = "CAL_RED_ON"
    off_cmd: str = "LIGHTS_OFF"
    no_light_control: bool = False
    pixel_format: str = "Mono16"
    gain_db: Optional[float] = 0.0
    min_us: float = 1000.0
    max_us: float = 23872.7
    cal_fps: Optional[float] = 10.0
    steps: int = 25
    frames_per_exposure: int = 1
    settle_s: float = 0.05
    discard_first_frame: bool = True
    save_preview_npy: bool = True
    display_percentile_low: float = 1.0
    display_percentile_high: float = 99.0
    enable_roi_selection: bool = True
    external_trigger_calibration: bool = False
    cal_trigger_cmd: str = "CAL_TRIGGER"
    trigger_line: str = "Line0"
    trigger_activation: str = "RisingEdge"

class BlackflyExposureCalibrator:
    def __init__(self, cfg: CalibratorConfig) -> None:
        self.cfg = cfg
        self.system = None
        self.cam_list = None
        self.cam = None
        self.ser = None

        safe_label = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(cfg.label)).strip("_") or "calibration"
        timestamp = time.strftime(f"exposure_calibration_{safe_label}_%Y%m%d_%H%M%S")
        self.cal_dir = cfg.output / timestamp
        self.cal_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Arduino / serial helpers
    # -----------------------------
    def open_serial(self) -> None:
        if self.cfg.no_light_control or self.cfg.port is None:
            print("Arduino light control disabled. Turn red illumination on manually.")
            return

        self.ser = serial.Serial(
            port=self.cfg.port,
            baudrate=self.cfg.baud,
            timeout=self.cfg.serial_timeout_s,
        )

        # Most Arduinos reset when the serial port opens.
        time.sleep(2.0)
        self.ser.reset_input_buffer()

        print(f"Opened Arduino serial port {self.cfg.port} at {self.cfg.baud} baud.")
        print("Waiting briefly for Arduino ready marker...")
        self.wait_for_marker({self.cfg.arduino_ready_marker}, timeout_s=3.0, required=False)

    def close_serial(self) -> None:
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def read_serial_line(self) -> Optional[str]:
        if self.ser is None:
            return None
        line = self.ser.readline()
        if not line:
            return None
        text = line.decode("utf-8", errors="ignore").strip()
        return text or None

    @staticmethod
    def marker_from_line(line: str) -> str:
        return line.split(",", 1)[0].strip()

    def wait_for_marker(self, expected: set[str], timeout_s: float, required: bool = True) -> Optional[str]:
        if self.ser is None:
            return None

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            line = self.read_serial_line()
            if line is None:
                continue
            marker = self.marker_from_line(line)
            print(f"[arduino] {line}")
            if marker in expected:
                return marker

        if required:
            raise TimeoutError(f"Timed out waiting for Arduino marker(s): {sorted(expected)}")
        return None

    def send_arduino_command(self, command: str, expected_ack: Optional[str] = None, timeout_s: float = 2.0) -> None:
        if self.ser is None:
            return
        self.ser.write((command.strip() + "\n").encode("utf-8"))
        self.ser.flush()
        print(f"Sent Arduino command: {command}")

        if expected_ack:
            self.wait_for_marker({expected_ack}, timeout_s=timeout_s, required=False)
        else:
            # Drain any immediate serial output without requiring a specific marker.
            t_end = time.time() + 0.25
            while time.time() < t_end:
                line = self.read_serial_line()
                if line:
                    print(f"[arduino] {line}")

    # -----------------------------
    # Camera / PySpin helpers
    # -----------------------------
    def setup_camera(self) -> None:
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()

        if self.cam_list.GetSize() < 1:
            self.teardown_camera()
            raise RuntimeError("No Blackfly/Spinnaker camera detected.")

        self.cam = self.cam_list.GetByIndex(0)
        self.cam.Init()
        self.configure_camera_for_calibration()

    def teardown_camera(self) -> None:
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

    def get_float_limits(self, name: str) -> tuple[float, float, float]:
        node = PySpin.CFloatPtr(self.get_node(name))
        if not PySpin.IsReadable(node):
            raise CameraConfigError(f"Float node not readable: {name}")
        return float(node.GetMin()), float(node.GetMax()), float(node.GetValue())

    def print_float_limits(self, name: str, label: Optional[str] = None) -> None:
        label = label or name
        try:
            min_value, max_value, current_value = self.get_float_limits(name)
            print(f"  {label} limits: min={min_value:.3f}, max={max_value:.3f}, current={current_value:.3f}")
        except CameraConfigError as exc:
            print(f"  {label} limits: unavailable ({exc})")

    def set_bool(self, name: str, value: bool) -> None:
        node = PySpin.CBooleanPtr(self.get_node(name))
        if not PySpin.IsWritable(node):
            raise CameraConfigError(f"Boolean node not writable: {name}")
        node.SetValue(value)

    def get_float(self, name: str) -> float:
        node = PySpin.CFloatPtr(self.get_node(name))
        if not PySpin.IsReadable(node):
            raise CameraConfigError(f"Float node not readable: {name}")
        return float(node.GetValue())

    def get_int(self, name: str) -> int:
        node = PySpin.CIntegerPtr(self.get_node(name))
        if not PySpin.IsReadable(node):
            raise CameraConfigError(f"Integer node not readable: {name}")
        return int(node.GetValue())

    def get_int_limits(self, name: str) -> tuple[int, int, int]:
        node = PySpin.CIntegerPtr(self.get_node(name))
        if not PySpin.IsReadable(node):
            raise CameraConfigError(f"Integer node not readable: {name}")
        inc = int(node.GetInc()) if hasattr(node, "GetInc") else 1
        return int(node.GetMin()), int(node.GetMax()), max(1, inc)

    def set_int(self, name: str, value: int) -> int:
        node = PySpin.CIntegerPtr(self.get_node(name))
        if not PySpin.IsWritable(node):
            raise CameraConfigError(f"Integer node not writable: {name}")

        min_value = int(node.GetMin())
        max_value = int(node.GetMax())
        inc = int(node.GetInc()) if hasattr(node, "GetInc") else 1
        inc = max(1, inc)

        clipped = max(min_value, min(max_value, int(value)))
        aligned = min_value + ((clipped - min_value) // inc) * inc
        aligned = max(min_value, min(max_value, aligned))

        node.SetValue(aligned)
        return int(node.GetValue())

    def reset_camera_roi_to_full_frame(self) -> None:
        """Reset hardware ROI so every calibration run starts from the full sensor frame.

        FLIR/Teledyne camera ROI nodes can persist between scripts/runs. If a prior
        acquisition used Width/Height/OffsetX/OffsetY for hardware ROI capture, the
        next calibration preview may otherwise start cropped to that old ROI.
        """
        try:
            # Offsets usually need to be zero before expanding Width/Height.
            self.set_int("OffsetX", 0)
            self.set_int("OffsetY", 0)

            _, width_max, _ = self.get_int_limits("Width")
            _, height_max, _ = self.get_int_limits("Height")

            actual_width = self.set_int("Width", width_max)
            actual_height = self.set_int("Height", height_max)

            # Re-zero offsets after restoring size, in case the camera adjusted them.
            actual_x = self.set_int("OffsetX", 0)
            actual_y = self.set_int("OffsetY", 0)

            print(
                "Camera ROI reset to full frame: "
                f"x={actual_x}, y={actual_y}, width={actual_width}, height={actual_height}"
            )
        except CameraConfigError as exc:
            print(f"Warning: could not reset camera ROI to full frame: {exc}")

    def configure_camera_for_calibration(self) -> None:
        # Free-running calibration: no external trigger pulses from Arduino needed.
        self.set_enum("TriggerMode", "Off")
        self.set_enum("AcquisitionMode", "Continuous")
        self.set_enum("PixelFormat", self.cfg.pixel_format)

        # Important: clear any hardware ROI left over from a previous triggered
        # acquisition or calibration run before collecting preview frames.
        self.reset_camera_roi_to_full_frame()

        self.set_enum("ExposureAuto", "Off")
        self.set_enum("ExposureMode", "Timed")

        print("Camera limits before calibration frame-rate setting:")
        self.print_float_limits("ExposureTime", "ExposureTime")

        if self.cfg.external_trigger_calibration:
            print("External-trigger calibration enabled; skipping free-running AcquisitionFrameRate control.")

            self.set_enum("TriggerSelector", "FrameStart")
            self.set_enum("TriggerSource", self.cfg.trigger_line)
            self.set_enum("TriggerActivation", self.cfg.trigger_activation)
            self.set_enum("TriggerMode", "On")

        else:
            if self.cfg.cal_fps is not None and self.cfg.cal_fps > 0:
                try:
                    self.set_bool("AcquisitionFrameRateEnable", True)
                    actual_fps = self.set_float("AcquisitionFrameRate", self.cfg.cal_fps)
                    print(f"AcquisitionFrameRate set to {actual_fps:.3f} FPS")
                except CameraConfigError as exc:
                    print(f"Warning: Could not set AcquisitionFrameRate({exc}).")
            else:
                print("AcquisitionFrameRate control skipped because --cal-fps was set to <= 0.")

        print("Camera limits after calibration frame-rate setting:")
        self.print_float_limits("AcquisitionFrameRate", "AcquisitionFrameRate")
        self.print_float_limits("ExposureTime", "ExposureTime")

        try:
            self.set_enum("GainAuto", "Off")
            if self.cfg.gain_db is not None:
                self.set_float("Gain", self.cfg.gain_db)
        except CameraConfigError:
            print("Warning: Gain node unavailable or not writable; leaving gain unchanged.")

        print("Camera configured for exposure calibration:")
        print(f"  TriggerMode: {self.get_enum_symbolic('TriggerMode')}")
        print(f"  PixelFormat: {self.get_enum_symbolic('PixelFormat')}")
        print(f"  Width:       {self.get_int('Width')}")
        print(f"  Height:      {self.get_int('Height')}")
        try:
            print(f"  Gain:        {self.get_float('Gain'):.2f} dB")
        except CameraConfigError:
            print("  Gain:        unavailable")

    # -----------------------------
    # Calibration acquisition + UI
    # -----------------------------
    @staticmethod
    def image_stats(img: np.ndarray, exposure_us: float) -> dict:
        if np.issubdtype(img.dtype, np.integer):
            dtype_max = int(np.iinfo(img.dtype).max)
            if img.dtype == np.uint16:
                saturation_threshold = 65520
            else:
                saturation_threshold = dtype_max
            saturated_fraction = float(np.mean(img >= saturation_threshold))
        else:
            dtype_max = float(np.nanmax(img))
            saturated_fraction = 0.0

        return {
            "exposure_us": float(exposure_us),
            "dtype": str(img.dtype),
            "shape": list(img.shape),
            "min": float(np.min(img)),
            "max": float(np.max(img)),
            "mean": float(np.mean(img)),
            "p01": float(np.percentile(img, 1)),
            "p50": float(np.percentile(img, 50)),
            "p95": float(np.percentile(img, 95)),
            "p99": float(np.percentile(img, 99)),
            "dtype_max": float(dtype_max),
            "saturated_fraction": saturated_fraction,
        }

    def capture_one_frame(self, timeout_ms: int = 1000) -> np.ndarray:
        image = self.cam.GetNextImage(timeout_ms)
        try:
            if image.IsIncomplete():
                raise RuntimeError(f"Incomplete image. Status: {image.GetImageStatus()}")
            return image.GetNDArray().copy()
        finally:
            image.Release()

    def capture_one_calibration_frame(self, timeout_ms: int = 2000) -> np.ndarray:
        """Capture one calibration frame.
        
        In free-running mode, this just grabs the next camera frame.
        In external-trigger mode, this asks Arduino to emit one trigger pulse first.
        """
        if not self.cfg.external_trigger_calibration:
            return self.capture_one_frame(timeout_ms=timeout_ms)
        
        if self.ser is None:
            raise RuntimeError(
                "External-trigger calibration requires Arduino serial control. "
                "Please provide a valid --port and do not set --no-light-control."
            )
        
        self.send_arduino_command(
            self.cfg.cal_trigger_cmd,
            expected_ack=self.cfg.cal_trigger_cmd,
            timeout_s=1.0,
        )

        return self.capture_one_frame(timeout_ms=timeout_ms)

    def capture_exposure_stack(self) -> tuple[np.ndarray, list[np.ndarray], list[dict]]:
        exposure_min_allowed, exposure_max_allowed, _ = self.get_float_limits("ExposureTime")

        sweep_min_us = max(self.cfg.min_us, exposure_min_allowed)
        sweep_max_us = min(self.cfg.max_us, exposure_max_allowed)

        if sweep_max_us < self.cfg.max_us:
            print(
                f"Warning: requested --max-us {self.cfg.max_us:.1f} exceeds the camera's current "
                f"ExposureTime maximum of {exposure_max_allowed:.1f} us. "
                f"Sweeping only up to {sweep_max_us:.1f} us."
            )

        if sweep_min_us > self.cfg.min_us:
            print(
                f"Warning: requested --min-us {self.cfg.min_us:.1f} is below the camera's current "
                f"ExposureTime minimum of {exposure_min_allowed:.1f} us. "
                f"Sweeping from {sweep_min_us:.1f} us."
            )

        if sweep_max_us <= sweep_min_us:
            raise RuntimeError(
                f"Invalid exposure sweep after camera limits: min={sweep_min_us:.3f} us, "
                f"max={sweep_max_us:.3f} us. Try lowering --cal-fps or changing camera settings."
            )

        exposures_requested = np.linspace(sweep_min_us, sweep_max_us, self.cfg.steps, dtype=float)
        exposures_actual: list[float] = []
        images: list[np.ndarray] = []
        stats: list[dict] = []

        self.cam.BeginAcquisition()
        try:
            for exposure_us in exposures_requested:
                actual_us = self.set_float("ExposureTime", float(exposure_us))
                exposures_actual.append(actual_us)
                time.sleep(self.cfg.settle_s)

                if self.cfg.discard_first_frame:
                    try:
                        _ = self.capture_one_calibration_frame(timeout_ms=max(2000, int(actual_us / 1000) + 1000))
                    except Exception as exc:
                        print(f"Warning: discard frame failed at {actual_us:.1f} us: {exc}")

                frame_stack = []
                for _ in range(self.cfg.frames_per_exposure):
                    frame_stack.append(self.capture_one_calibration_frame(timeout_ms=max(2000, int(actual_us / 1000) + 1000)))

                if len(frame_stack) == 1:
                    img = frame_stack[0]
                else:
                    img = np.median(np.stack(frame_stack, axis=0), axis=0).astype(frame_stack[0].dtype)

                stat = self.image_stats(img, actual_us)
                images.append(img)
                stats.append(stat)

                print(
                    f"{actual_us:9.1f} us | "
                    f"mean={stat['mean']:9.1f} | "
                    f"p99={stat['p99']:9.1f} | "
                    f"max={stat['max']:9.1f} | "
                    f"sat={100.0 * stat['saturated_fraction']:.5f}%"
                )

        finally:
            try:
                self.cam.EndAcquisition()
            except Exception:
                pass

        return np.array(exposures_actual, dtype=float), images, stats

    @staticmethod
    def format_title(stat: dict) -> str:
        return (
            f"Exposure: {stat['exposure_us']:.1f} us | "
            f"mean: {stat['mean']:.1f} | "
            f"p99: {stat['p99']:.1f} | "
            f"max: {stat['max']:.1f} | "
            f"sat: {100.0 * stat['saturated_fraction']:.5f}%"
        )

    def choose_exposure(self, exposures: np.ndarray, images: list[np.ndarray], stats: list[dict]) -> tuple[float, dict, int]:
        selected = {"idx": len(images) // 2}

        low_name = f"p{int(self.cfg.display_percentile_low):02d}"
        high_name = f"p{int(self.cfg.display_percentile_high):02d}"

        # These keys exist for the default 1/99. Fall back gracefully for other values.
        if low_name in stats[0] and high_name in stats[0]:
            vmin = min(s[low_name] for s in stats)
            vmax = max(s[high_name] for s in stats)
        else:
            vmin = min(float(np.percentile(img, self.cfg.display_percentile_low)) for img in images)
            vmax = max(float(np.percentile(img, self.cfg.display_percentile_high)) for img in images)

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = None
            vmax = None

        fig, ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.25)

        idx0 = selected["idx"]
        im = ax.imshow(images[idx0], cmap="gray", vmin=vmin, vmax=vmax)
        ax.axis("off")
        title = ax.set_title(self.format_title(stats[idx0]), fontsize=10)

        slider_ax = plt.axes([0.18, 0.10, 0.60, 0.03])
        slider = Slider(
            ax=slider_ax,
            label="Exposure index",
            valmin=0,
            valmax=len(images) - 1,
            valinit=idx0,
            valstep=1,
        )

        button_ax = plt.axes([0.82, 0.075, 0.13, 0.075])
        lock_button = Button(button_ax, "Lock in")

        def update(_val) -> None:
            idx = int(slider.val)
            selected["idx"] = idx
            im.set_data(images[idx])
            title.set_text(self.format_title(stats[idx]))
            fig.canvas.draw_idle()

        def lock(_event) -> None:
            plt.close(fig)

        slider.on_changed(update)
        lock_button.on_clicked(lock)

        plt.show()

        idx = selected["idx"]
        return float(exposures[idx]), stats[idx], idx


    def choose_roi(self, image: np.ndarray) -> dict:
        """Let the user draw a rectangular acquisition ROI on the selected calibration image."""
        h, w = image.shape[:2]
        roi = {
            "x": 0,
            "y": 0,
            "width": int(w),
            "height": int(h),
            "full_width": int(w),
            "full_height": int(h),
            "enabled": False,
        }

        if not self.cfg.enable_roi_selection:
            return roi

        lo = float(np.percentile(image, self.cfg.display_percentile_low))
        hi = float(np.percentile(image, self.cfg.display_percentile_high))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = None
            hi = None

        fig, ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.18)
        ax.imshow(image, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title("Draw ROI around the region to capture, then click 'Lock ROI'. Click 'Use full frame' to skip ROI.")
        ax.axis("off")

        state = {"roi": roi.copy(), "has_selection": False}

        def clamp_roi(x0: float, x1: float, y0: float, y1: float) -> dict:
            xa = int(round(min(x0, x1)))
            xb = int(round(max(x0, x1)))
            ya = int(round(min(y0, y1)))
            yb = int(round(max(y0, y1)))
            xa = max(0, min(w - 1, xa))
            xb = max(xa + 1, min(w, xb))
            ya = max(0, min(h - 1, ya))
            yb = max(ya + 1, min(h, yb))
            return {
                "x": int(xa),
                "y": int(ya),
                "width": int(xb - xa),
                "height": int(yb - ya),
                "full_width": int(w),
                "full_height": int(h),
                "enabled": True,
            }

        def on_select(eclick, erelease) -> None:
            state["roi"] = clamp_roi(eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata)
            state["has_selection"] = True
            r = state["roi"]
            ax.set_title(f"ROI: x={r['x']}, y={r['y']}, width={r['width']}, height={r['height']} | Click 'Lock ROI'")
            fig.canvas.draw_idle()

        selector = RectangleSelector(
            ax,
            on_select,
            useblit=True,
            button=[1],
            minspanx=10,
            minspany=10,
            spancoords="pixels",
            interactive=True,
        )

        lock_ax = plt.axes([0.72, 0.05, 0.12, 0.07])
        full_ax = plt.axes([0.85, 0.05, 0.12, 0.07])
        lock_button = Button(lock_ax, "Lock ROI")
        full_button = Button(full_ax, "Full frame")

        def lock(_event) -> None:
            if not state["has_selection"]:
                state["roi"] = roi.copy()
            plt.close(fig)

        def full_frame(_event) -> None:
            state["roi"] = roi.copy()
            plt.close(fig)

        lock_button.on_clicked(lock)
        full_button.on_clicked(full_frame)
        plt.show()

        # Keep the selector alive until the window closes.
        _ = selector
        return state["roi"]

    # -----------------------------
    # Output
    # -----------------------------
    def save_outputs(
        self,
        selected_exposure_us: float,
        selected_stats: dict,
        selected_index: int,
        roi: dict,
        exposures: np.ndarray,
        images: list[np.ndarray],
        stats: list[dict],
    ) -> None:
        result = {
            "selected_exposure_us": float(selected_exposure_us),
            "selected_index": int(selected_index),
            "selected_stats": selected_stats,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": {**asdict(self.cfg), "output": str(self.cfg.output)},
            "calibration_dir": str(self.cal_dir),
            "selected_roi": roi,
        }

        json_path = self.cal_dir / "selected_exposure.json"
        txt_path = self.cal_dir / "selected_exposure.txt"
        roi_json_path = self.cal_dir / "selected_roi.json"
        roi_txt_path = self.cal_dir / "selected_roi.txt"
        csv_path = self.cal_dir / "exposure_sweep_stats.csv"

        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=2)

        with open(txt_path, "w", encoding="utf-8") as fp:
            fp.write(f"{selected_exposure_us:.3f}\n")

        with open(roi_json_path, "w", encoding="utf-8") as fp:
            json.dump(roi, fp, indent=2)

        with open(roi_txt_path, "w", encoding="utf-8") as fp:
            fp.write(f"x={roi['x']} y={roi['y']} width={roi['width']} height={roi['height']} enabled={roi['enabled']}\n")

        fieldnames = [
            "exposure_us", "dtype", "shape", "min", "max", "mean", "p01", "p50", "p95", "p99",
            "dtype_max", "saturated_fraction",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in stats:
                out = row.copy()
                out["shape"] = "x".join(str(x) for x in row["shape"])
                writer.writerow(out)

        if self.cfg.save_preview_npy:
            np.save(self.cal_dir / "exposures_us.npy", exposures)
            np.save(self.cal_dir / "preview_stack.npy", np.stack(images, axis=0))

        print("\nSelected exposure:")
        print(f"  {selected_exposure_us:.3f} us")
        print("\nSelected image stats:")
        print(f"  mean:      {selected_stats['mean']:.2f}")
        print(f"  p99:       {selected_stats['p99']:.2f}")
        print(f"  max:       {selected_stats['max']:.2f}")
        print(f"  saturated: {100.0 * selected_stats['saturated_fraction']:.5f}%")
        print("\nSaved:")
        print(f"  {json_path}")
        print(f"  {txt_path}")
        print(f"  {roi_json_path}")
        print(f"  {roi_txt_path}")
        print(f"  {csv_path}")
        print("\nUse this with your triggered acquisition script:")
        print(f"  --exposure-us {selected_exposure_us:.3f} --roi-config {roi_json_path}")

def _arg_value(args: argparse.Namespace, names: tuple[str, ...], default=None):
    """Return the first present argparse value from several possible argument names."""
    for name in names:
        if hasattr(args, name):
            return getattr(args, name)
    return default

def run_one_calibration_from_args(
    args: argparse.Namespace,
    *,
    label: str,
    light_cmd: str,
    enable_roi_selection: bool,
    max_us_override: Optional[float] = None,
    lights_off_on_exit: bool = True,
    keep_serial_open_after: bool = False,
) -> tuple[float, Path, Path, Optional["serial.Serial"]]:
    """Run one exposure calibration and return exposure, ROI JSON, and folder.

    max_us_override     : overrides --cal-max-us for this sweep only (e.g. a lower red ceiling).
    lights_off_on_exit  : if False, leave illumination ON after a SUCCESSFUL calibration so the LED
                          stays warm into the next phase. On any error/interrupt the light is still
                          turned off. Only meaningful with the Arduino auto-reset cap installed.
    """
    gain_db: Optional[float]
    arg_gain_db = _arg_value(args, ("gain_db",), 0.0)
    if arg_gain_db is not None and arg_gain_db < 0:
        gain_db = None
    else:
        gain_db = arg_gain_db

    cfg = CalibratorConfig(
        output=_arg_value(args, ("output",)),
        label=label,
        port=_arg_value(args, ("port",), None),
        baud=_arg_value(args, ("baud",), 115200),
        serial_timeout_s=_arg_value(args, ("serial_timeout_s",), 0.1),
        arduino_ready_marker=_arg_value(args, ("arduino_ready_marker",), "ARDUINO_READY"),
        red_cmd=light_cmd,
        off_cmd=_arg_value(args, ("lights_off_cmd", "off_cmd"), "LIGHTS_OFF"),
        no_light_control=_arg_value(args, ("no_light_control",), False),
        pixel_format=_arg_value(args, ("pixel_format",), "Mono16"),
        gain_db=gain_db,
        min_us=_arg_value(args, ("cal_min_us", "min_us"), 1000.0),
        max_us=(max_us_override if max_us_override is not None
                else _arg_value(args, ("cal_max_us", "max_us"), 40000.0)),
        cal_fps=_arg_value(args, ("cal_fps",), 10.0),
        steps=_arg_value(args, ("cal_steps", "steps"), 30),
        frames_per_exposure=_arg_value(args, ("cal_frames_per_exposure", "frames_per_exposure"), 1),
        settle_s=_arg_value(args, ("cal_settle_s", "settle_s"), 0.05),
        discard_first_frame=not _arg_value(args, ("cal_no_discard_first_frame", "no_discard_first_frame"), False),
        save_preview_npy=_arg_value(args, ("cal_save_preview_npy", "save_preview_npy"), False),
        display_percentile_low=_arg_value(args, ("cal_display_percentile_low", "display_percentile_low"), 1.0),
        display_percentile_high=_arg_value(args, ("cal_display_percentile_high", "display_percentile_high"), 99.0),
        external_trigger_calibration=_arg_value(args, ("cal_external_trigger", "external_trigger_calibration"), False),
        cal_trigger_cmd=_arg_value(args, ("cal_trigger_cmd",), "CAL_TRIGGER"),
        enable_roi_selection=enable_roi_selection,
    )

    calibrator = BlackflyExposureCalibrator(cfg)
    success = False
    result: Optional[tuple[float, Path, Path]] = None
    try:
        calibrator.open_serial()

        if calibrator.ser is not None:
            calibrator.send_arduino_command(cfg.red_cmd, expected_ack=cfg.red_cmd, timeout_s=2.0)
            print(f"{label.capitalize()} illumination command sent for calibration.")
        else:
            input(f"Turn ON {label} illumination manually, then press Enter...")

        calibrator.setup_camera()

        print("\nCapturing exposure sweep...")
        exposures, images, stats = calibrator.capture_exposure_stack()

        print("\nUse the slider to inspect exposures. Click 'Lock in' or close the window to accept the current slider value.")
        selected_exposure_us, selected_stats, selected_index = calibrator.choose_exposure(exposures, images, stats)

        if enable_roi_selection:
            print("\nDraw the acquisition ROI on the selected exposure image. Close/lock the window when done.")
        else:
            print("\nSkipping ROI selection for this calibration.")
        selected_roi = calibrator.choose_roi(images[selected_index])

        calibrator.save_outputs(
            selected_exposure_us, selected_stats, selected_index,
            selected_roi, exposures, images, stats,
        )

        result = (float(selected_exposure_us), calibrator.cal_dir / "selected_roi.json", calibrator.cal_dir)
        success = True

    finally:
        # Turn light off if requested OR if the run failed (safety on crash/Ctrl+C).
        turn_off = lights_off_on_exit or not success
        try:
            if calibrator.ser is not None and turn_off:
                calibrator.send_arduino_command(cfg.off_cmd, expected_ack=cfg.off_cmd, timeout_s=1.0)
            elif calibrator.ser is not None:
                print(f"Leaving {label} illumination ON to stay warm into imaging (requires the auto-reset cap).")
        except Exception as exc:
            print(f"Warning: failed to send lights-off command: {exc}")

        keep_open = keep_serial_open_after and success and not turn_off
        if keep_open:
            print(f"Keeping {label} calibration's serial connection open for the next phase (no reopen, no reset.)")
        else:
            calibrator.close_serial()

        calibrator.teardown_camera()

    if result is None:
        raise RuntimeError(f"{label} calibration did not complete successfully.")
    
    selected_exposure_us, roi_json_path, cal_dir = result
    return selected_exposure_us, roi_json_path, cal_dir, calibrator.ser

def run_red_green_calibration_from_args(args: argparse.Namespace, *, keep_red_on_after: bool = False) -> dict:
    red_cmd = _arg_value(args, ("cal_red_cmd", "red_cmd"), "CAL_RED_ON")
    green_cmd = _arg_value(args, ("cal_green_cmd", "green_cmd"), "CAL_GREEN_ON")
    skip_roi_selection = _arg_value(args, ("skip_roi_selection",), False)
    red_max_us = _arg_value(args, ("cal_red_max_us",), None)

    # ---- 1) Green-reference calibration FIRST (exposure only, no ROI, light off after) ----
    manual_green_exposure = _arg_value(args, ("green_exposure_us",), None)
    skip_green = _arg_value(args, ("skip_green_calibration",), False)

    if manual_green_exposure is not None:
        green_exposure_us = float(manual_green_exposure)
        green_cal_dir = None
        print(f"\nUsing manually provided green exposure: {green_exposure_us:.3f} us")
    elif skip_green:
        green_exposure_us = None
        green_cal_dir = None
        print("\nSkipping green calibration; green reference will use red/trial exposure.")
    else:
        print("\nStarting green-reference exposure calibration (running first)...")
        green_exposure_us, _green_roi_json_path, green_cal_dir, _green_ser = run_one_calibration_from_args(
            args, label="green", light_cmd=green_cmd, enable_roi_selection=False,
        )
        # _green_ser will always be None here: keep_serial_open_after defaults False,
        # so green calibration always closes its connection. Nothing to do.

    # ---- 2) Red/trial calibration LAST (exposure + ROI; optionally hands off live connection) ----
    print("\nStarting red/trial exposure + ROI calibration (running last)...")
    red_exposure_us, roi_json_path, red_cal_dir, red_ser = run_one_calibration_from_args(
        args,
        label="red",
        light_cmd=red_cmd,
        enable_roi_selection=not skip_roi_selection,
        max_us_override=red_max_us,
        lights_off_on_exit=not keep_red_on_after,
        keep_serial_open_after=keep_red_on_after,
    )

    return {
        "red_exposure_us": float(red_exposure_us),
        "green_exposure_us": None if green_exposure_us is None else float(green_exposure_us),
        "roi_json_path": roi_json_path,
        "red_cal_dir": red_cal_dir,
        "green_cal_dir": green_cal_dir,
        "serial_connection": red_ser,  # live serial.Serial if keep_red_on_after succeeded, else None
    }

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Blackfly exposure calibration script")

    parser.add_argument("--output", type=Path, required=True, help="Directory where calibration output folder will be created")
    parser.add_argument("--label", type=str, default="calibration", help="Label included in the calibration output folder name, e.g. red or green.")

    parser.add_argument("--port", type=str, default=None, help="Arduino serial port, e.g. COM4. Omit for manual light control.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--serial-timeout-s", type=float, default=0.1)
    parser.add_argument("--arduino-ready-marker", type=str, default="ARDUINO_READY")
    parser.add_argument("--red-cmd", type=str, default="CAL_RED_ON", help="Command sent to Arduino to turn red light on")
    parser.add_argument("--green-cmd", type=str, default="CAL_GREEN_ON", help="Command sent to Arduino to turn green light on")
    parser.add_argument("--skip-green-calibration", action="store_true", help="Use the red/trial exposure for green reference instead of running a separate green exposure calibration.")
    parser.add_argument("--green-exposure-us", type=float, default=None, help="Manually provide green-reference exposure and skip green exposure calibration.")
    parser.add_argument("--off-cmd", type=str, default="LIGHTS_OFF", help="Command sent to Arduino to turn lights off")
    parser.add_argument("--no-light-control", action="store_true", help="Do not send Arduino light commands")

    parser.add_argument("--pixel-format", type=str, default="Mono16", choices=["Mono8", "Mono12", "Mono16"])
    parser.add_argument("--gain-db", type=float, default=0.0, help="Gain in dB. Use --gain-db -1 to leave unchanged.")
    parser.add_argument("--min-us", type=float, default=1000.0)
    parser.add_argument("--max-us", type=float, default=23872.7)
    parser.add_argument("--cal-fps", type=float, default=10.0, help="Free-running calibration frame rate. Lower values allow longer exposures. Use <=0 to skip setting frame rate.")
    parser.add_argument("--steps", type=int, default=25, help="Number of exposures to sweep between --min-us and --max-us.")
    parser.add_argument("--frames-per-exposure", type=int, default=1)
    parser.add_argument("--settle-s", type=float, default=0.05)
    parser.add_argument("--no-discard-first-frame", action="store_true")
    parser.add_argument("--save-preview-npy", action="store_true")
    parser.add_argument("--display-percentile-low", type=float, default=1.0)
    parser.add_argument("--display-percentile-high", type=float, default=99.0)
    parser.add_argument("--skip-roi-selection", action="store_true", help="Skip drawing an acquisition ROI and use the full camera frame.")
    parser.add_argument("--external-trigger-calibration", action="store_true", help="Use Arduino trigger pulses for each calibration frame instead of free-running camera capture.")
    parser.add_argument("--cal-trigger-cmd", type=str, default="CAL_TRIGGER", help="Arduino command that emits one camera trigger pulse during calibration.")
    parser.add_argument(
        "--cal-red-max-us", type=float, default=None,
        help="Upper exposure bound override used for the red sweep only "
             "(green, and --stage both's red step, still respect this via "
             "run_red_green_calibration_from_args). Omit to just use --max-us.",
    )
    parser.add_argument(
        "--stage", type=str, default="both", choices=["both", "green", "red"],
        help="Run the full red+green sequence (both, default), or just one stage "
             "so a staged workflow can pause between them (e.g. for a live camera "
             "preview / refocus). green: exposure-only, no ROI. red: exposure + ROI.",
    )

    return parser.parse_args(argv)


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
    args = parse_args(argv)

    if args.steps < 2:
        raise SystemExit("--steps must be at least 2")
    if args.max_us <= args.min_us:
        raise SystemExit("--max-us must be greater than --min-us")
    if args.frames_per_exposure < 1:
        raise SystemExit("--frames-per-exposure must be at least 1")
    if args.display_percentile_high <= args.display_percentile_low:
        raise SystemExit("--display-percentile-high must be greater than --display-percentile-low")

    if args.stage == "green":
        green_cmd = _arg_value(args, ("cal_green_cmd", "green_cmd"), "CAL_GREEN_ON")
        green_exposure_us, _roi_json_path, green_cal_dir, _ser = run_one_calibration_from_args(
            args, label="green", light_cmd=green_cmd, enable_roi_selection=False,
        )
        print("\nGreen exposure calibration complete:")
        print(f"  green exposure_us = {green_exposure_us:.3f}")
        print(f"  calibration dir   = {green_cal_dir}")
        return 0

    if args.stage == "red":
        red_cmd = _arg_value(args, ("cal_red_cmd", "red_cmd"), "CAL_RED_ON")
        skip_roi_selection = _arg_value(args, ("skip_roi_selection",), False)
        red_max_us = _arg_value(args, ("cal_red_max_us",), None)
        red_exposure_us, roi_json_path, red_cal_dir, _ser = run_one_calibration_from_args(
            args, label="red", light_cmd=red_cmd,
            enable_roi_selection=not skip_roi_selection, max_us_override=red_max_us,
        )
        print("\nRed exposure + ROI calibration complete:")
        print(f"  red exposure_us = {red_exposure_us:.3f}")
        print(f"  roi_config      = {roi_json_path}")
        print(f"  calibration dir = {red_cal_dir}")
        return 0

    # args.stage == "both" (default) — unchanged existing behavior.
    results = run_red_green_calibration_from_args(args)

    print("\nCalibration sequence complete:")
    print(f"  red/trial exposure_us = {results['red_exposure_us']:.3f}")
    if results["green_exposure_us"] is None:
        print(f"  green exposure_us     = {results['red_exposure_us']:.3f} (using red/trial exposure)")
    else:
        print(f"  green exposure_us     = {results['green_exposure_us']:.3f}")
    print(f"  roi_config            = {results['roi_json_path']}")
    print(f"  red calibration       = {results['red_cal_dir']}")
    if results["green_cal_dir"] is not None:
        print(f"  green calibration     = {results['green_cal_dir']}")

    return 0


if __name__ == "__main__":
    # main() has no KeyboardInterrupt handler of its own; catch it here so a
    # graceful stop reports like the other entrypoints (message + 130) instead
    # of a traceback. The lights-off finally has already run by this point.
    try:
        exit_code = main(sys.argv[1:])
    except KeyboardInterrupt:
        print("Interrupted by user.")
        exit_code = 130
    raise SystemExit(exit_code)
